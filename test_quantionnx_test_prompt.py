import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.engine.results import Results

CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

CONF_THRES = 0.25
IOU_THRES = 0.45
IMGSZ = 640
NC = 80  # number of classes (must match export)
NM = 32  # number of mask coefficients

# Load image
img_path = "./ultralytics/assets/bus.jpg"
model_path = "./yoloe26n_seg_output/yoloe26n_seg_640x640_nv12_quantized_model.onnx"

orig_img = cv2.imread(img_path)  # BGR, HWC, uint8

# Preprocess: letterbox resize, keep BGR uint8 NHWC
letterbox = LetterBox(new_shape=(IMGSZ, IMGSZ), stride=32)
img = letterbox(image=orig_img)
img = img[None]  # (1, 640, 640, 3) NHWC

# ONNX inference with HB_ONNXRuntime (quantized model)
from horizon_tc_ui import HB_ONNXRuntime

sess = HB_ONNXRuntime(model_file=model_path)
input_names = [input.name for input in sess.get_inputs()]
output_names = [output.name for output in sess.get_outputs()]
outputs = sess.run(output_names, {input_names[0]: img})
# outputs = sess.run(output_names, {input_names[0]: img, input_names[1]: np.array([1.0], dtype=np.float32), input_names[2]: np.array([1.0], dtype=np.float32)})

det_out = torch.from_numpy(outputs[0])  # (1, 116, 8400) BCN: boxes(4)+scores(80)+masks(32)
proto = torch.from_numpy(outputs[1])    # (1, 32, 160, 160)

# Diagnostic: check raw output values and dtypes
print(f"det_out dtype: {det_out.dtype}, shape: {det_out.shape}")
print(f"proto dtype: {proto.dtype}, shape: {proto.shape}")

# Per-channel statistics to verify value ranges
boxes = det_out[0, :4]       # (4, 8400) decoded xywh, should be in [0, 640]
scores = det_out[0, 4:84]    # (80, 8400) class scores, should be in [0, 1] (sigmoided)
mask_coeffs = det_out[0, 84:]  # (32, 8400) mask coefficients
print(f"boxes  — min: {boxes.min().item():.3f}, max: {boxes.max().item():.3f}, mean: {boxes.mean().item():.3f}")
print(f"scores — min: {scores.min().item():.6f}, max: {scores.max().item():.6f}, mean: {scores.mean().item():.6f}")
print(f"masks  — min: {mask_coeffs.min().item():.4f}, max: {mask_coeffs.max().item():.4f}, mean: {mask_coeffs.mean().item():.4f}")
print(f"proto  — min: {proto.min().item():.4f}, max: {proto.max().item():.4f}, mean: {proto.mean().item():.4f}")

# NMS: standard BCN format, boxes are decoded xywh, nc=80
# NMS will: xywh→xyxy, class confidence filtering, IoU NMS
# Returns per-image tensor of shape (N, 6+NM): x1,y1,x2,y2,conf,cls + mask_coeffs
preds = non_max_suppression(det_out, CONF_THRES, IOU_THRES, nc=NC, max_det=300)
pred = preds[0]

# Process masks BEFORE scaling boxes (boxes still in 640x640 model space)
if pred.shape[0] > 0:
    masks = ops.process_mask(proto[0], pred[:, 6:], pred[:, :4], (IMGSZ, IMGSZ), upsample=True)
    keep = masks.amax((-2, -1)) > 0
    if not all(keep):
        pred, masks = pred[keep], masks[keep]
    # Scale boxes from model input size (640x640) to original image size
    pred[:, :4] = ops.scale_boxes((IMGSZ, IMGSZ), pred[:, :4], orig_img.shape[:2])
else:
    masks = None

# Build Results and save
result = Results(orig_img, path=img_path, names=CLASS_NAMES, boxes=pred[:, :6], masks=masks)
rendered = result.plot()
cv2.imwrite("quantized_onnx_without_nms_output.jpg", rendered)
print(f"\nSaved quantized_onnx_without_nms_output.jpg with {pred.shape[0]} detections")
if pred.shape[0] > 0:
    for i in range(pred.shape[0]):
        cls_name = CLASS_NAMES.get(int(pred[i, 5].item()), "?")
        print(f"  [{cls_name}] conf={pred[i, 4]:.4f} box={pred[i, :4].tolist()}")
