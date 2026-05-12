import cv2
import numpy as np
import onnxruntime as ort
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
model_path = "./yoloe-26n-seg.onnx"  # exported without NMS (end2end=False)
orig_img = cv2.imread(img_path)  # BGR, HWC, 0-255

# Preprocess: letterbox resize, BGR->RGB, HWC->CHW, normalize
letterbox = LetterBox(new_shape=(IMGSZ, IMGSZ), stride=32)
img = letterbox(image=orig_img)
img = img[..., ::-1]  # BGR to RGB
img = img.transpose(2, 0, 1)  # HWC to CHW
img = np.ascontiguousarray(img)
img = img.astype(np.float32) / 255.0  # normalize to 0-1
img = img[None]  # (1, 3, 640, 640)

# ONNX inference
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
outputs = session.run(None, {"images": img})
det_out = torch.from_numpy(outputs[0])  # (1, 116, 8400) BCN: boxes(4)+scores(80)+masks(32)
proto = torch.from_numpy(outputs[1])    # (1, 32, 160, 160)

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

# Build Results and save rendered image
result = Results(orig_img, path=img_path, names=CLASS_NAMES, boxes=pred[:, :6], masks=masks)
rendered = result.plot()
cv2.imwrite("onnx_without_nms_output.jpg", rendered)
print(f"Saved onnx_without_nms_output.jpg with {pred.shape[0]} detections")
if pred.shape[0] > 0:
    for i in range(pred.shape[0]):
        cls_name = CLASS_NAMES.get(int(pred[i, 5].item()), "?")
        print(f"  [{cls_name}] conf={pred[i, 4]:.4f} box={pred[i, :4].tolist()}")
