from ultralytics import YOLOE

# Load YOLOE model
model = YOLOE("yoloe-26n-seg.pt")

# Set classes
model.set_classes([
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
])

# Disable end2end to remove ArgMax/TopK/GatherElements/Concat_5 from the ONNX graph.
# Without this, the quantized model's Concat_5 merges boxes (range [0,640]) and
# scores (range [0,1]) with the same int8 threshold, destroying score precision.
# Must set on model.model (DetectionModel), not on YOLOE wrapper.
# YOLOE wrapper's __setattr__ just stores in __dict__, but the detection head's
# forward() reads self.end2end from the head itself.
model.model.end2end = False

# Export ONNX without NMS postprocessing
# output0: (1, 4+nc+nm, anchors) = (1, 116, 8400) — raw boxes + class scores + mask coefficients
# output1: (1, 32, 160, 160) — mask prototypes
model.export(format="onnx", opset=11)
