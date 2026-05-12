# YOLOE ONNX Export & PTQ Quantization

English | [中文](./README_cn.md)

An ONNX export and post-training quantization (PTQ) toolchain based on [Ultralytics YOLOE](https://github.com/ultralytics/ultralytics), supporting YOLOE open-vocabulary detection/segmentation model export to ONNX format with Horizon Robotics HBDK int16 quantization for edge deployment.

> Repository: https://github.com/zixi01chen/ultralytics

## Installation

```bash
git clone https://github.com/zixi01chen/ultralytics.git && cd ultralytics
pip install -e .
pip install onnx onnxruntime
```

The Horizon Robotics HBDK/HBRuntime toolchain (`horizon_tc_ui`) is required for quantization and inference.

## Model Preparation

Download the pretrained YOLOE checkpoint to the project root:

```bash
# Download YOLOE-26n-seg model
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt -O yoloe-26n-seg.pt
```

> A pre-exported ONNX file `yoloe-26n-seg.onnx` and text model `mobileclip2_b.ts` are already included. Re-run step ① if you need to re-export.

## Pipeline Overview

```
① export_onnx_text_prompt.py              → Export ONNX model (opset 11, no NMS)
        ↓
② test_onnx_test_prompt.py                → Floating-point ONNX Runtime validation
        ↓
③ hb_compile + yoloe26n_int16_config.yaml → PTQ int16 quantization (generates .bc)
        ↓
④ HBRuntime inference                     → Quantized model accuracy validation
```

## Usage

The workflow consists of three steps, executed in order:

### ① Export ONNX Model

Export YOLOE to ONNX format (opset 11, end2end disabled to remove postprocessing ops):

```bash
python export_onnx_text_prompt.py
```

Outputs:

| File | Description |
|---|---|
| `yoloe-26n-seg.onnx` | YOLOE-26n segmentation model. Input `images` (1,3,640,640), output `output0` (1,116,8400) with bbox+scores+mask_coeffs and `output1` (1,32,160,160) mask prototypes |

> Setting `model.model.end2end = False` removes ArgMax/TopK/GatherElements/Concat from the ONNX graph. Without this, the quantized model's Concat layer merges boxes (range [0,640]) and scores (range [0,1]) with the same int8 threshold, destroying score precision.

### ② Floating-Point ONNX Validation

Run inference with ONNX Runtime to validate the exported model:

```bash
python test_onnx_test_prompt.py
```

This script performs:
- Image preprocessing (LetterBox resize → RGB → normalize)
- ONNX Runtime inference (supports CUDA/CPU)
- NMS postprocessing + mask decoding
- Visualized output saved to `onnx_without_nms_output.jpg`

> The default test image is `bus.jpg`. Change the `img_path` variable in the script to use a different image.

### ③ PTQ Quantization

#### RDK X5 Quantization

Run int16 quantization inside the RDK X5 OE>=1.2.8 docker environment.

Calibration uses 50 COCO val2014 images (pre-exported as RGB float32 format in `quantification/calibration_data_rgb_f32/`).

```shell
hb_mapper makertbin --config quantification/yoloe26n_int16_config.yaml --model-type onnx
```

Key quantization settings:

| Parameter | Value | Description |
|---|---|---|
| `march` | `bayes-e` | Target BPU architecture |
| `input_type_rt` | `nv12` | Runtime input format (matches hardware encoder) |
| `input_type_train` | `rgb` | Training/floating-point model input format |
| `norm_type` | `data_scale` | Preprocessing: pixel × 1/255 |
| `scale_value` | `0.003921568627451` | Normalization coefficient |
| `optimization` | `set_all_nodes_int16` | Full network int16 quantization |
| `compile_mode` | `latency` | Compilation target: latency-optimized |
| `optimize_level` | `O3` | Maximum compiler optimization |

Outputs:

| File | Description |
|---|---|
| `yoloe26n_seg_output/yoloe26n_seg_640x640_nv12_int16.bc` | Horizon BPU-ready quantized model |

### ④ Quantized Model Inference

Load the quantized `.bc` model with HBRuntime for inference, comparing cosine similarity against floating-point ONNX Runtime results to validate quantization accuracy.

## License

Based on Ultralytics code, licensed under [AGPL-3.0](./LICENSE).
