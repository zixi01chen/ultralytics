# Changelog

Notable changes to the YOLOE ONNX export & PTQ quantization fork.

## [0.1.0] — 2026-05-12

Base: [ultralytics v8.4.48](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.48)

### Added

- **ONNX export script** (`export_onnx_text_prompt.py`): Export YOLOE-26n-seg to ONNX format with opset 11, end2end disabled to remove postprocessing ops (ArgMax/TopK/GatherElements/Concat) that would degrade quantization precision.
- **Floating-point validation script** (`test_onnx_test_prompt.py`): ONNX Runtime inference with full preprocessing (LetterBox → RGB → normalize), NMS postprocessing, mask decoding, and visualization output.
- **Quantized inference script** (`test_quantionnx_test_prompt.py`): HB_ONNXRuntime inference with diagnostic logging (per-channel value range checks for boxes/scores/masks), NHWC NV12-compatible preprocessing, and visualization output.
- **Horizon HBDK quantization config** (`quantification/yoloe26n_int16_config.yaml`): RDK X5 (bayes-e) int16 quantization configuration with NV12 runtime input, rgb training input, data_scale normalization (1/255), latency-optimized compilation at O3.
- **Calibration dataset** (`quantification/calibration_data_rgb_f32/`): 50 COCO val2014 images exported as raw RGB float32 binary files for PTQ calibration.
- **Pre-exported ONNX model** (`yoloe-26n-seg.onnx`): YOLOE-26n-seg exported with end2end=False, opset 11.
- **Text model** (`mobileclip2_b.ts`): MobileCLIP2-B TorchScript model for YOLOE text prompt encoding.
- **Bilingual README**: Project documentation in both English (`README.md`) and Chinese (`README_cn.md`).

### Changed

- Replaced upstream Ultralytics README with ONNX export & PTQ focused documentation.
