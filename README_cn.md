# YOLOE ONNX 导出与 PTQ 量化

[English](./README.md) | 中文

基于 [Ultralytics YOLOE](https://github.com/ultralytics/ultralytics) 的 ONNX 导出与后训练量化（PTQ）工具链，支持将 YOLOE 开放词汇检测/分割模型导出为 ONNX 格式，并使用地平线 HBDK 工具链进行 int16 量化部署。

> 项目地址：https://github.com/zixi01chen/ultralytics

## 环境安装

```bash
git clone https://github.com/zixi01chen/ultralytics.git && cd ultralytics
pip install -e .
pip install onnx onnxruntime
```

## 模型准备

下载预训练 YOLOE checkpoint 放置到项目根目录：

```bash
# 下载 YOLOE-26n-seg 模型
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt -O yoloe-26n-seg.pt
```

> 本项目已包含预导出的 ONNX 文件 `yoloe-26n-seg.onnx` 和文本模型 `mobileclip2_b.ts`。如需重新导出，请按步骤①操作。

## 完整流程说明

```
① export_onnx_text_prompt.py              → 导出 ONNX 模型 (opset 11, 无 NMS)
        ↓
② test_onnx_test_prompt.py                → 浮点 ONNX Runtime 推理验证
        ↓
③ hb_compile + yoloe26n_int16_config.yaml → PTQ int16 量化（生成 .bc）
        ↓
④ HBRuntime 推理                          → 量化模型精度验证
```

## 使用流程

整个工作流程分为三步，按顺序执行：

### ① 导出 ONNX 模型

将 YOLOE 导出为 ONNX 格式（opset 11，关闭 end2end 以移除后处理算子）：

```bash
python export_onnx_text_prompt.py
```

生成产物：

| 文件 | 说明 |
|---|---|
| `yoloe-26n-seg.onnx` | YOLOE-26n 分割模型 ONNX 导出。输入 `images` (1,3,640,640)，输出 `output0` (1,116,8400) 包含 bbox+score+mask_coeff、`output1` (1,32,160,160) mask prototypes |

> 导出时设置 `model.model.end2end = False`，移除 ArgMax/TopK/GatherElements/Concat 等后处理算子，避免量化时 Concat 层将 bbox（范围 [0,640]）与 score（范围 [0,1]）用同一 int8 阈值量化导致精度损失。

### ② 浮点 ONNX 推理验证

用 ONNX Runtime 运行推理，验证导出模型的精度：

```bash
python test_onnx_test_prompt.py
```

该脚本完成：
- 图像预处理（LetterBox resize → RGB → normalize）
- ONNX Runtime 推理（支持 CUDA/CPU）
- NMS 后处理 + mask 解码
- 输出可视化结果 `onnx_without_nms_output.jpg`

> 测试图像默认为 `bus.jpg`，可修改脚本中的 `img_path` 变量更换测试图像。

### ③ PTQ 量化

#### RDK X5 量化

在 RDK X5 工具链 OE>=1.2.8 docker 中进行 int16 量化。

校准数据使用 50 张 COCO val2014 图像（已导出为 RGB float32 格式，存放于 `quantification/calibration_data_rgb_f32/`）。

```shell
hb_mapper makertbin --config quantification/yoloe26n_int16_config.yaml --model-type onnx
```

量化配置要点：

| 参数 | 值 | 说明 |
|---|---|---|
| `march` | `bayes-e` | 目标 BPU 架构 |
| `input_type_rt` | `nv12` | 运行时输入格式（匹配硬件编码要求） |
| `input_type_train` | `rgb` | 训练/浮点模型输入格式 |
| `norm_type` | `data_scale` | 预处理：像素值 × 1/255 |
| `scale_value` | `0.003921568627451` | 归一化系数 |
| `optimization` | `set_all_nodes_int16` | 全网络 int16 量化 |
| `compile_mode` | `latency` | 编译优化目标：延迟优先 |
| `optimize_level` | `O3` | 最高编译优化等级 |

生成产物：

| 文件 | 说明 |
|---|---|
| `yoloe26n_seg_output/yoloe26n_seg_640x640_nv12_int16.bc` | 地平线 BPU 可执行的量化模型 |

### ④ 量化模型推理

使用 HBRuntime 加载量化后的 `.bc` 模型进行推理，与浮点 ONNX Runtime 结果对比 cosine similarity，验证量化精度损失。

## License

基于 Ultralytics 代码，遵循 [AGPL-3.0](./LICENSE) 许可。
