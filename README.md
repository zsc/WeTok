<div align="center">
<h1>🚀 WeTok: 面向高保真视觉重建的强大离散分词器</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.05599-b31b1b.svg)](https://arxiv.org/abs/2508.05599)
[![Github](https://img.shields.io/badge/Github-WeTok-blue)](https://github.com/zhuangshaobin/WeTok)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/GrayShine/WeTok)

</div>

## 项目背景

本项目介绍了 **WeTok**，这是一个强大的离散视觉分词器，旨在解决压缩效率与重建保真度之间长期存在的冲突。WeTok 通过引入 **分组无查找量化 (Group-Wise Lookup-Free Quantization, GQ)** 和 **生成式解码器 (Generative Decoder, GD)**，实现了最先进的重建质量，超越了以往领先的离散和连续分词器。

> <a href="https://github.com/zhuangshaobin/WeTok">WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction</a><br>
> [Shaobin Zhuang](https://scholar.google.com/citations?user=PGaDirMAAAAJ&hl=zh-CN&oi=ao), [Yiwei Guo](https://scholar.google.com/citations?user=HCAyeJIAAAAJ&hl=zh-CN&oi=ao), [Canmiao Fu](), [Zhipeng Huang](), [Zeyue Tian](https://scholar.google.com/citations?user=dghq4MQAAAAJ&hl=zh-CN&oi=ao), [Fangyikang Wang](https://scholar.google.com/citations?user=j80akcEAAAAJ&hl=zh-CN&oi=ao), [Ying Zhang](https://scholar.google.com/citations?user=R_psgxkAAAAJ&hl=zh-CN&oi=ao), [Chen Li](https://scholar.google.com/citations?hl=zh-CN&user=WDJL3gYAAAAJ), [Yali Wang](https://scholar.google.com/citations?hl=zh-CN&user=hD948dkAAAAJ)<br>

<p align="center">
  <img src="./assets/teaser.png" width="90%">
  <br>
  <em>WeTok 在重建保真度方面达到了新的最先进水平，同时提供了高压缩比。</em>
</p>

## `generate_wetok.py` 工具使用说明

本项目提供了一个集成的脚本 `generate_wetok.py`，用于执行 WeTok 的核心功能：图像编码 (Encoding) 和图像重建 (Decoding)。该脚本合并了原有的生成和重建逻辑，提供了更统一的接口，并支持根据输入文件扩展名自动判断模式。

### 快速开始（推理 / GPU）

#### 0. 最小推理依赖（推荐）

仅使用 `generate_wetok.py` 做 **编码/解码推理** 时，不需要安装完整训练栈（例如 `lightning`、LPIPS 等）。

建议安装的最小依赖：
- `torch`（有 CUDA 则自动 GPU 加速）
- `numpy`
- `Pillow`
- `omegaconf`
- `einops`

如果你要进行训练/评测，再按 `requirements.txt` 安装完整依赖即可。

#### 1. 下载默认权重（GrayShine）

仓库内提供 `download.sh` 用于下载默认权重到：

- `GrayShine/ImageNet/downsample8/WeTok.ckpt`

```bash
bash download.sh
```

#### 2. 一键编码 / 解码（使用默认 config + ckpt）

`generate_wetok.py` 已内置默认参数：
- `--config configs/WeToK/Inference/ImageNet_downsample8_imagenet.yaml`
- `--ckpt GrayShine/ImageNet/downsample8/WeTok.ckpt`
- `--size 256`

因此最简单只需要给输入输出即可：

```bash
# Encode: image -> json
python generate_wetok.py assets/teaser.png wetok_data.json

# Decode: json -> image
python generate_wetok.py wetok_data.json reconstructed.png
```

#### 3. GPU 加速说明

脚本会按如下优先级选择设备：
1) NPU（若可用）
2) CUDA（`cuda:0`，若可用）
3) CPU

通常只要安装的是 CUDA 版 PyTorch，就会自动走 GPU。

### 编码模式 (Encode)

将输入图像编码为 WeTok 的离散 Token 数据，并保存为 JSON 文件。

**命令参数：**
- `input`: 输入图像的路径。支持 jpg, jpeg, png, bmp, webp, tiff, avif。
- `output`: 输出 JSON 文件的路径。
- `--config`: 模型配置文件 (.yaml) 的路径。
- `--ckpt`: 模型权重文件 (.ckpt) 的路径。
- `--size`: (可选) 图像处理尺寸，默认为 256。
- `--mode`: (可选) 显式指定为 `encode`。通常可自动检测。

**示例：**

```bash
python generate_wetok.py \
    assets/teaser.png \
    wetok_data.json \
    --config configs/WeToK/Inference/ImageNet_downsample8_imagenet.yaml \
    --ckpt GrayShine/ImageNet/downsample8/WeTok.ckpt \
    --size 256
```

### 解码模式 (Decode)

读取包含 WeTok Token 数据的 JSON 文件，并将其重建为图像。

**命令参数：**
- `input`: 输入 JSON 文件的路径（通常由编码模式生成）。
- `output`: 重建后输出图像的路径。
- `--config`: (可选) 模型配置文件路径。如果 JSON 中记录的路径有效，则无需指定。
- `--ckpt`: (可选) 模型权重文件路径。如果 JSON 中记录的路径有效，则无需指定。
- `--mode`: (可选) 显式指定为 `decode`。通常可自动检测。

**示例：**

```bash
python generate_wetok.py \
    wetok_data.json \
    reconstructed_image.png
```

### 误差指标（PSNR / MSE / MAE）怎么理解？

- Encode 阶段会打印 `Reconstruction PSNR`，它是在“**预处理后的模型输入分辨率**”上计算的（图像会按 `--size` 缩放，并被修正为可被模型下采样因子整除）。
- Decode 阶段默认会把输出 **resize 回原图尺寸**（如果 JSON 里记录了原图尺寸），因此你如果直接在“原始分辨率”上对比原图与重建图，误差会包含额外的缩放影响，PSNR 往往会更低。

示例（`assets/teaser.png`，`--size 256`，模型输入被缩放到 `256x576`）：
- Encode 阶段（脚本输出）：PSNR ≈ `23.86 dB`
- Encode→JSON→Decode 后输出 PNG vs 原图（`918x2058`）：PSNR ≈ `17.7074 dB`（MSE `0.01695365`，MAE `0.05928486`，像素归一化到 `[0,1]`）
- 若把“原图”和“重建 PNG”都 resize 到 `256x576` 再对比：PSNR ≈ `23.9602 dB`（MSE `0.00401776`，MAE `0.03134260`，像素归一化到 `[0,1]`）

## 依赖环境

仅做推理建议走“最小依赖”；如需训练/评测，请按 `requirements.txt` 安装完整依赖环境。
