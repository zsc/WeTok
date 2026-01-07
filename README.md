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

### 1. 编码模式 (Encode)

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
    --config configs/WeToK/Inference/GeneralDomain_compratio192_imagenet.yaml \
    --ckpt GrayShine/ImageNet/WeTok.ckpt
```

### 2. 解码模式 (Decode)

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

## 依赖环境

在运行脚本之前，请确保已按照 `env.sh` 安装了所需的依赖环境：

```bash
bash env.sh
```
