# Lessons Learnt（WeTok 推理 / GPU 加速排障总结）

本文记录这次把 `GrayShine/WeTok` 权重落地到本仓库、并确保 `generate_wetok.py` 可在 GPU 上稳定跑通的经验要点，方便后续复用与排障。

## 1) 推理入口应尽量“轻依赖”

- **训练栈 ≠ 推理栈**：训练常用的 `lightning`、对抗损失、LPIPS/VGG 等依赖，对纯 `encode/decode` 推理并非必需。
- 推理脚本加载模型时，**尽量避免实例化训练期 loss / discriminator**，否则会引入额外依赖、甚至触发权重下载（例如 LPIPS 的 VGG 权重），导致“脚本能跑但环境很重 / 运行被网络卡住”。

## 2) 模型代码不要反向依赖训练入口（例如 `main.py`）

- 模型文件里 `from main import instantiate_from_config` 这类写法，会把训练 CLI/Lightning 一并拉进来。
- 更稳的做法是把“配置实例化（instantiate）”抽成独立、轻量的工具模块，供训练和推理共享。

## 3) 可选依赖要做兼容层（Optional Dependency Compatibility）

- 如果推理场景不需要 `lightning`，可以提供一个极简的兼容层：
  - `lightning` 存在时使用真实的 `LightningModule`
  - 不存在时回退到 `torch.nn.Module`（并补一个最常用的 `device` 属性）
- 好处：推理脚本在“只装 PyTorch + 少量依赖”的环境也能直接运行。

## 4) 误差指标必须明确“对齐的分辨率/流程”

- `generate_wetok.py` 的 **Encode 阶段 PSNR** 是在“预处理后的模型输入分辨率”上计算（图像会按 `--size` 缩放，并被修正为可被下采样因子整除）。
- Decode 阶段默认会把结果 **resize 回原图尺寸**，因此如果直接与原图在原始分辨率上对比，误差会包含“下采样 + 上采样”的影响，PSNR 往往更低。
- 正确做法：比较误差时，确保两张图在**同一分辨率、同一色彩空间、同一数值范围（例如 [0,1]）**。

## 5) GPU OOM 属于“环境态”，需要可操作的降级策略

- 即使脚本本身没问题，GPU 上若有其它进程占用显存，也可能触发 OOM。
- 排障/缓解建议（从简单到激进）：
  - 调小 `--size`（降低模型输入分辨率）
  - 改用 CPU（虽然慢但稳定）
  - 释放 GPU 上其他进程占用的显存（确认后再处理）

