# ComfyUI-YogurtNodes-SenseNova

使用官方 **SenseNova-U1.5-8B-MoT（RL）/ SFT** 权重，支持文生图、多参考图编辑、视觉问答和图文交错生成。图像模型、注意力、采样与显存管理复用 ComfyUI；插件补充带 KV 缓存的自回归文本解码。

## 安装与权重

需要包含 `comfy/ldm/sensenova/model.py` 的 ComfyUI 版本。插件放到 `custom_nodes/ComfyUI-YogurtNodes-SenseNova`，在 ComfyUI 环境安装 `requirements.txt` 后重启。

加载器的 `model_path` 可以填官方分片目录，或完整的单文件 Safetensors。分片目录必须包含 `model.safetensors.index.json` 及其引用的全部文件；无需复制模型，也不依赖目录中的 Hugging Face Python 代码。

本地目录示例：

```text
D:/Models/hf/sensenova--SenseNova-U1.5-8B-MoT
D:/Models/hf/sensenova--SenseNova-U1.5-8B-MoT-SFT
```

`download_model` 默认关闭。仅在主动开启且 `model_path` 留空时下载指定官方权重，保存到 ComfyUI 注册的 `SenseNovaU1` 模型目录，不使用全局 Hugging Face 缓存。可通过 `extra_model_paths.yaml` 指定：

```yaml
sensenova:
    base_path: E:/ComfyUI
    SenseNovaU1: models/SenseNovaU1/
```

此时自动查找/下载路径为 `E:/ComfyUI/models/SenseNovaU1/<model_name>/`。插件不会将专用分片放入 `checkpoints` 或 `diffusion_models`。已有本地目录始终可直接加载；也可在 `SenseNovaU1` 路径列表加入 `D:/Models/hf/`，自动查找 `sensenova--<model_name>` 子目录。

## 节点

`example_workflows` 提供可拖入 ComfyUI 的文生图、双参考图编辑、视觉问答和图文交错生成工作流。没有配置搜索路径时，先在加载器填写 `model_path`；编辑/问答工作流需选择输入图。问答与交错节点直接显示文本，且可作为流程终点执行。

- **Model Loader**：输出专用 `sensenova` 句柄，以及原生 `MODEL / CLIP / VAE`。
- **Apply LoRA**：加载官方原始键名 LoRA，克隆模型，不改变未应用 LoRA 的分支。`lora_path` 填本地 `.safetensors` 文件。
- **Generate / Edit**：无参考图即文生图；连接 `images` 即编辑/多图参考。参考图按插槽顺序排列，也接受 IMAGE batch；提示词可用 `<image>` 指定位置。
- **Visual Q&A**：输出文本，参考图可选。
- **Interleaved Generation**：输出文本、逐张 IMAGE 列表及保留图文顺序的 `content_json`；文本中的 `<image1>` 对应列表第一张图，JSON 图片索引从 0 开始。可连接 `SaveImage` 保存图片列表；`max_images=0` 仅输出文本。

正常图像推理默认 50 步、CFG 4、image CFG 1、shift 3。官方 `LoRA-8step-V2` 建议 8 步、CFG 1、image CFG 1。分辨率默认 1536；低显存机器可降低。加载器使用 ComfyUI 的卸载/动态显存策略；完整 BF16 权重约 35 GB，并非 8 GB 模型。

原生工作流也可直接使用加载器的 MODEL / CLIP / VAE，搭配 `CLIPTextEncode`、`EmptyHiDreamO1LatentImage`、`SenseNovaSamplingOptions`、采样器和 `VAEDecode`；参考图路径使用 `HiDreamO1ReferenceImages`。视觉问答和图文交错生成需要本插件句柄与完整官方 `lm_head` 权重。

插件不修改 `sys.path`，不安装或直接调用外部 `transformers` / `diffusers` 模型实现。分词器由 ComfyUI 原生加载，因此沿用 ComfyUI 已有的分词依赖。

## RL 与 SFT 的区别

两者架构、参数规模相同。SFT 是监督微调版本；不带 SFT 后缀的是在统一 SFT 后继续强化学习的正式版，优化图像美学、文字渲染、信息图和编辑等能力。日常生成/编辑优先使用 RL 正式版；SFT 可用于监督训练基线或进一步微调。SFT 并不表示体积更小或推理更快。

参考：[官方 RL 模型卡](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)、[SFT 模型卡](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-SFT)、[官方实现](https://github.com/OpenSenseNova/SenseNova-U1)、[技术报告](https://arxiv.org/html/2609.11929v1)。
