
# ComfyUI_OmniAvatar

基于 [OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) 的 ComfyUI 自定义节点  
支持输入人像图片、音频、文本提示，生成同步口型与表情的视频序列。  
节点参数和调用与OmniAvatar官方推理完全一致。

---

# OmniAvatar

This repository contains a custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) node that wraps the open-source **OmniAvatar 14B** model. The node is called **"OmniAvatar All-in-One (14B)"** and accepts a portrait image, an audio file, and a text prompt to generate a talking video.

## Files

- `__init__.py` – 暴露节点入口。
- `node_omniavatar_allinone.py` – 节点实现，直接调用 OmniAvatar 推理接口。
- `requirements.txt` – 基础依赖列表。


## 安装步骤

1. **获取 OmniAvatar 源码及模型**
   ```bash
   git clone https://github.com/Omni-Avatar/OmniAvatar
   ```
   下载 14B 权重和配置文件，并记录其路径。
2. **设置 PYTHONPATH** 将 OmniAvatar 的 `src` 目录加入环境变量：
   ```bash
   export PYTHONPATH=/path/to/OmniAvatar/src:$PYTHONPATH
   ```
3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   其余依赖请参考 OmniAvatar 仓库提供的 requirements。
4. 将本仓库放入 ComfyUI 的 `custom_nodes` 目录下。

## 使用说明

启动 ComfyUI 后，节点位于 **OmniAvatar** 分类。需要提供：

- `portrait_image`：人像图像。
- `audio_file`：音频文件路径。
- `prompt_text`：文本提示。
- `config_path`：OmniAvatar 配置文件路径。
- `ckpt_path`：模型权重路径。

节点执行时会暂存输入，并调用 `OmniAvatar.infer` 生成视频，控制台会显示进度条。



