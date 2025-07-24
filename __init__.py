# coding: utf-8
"""
OmniAvatar ComfyUI 自定义节点包注册文件

说明：
- 本文件放置于：
    F:\Comfyui\custom_nodes\ComfyUI_OmniAvatar\__init__.py
- 用于注册自定义节点，让 ComfyUI 启动时能识别 “OmniAvatar All-in-One (14B)” 节点。

使用：
- 启动或重启 ComfyUI 后，在节点列表会出现该节点。
"""

# 从同目录下导入节点实现
from .node_omniavatar_allinone import OmniAvatarAllInOneNode

# 将 UI 显示名称映射到实现类
NODE_CLASS_MAPPINGS = {
    "OmniAvatar All-in-One (14B)": OmniAvatarAllInOneNode,
}
