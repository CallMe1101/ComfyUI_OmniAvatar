# coding: utf-8
"""
OmniAvatar ComfyUI 自定义节点包
注册 OmniAvatar All-In-One 节点
"""

# 导入节点类
from .node_omniavatar_allinone import OmniAvatarAllInOneNode

# 节点映射：键为用户在 UI 中看到的名称，值为对应的类
NODE_CLASS_MAPPINGS = {
    "OmniAvatar All-in-One (14B)": OmniAvatarAllInOneNode,
}
