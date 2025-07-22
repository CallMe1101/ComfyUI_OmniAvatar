"""ComfyUI 节点：OmniAvatar All-in-One (14B)"""

import os
import shutil
import tempfile
from PIL import Image
from tqdm import tqdm

# 导入 OmniAvatar 官方推理接口
try:
    from src.inference.inference_avatar import OmniAvatar
except Exception as e:  # noqa: BLE001 - ignore import error for packaging
    OmniAvatar = None  # 运行环境未提供 OmniAvatar 源码


def _save_audio(src: str, dst: str) -> None:
    """保存或拷贝音频文件"""
    if os.path.isfile(src):
        shutil.copy(src, dst)
    else:
        with open(dst, "wb") as f:
            f.write(src)


class OmniAvatarAllInOneNode:
    """在 ComfyUI 中调用 OmniAvatar 14B 模型的节点"""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    CATEGORY = "OmniAvatar"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "portrait_image": ("IMAGE", {}),
                "audio_file": ("STRING", {"default": ""}),
                "prompt_text": ("STRING", {"default": "你好，欢迎体验OmniAvatar"}),
                "config_path": ("STRING", {"default": "configs/14B.yaml"}),
                "ckpt_path": ("STRING", {"default": "checkpoints/OmniAvatar14B.pth"}),
            }
        }

    def __init__(self):
        self._models = {}

    def _get_model(self, config_path: str, ckpt_path: str):
        """获取或创建模型实例"""
        key = f"{config_path}|{ckpt_path}"
        if key not in self._models:
            if OmniAvatar is None:  # pragma: no cover - 环境缺少源码
                raise RuntimeError("未找到 OmniAvatar 源码，请检查 PYTHONPATH")
            self._models[key] = OmniAvatar(config_path=config_path, ckpt_path=ckpt_path)
        return self._models[key]

    def run(
        self,
        portrait_image: Image.Image,
        audio_file: str,
        prompt_text: str,
        config_path: str,
        ckpt_path: str,
    ):
        """执行 OmniAvatar 推理，返回视频路径"""

        tmpdir = tempfile.mkdtemp()
        image_path = os.path.join(tmpdir, "input.png")
        audio_path = os.path.join(tmpdir, "input.wav")
        video_path = os.path.join(tmpdir, "output.mp4")

        # 保存输入图片
        Image.fromarray(portrait_image).save(image_path)
        # 保存音频
        _save_audio(audio_file, audio_path)

        # 创建或获取模型实例
        model = self._get_model(config_path, ckpt_path)

        # 调用官方推理接口并显示进度
        with tqdm(total=1, desc="OmniAvatar") as pbar:
            model.infer(
                img_path=image_path,
                audio_path=audio_path,
                save_path=video_path,
                text=prompt_text,
            )
            pbar.update(1)

        return (video_path,)


NODE_CLASS_MAPPINGS = {"OmniAvatarAllInOneNode": OmniAvatarAllInOneNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniAvatarAllInOneNode": "OmniAvatar All-in-One (14B)"}
