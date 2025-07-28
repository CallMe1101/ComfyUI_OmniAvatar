# coding: utf-8
"""
ComfyUI 自定义节点：OmniAvatar All-in-One (14B)

功能概述：
1. 接收静态人像（IMAGE）、音频流（AUDIO）与文本提示（STRING）。
2. 自动将各种形式的音频（内存 buffer、MP3/WAV 文件等）转换为推理所需的 WAV。
3. 调用 GitHub 原项目的 OmniAvatar 推理脚本，生成同步口型与表情的视频。
4. 从生成的视频中提取帧，输出为 IMAGE 张量序列；同时封装为 ComfyUI 标准 VIDEO 对象，
   可无缝连接 GetVideoComponents、SaveVideo、VideoViewer 等下游节点。
custom_nodes/
├─ ComfyUI_OmniAvatar/
│   ├─ __init__.py
│   └─ node_omniavatar_allinone.py
└─ OmniAvatar/
    ├─ configs/
    │   ├─ inference.yaml
    │   └─ inference_1.3B.yaml
    ├─ OmniAvatar/      # 原项目源码
    └─ scripts/
        └─ inference.py
放置路径：
F:\Comfyui\custom_nodes\ComfyUI_OmniAvatar\node_omniavatar_allinone.py
"""

import os
import io
import tempfile
import random

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

# ComfyUI 内置视频类型封装
from comfyui.node_types.video import VideoComponents, VideoFromComponents

# 用于读写 WAV，以及音频重采样
import soundfile as sf
import librosa

# === 导入并初始化原项目 OmniAvatar 模型接口 ===
# 原项目脚本位于：F:\Comfyui\custom_nodes\OmniAvatar\scripts\inference.py
# 需要将上级 OmniAvatar 目录加入 sys.path 才能导入
import sys
# 获取本文件目录：.../custom_nodes/ComfyUI_OmniAvatar
_this_dir = os.path.dirname(__file__)
# 上级 custom_nodes 目录
_parent = os.path.abspath(os.path.join(_this_dir, os.pardir))
# 将原项目核心目录加入搜索路径：.../custom_nodes/OmniAvatar
sys.path.insert(0, os.path.join(_parent, "OmniAvatar"))
try:
    # inference.py 中定义了 WanInferencePipeline
    from scripts.inference import WanInferencePipeline
    from utils.args_config import parse_args
except Exception as e:
    WanInferencePipeline = None  # 如导入失败，稍后会抛错提示


class OmniAvatarAllInOneNode:
    """
    ComfyUI 节点：OmniAvatar All-in-One (14B)
    -----------------------------
    输入：
      portrait : IMAGE — 静态人脸图 (H×W×3 uint8)
      audio    : AUDIO — ComfyUI 音频 buffer (np.ndarray, sr) / torch.Tensor / bytes
      prompt   : STRING — 文本提示
    可调参数：
      sample_steps            — 推理采样步数
      sample_text_guide_scale — 文本引导强度
      sample_audio_guide_scale— 音频引导强度
      teacache_thresh         — TeaCache 一级缓存阈值
      seed                    — 随机种子
      control_after_generate  — 控制模式：fixed/dynamic
    输出：
      frames : IMAGE — 张量列表 (F, C, H, W)，像素归一化到 [0,1]
      video  : VIDEO — ComfyUI 标准视频对象，包含帧与音频
    """

    CATEGORY = "OmniAvatar"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("frames", "video")

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义 UI 面板上的输入接口和调节参数：
        - portrait: ComfyUI IMAGE
        - audio   : ComfyUI AUDIO
        - prompt  : STRING
        - 其余为滑块或下拉菜单，可选
        """
        return {
            "required": {
                "portrait": ("IMAGE", {}),
                "audio":   ("AUDIO", {}),
                "prompt":  ("STRING", {
                    "default": "你好，欢迎体验 OmniAvatar",
                    "multiline": True
                }),
            },
            "optional": {
                "sample_steps": ("INT", {
                    "default": 20, "min": 1, "max": 100, "step": 1
                }),
                "sample_text_guide_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1
                }),
                "sample_audio_guide_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1
                }),
                "teacache_thresh": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "seed": ("INT", {
                    "default": 91123456789, "min": 0, "max": 2**64 - 1, "step": 1
                }),
                "control_after_generate": ("OPTION", {
                    "default": "fixed",
                    "options": ["fixed", "dynamic"]
                }),
            }
        }

    def __init__(self):
        """
        延迟初始化模型实例：
        防止 ComfyUI 启动时因模型加载过慢而卡顿
        """
        self._pipeline = None
        self._args = None

    def _ensure_pipeline(self):
        """
        延迟创建并缓存 WanInferencePipeline 实例
        - 从 config 文件读取参数（已在 scripts/inference.py 内部实现）
        - parse_args 会自动加载同目录下的 configs/inference.yaml
        """
        if self._pipeline is None:
            if WanInferencePipeline is None:
                raise RuntimeError(
                    "无法导入 OmniAvatar 推理接口，请检查 custom_nodes/OmniAvatar/scripts 路径"
                )
            # 解析 args，默认会使用：OmniAvatar/configs/inference.yaml
            self._args = parse_args()
            # 创建推理管线对象
            self._pipeline = WanInferencePipeline(self._args)
        return self._pipeline, self._args

    def run(
        self,
        portrait: np.ndarray,
        audio,
        prompt: str,
        sample_steps: int = 20,
        sample_text_guide_scale: float = 3.0,
        sample_audio_guide_scale: float = 3.0,
        teacache_thresh: float = 0.20,
        seed: int = 42,
        control_after_generate: str = "fixed",
    ):
        """
        主流程：
        1. 随机种子设定
        2. 保存 portrait & audio 为临时文件
        3. 调用 WanInferencePipeline.log_video 生成视频数组
        4. 写出临时 MP4，并封装帧与音频输出
        """

        # ─── 0. 随机种子 ─────────────────────────────────────
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ─── 1. 创建临时目录 ─────────────────────────────────
        tmpdir = tempfile.mkdtemp(prefix="omniavatar_")
        image_path = os.path.join(tmpdir, "input.png")
        audio_path = os.path.join(tmpdir, "input.wav")
        video_path = os.path.join(tmpdir, "output.mp4")

        # ─── 2. 保存人像 ─────────────────────────────────────
        # portrait 为 (H,W,3) uint8 数组
        img = Image.fromarray(portrait.astype("uint8"))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(image_path)

        # ─── 3. 处理音频并保存 WAV ────────────────────────────
        # 支持多种形式：tuple/torch.Tensor/bytes/文件路径
        if isinstance(audio, tuple) and len(audio) == 2:
            samples, sr = audio
            sf.write(audio_path, samples, sr)
        elif isinstance(audio, torch.Tensor):
            arr = audio.cpu().numpy()
            sf.write(audio_path, arr, 16000)
        elif isinstance(audio, (bytes, bytearray)):
            data, sr = sf.read(io.BytesIO(audio))
            sf.write(audio_path, data, sr)
        else:
            # 假设 audio 是文件路径
            data, _ = librosa.load(audio, sr=16000)
            sf.write(audio_path, data, 16000)

        # ─── 4. 获取推理管线并覆盖参数 ─────────────────────────
        pipeline, args = self._ensure_pipeline()
        args.num_steps = sample_steps
        args.guidance_scale = sample_text_guide_scale
        args.audio_scale = sample_audio_guide_scale
        args.tea_cache_l1_thresh = teacache_thresh
        args.control_after_generate = control_after_generate

        # ─── 5. 执行模型推理 ─────────────────────────────────
        # 调用原项目中的 log_video 生成 numpy 视频数组 (T, H, W, 3)
        with tqdm(total=1, desc="OmniAvatar 推理") as pbar:
            video_np = pipeline.log_video(
                image_path=image_path,
                audio_path=audio_path,
                prompt=prompt
            )
            pbar.update(1)

        # ─── 6. 将视频数组保存为 MP4 ───────────────────────────
        # 原项目提供 save_video_as_grid_and_mp4 工具也可直接使用
        from scripts.inference import save_video_as_grid_and_mp4
        save_video_as_grid_and_mp4(
            video_np,
            out_path=video_path,
            fps=args.fps,
            audio_path=audio_path
        )

        # ─── 7. 提取帧并构建张量 ───────────────────────────────
        import imageio
        reader = imageio.get_reader(video_path, "ffmpeg")
        frames = []
        for frame in reader:  # frame: H,W,3 uint8
            t = torch.from_numpy(frame.astype(np.float32) / 255.0)  # 归一化
            t = t.permute(2, 0, 1)  # C,H,W
            frames.append(t)
        reader.close()

        if not frames:
            raise RuntimeError("未从生成视频中提取到任何帧，请检查推理是否成功。")
        frames_tensor = torch.stack(frames, dim=0)  # (F, C, H, W)

        # ─── 8. 封装为 ComfyUI VIDEO 对象 ───────────────────────
        vc = VideoComponents(
            frames=frames_tensor,
            audio_path=audio_path,
            fps=args.fps
        )
        video_obj = VideoFromComponents(vc)

        # ─── 9. 返回 IMAGE 张量序列 与 VIDEO 对象 ─────────────
        return (frames_tensor, video_obj)

NODE_CLASS_MAPPINGS = {"OmniAvatarAllInOneNode": OmniAvatarAllInOneNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniAvatarAllInOneNode": "OmniAvatar All-in-One (14B)"}
