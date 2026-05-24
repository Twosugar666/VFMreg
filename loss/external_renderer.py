"""
外部渲染器适配（参考实现：DiffRegistration / NormalRendering / sim.py）
==================================================================
作用：
    将 toukui/DiffRegistration 项目中基于 PyVista 的**非可微**正向渲染器
    包装为本项目第5章 VFMReg 合成数据训练管线中的 **GT mask / RGB 生成器**。

设计理由：
    论文第5章 Stage-1 (合成数据预训练) 中需要：
      - 软渲染轮廓 P̂  (来自我们的可微渲染器，参与 L_iou / L_silhouette 反传)
      - 目标 mask M*  (来自 GT 几何，**不需要可微**，仅作为监督信号)
    DiffRegistration/NormalRendering/sim.py 正好提供了
    "给定 mesh + 内外参 -> 渲染 RGB / mask" 的快速正向流程，可作为
    GT 侧（M*, RGB*）的离线/在线生成工具。

与 render_losses.VFMRegRenderLoss 的对应关系：
    +-------------------------------------+--------------------------+
    | sim.Simulation3D.GenerateImages     |  produces  -> target_rgb |
    | sim.Simulation3D (mesh silhouette)  |  produces  -> target_mask|
    | (本项目可微渲染器, 训练中)          |  produces  -> rendered_* |
    | render_losses.VFMRegRenderLoss      |  consume both pairs      |
    +-------------------------------------+--------------------------+

注意：
    PyVista 仅在有 GPU+OpenGL/EGL 的图形环境下可正常运行；本文件做了延迟导入，
    在无图形环境的机器上可以仍然 `import` 本模块（不会报错），只有真正
    调用 render() 时才会触发 PyVista 加载。
"""

from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 路径注入：把 DiffRegistration 仓库加入 sys.path，方便复用其 sim.py
# ----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DIFFREG_ROOT = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "toukui", "DiffRegistration")
)
_NORMAL_RENDER_DIR = os.path.join(_DIFFREG_ROOT, "NormalRendering")
if os.path.isdir(_NORMAL_RENDER_DIR) and _NORMAL_RENDER_DIR not in sys.path:
    sys.path.insert(0, _NORMAL_RENDER_DIR)


# ======================================================================
# 配置数据类
# ======================================================================
@dataclass
class RendererConfig:
    """外部渲染器配置（对齐 sim.Simulation3D 构造参数）"""

    obj_head_path: str                       # 头部 mesh 文件路径 (.obj)
    obj_helmet_path: str                     # 头盔/标记物 mesh
    texture_path: str                        # 棋盘格贴图路径（侧面）
    texture_path_center: str                 # 棋盘格贴图（中央）
    image_size: Tuple[int, int] = (1024, 768)
    background_color: Tuple[int, int, int] = (28, 40, 51)
    rotation_range_rad: float = 1 * math.pi / 180   # 论文第5.3节, 域随机化范围
    translation_range_mm: float = 2.0
    camera_distance: float = 500.0
    view_angle_deg: float = 50.0


# ======================================================================
# 适配器主体
# ======================================================================
class ExternalReferenceRenderer:
    """🌟 DiffRegistration/sim.py 的薄包装

    用法：
        cfg = RendererConfig(
            obj_head_path='toukui/DiffRegistration/head/headGTY/headH.obj',
            obj_helmet_path='toukui/DiffRegistration/helmet/stdTK.obj',
            texture_path='toukui/DiffRegistration/element/qipan2.png',
            texture_path_center='toukui/DiffRegistration/element/qipan25.png',
        )
        renderer = ExternalReferenceRenderer(cfg)
        rgb, mask = renderer.render(rotation=np.zeros(3), translation=np.zeros(3))

    输出与 render_losses.VFMRegRenderLoss 直接配对：
        rgb  : np.uint8 [H, W, 3]   --> 转 tensor 后作为 target_rgb
        mask : np.uint8 [H, W]      --> 转 tensor 后作为 target_mask
    """

    def __init__(self, cfg: RendererConfig):
        self.cfg = cfg
        self._sim = None  # 延迟构造

    # ------------------------------------------------------------------
    # 延迟初始化（仅在真正需要渲染时才 import pyvista）
    # ------------------------------------------------------------------
    def _ensure_sim(self):
        if self._sim is not None:
            return
        try:
            from sim import Simulation3D  # noqa: F401  (来自 NormalRendering/sim.py)
        except ImportError as e:
            raise ImportError(
                "无法导入 DiffRegistration/NormalRendering/sim.py，"
                "请确认 toukui/DiffRegistration 目录已克隆，且已安装 pyvista, opencv-python。"
            ) from e

        from sim import Simulation3D
        self._sim = Simulation3D(
            objHpath=self.cfg.obj_head_path,
            objTKpath=self.cfg.obj_helmet_path,
            texturePath=self.cfg.texture_path,
            texturePathC=self.cfg.texture_path_center,
            Rrange=self.cfg.rotation_range_rad,
            Trange=self.cfg.translation_range_mm,
            w_s=self.cfg.image_size,
            BackgroundColor=self.cfg.background_color,
        )

    # ------------------------------------------------------------------
    # 单帧渲染 → (rgb, mask)
    # ------------------------------------------------------------------
    def render(
        self,
        rotation: np.ndarray,        # [3]  Rodrigues 向量
        translation: np.ndarray,     # [3]  mm
        save_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """渲染单帧 RGB 与 mask。
        mask 通过 RGB 与背景色的差异提取（背景色固定 = cfg.background_color）。
        """
        import cv2  # 局部导入，避免无 cv2 环境下导入失败
        self._ensure_sim()

        # sim.Simulation3D.GenerateImages 内部会写图到 obj 同目录
        path = self._sim.GenerateImages(
            PositionOffset=np.asarray(translation, dtype=np.float32),
            AngleOffset=np.asarray(rotation, dtype=np.float32),
        )
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"渲染输出读取失败: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 由背景颜色生成 mask（前景=非背景）
        bg = np.array(self.cfg.background_color, dtype=np.uint8)[None, None, :]
        diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
        mask = (diff > 15).astype(np.uint8) * 255  # 0/255

        # 可选保存到自定义目录
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(save_dir, f"{stem}_rgb.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, f"{stem}_mask.png"), mask)

        return rgb, mask

    # ------------------------------------------------------------------
    # 批量域随机化数据生成（对应论文第5.3节合成数据增广）
    # ------------------------------------------------------------------
    def render_batch(
        self,
        n_samples: int,
        rotation_offsets: Optional[Sequence[np.ndarray]] = None,
        translation_offsets: Optional[Sequence[np.ndarray]] = None,
        save_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """批量渲染，返回 [(rgb, mask, rvec, tvec), ...]
        若不传 offsets，则在 cfg 设定范围内随机采样（与 sim 内部行为一致）。
        """
        if seed is not None:
            np.random.seed(seed)

        out: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n_samples):
            r = (rotation_offsets[i] if rotation_offsets is not None
                 else np.zeros(3, dtype=np.float32))
            t = (translation_offsets[i] if translation_offsets is not None
                 else np.zeros(3, dtype=np.float32))
            rgb, mask = self.render(r, t,
                                    save_dir=os.path.join(save_dir, f"frame_{i:04d}")
                                    if save_dir else None)
            out.append((rgb, mask, np.asarray(r), np.asarray(t)))
        return out


# ======================================================================
# Torch 适配 ── 与 render_losses.VFMRegRenderLoss 直接对接
# ======================================================================
def numpy_pair_to_torch(
    rgb: np.ndarray,
    mask: np.ndarray,
    device: str = "cpu",
):
    """把单帧 (rgb, mask) numpy 转为 VFMRegRenderLoss 需要的张量格式。

    返回:
        target_rgb  : [1, 3, H, W]  in [0, 1]
        target_mask : [1, 1, H, W]  in {0, 1}
    """
    import torch
    rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0
    return rgb_t.to(device), mask_t.to(device)


# ======================================================================
# CLI: 离线生成一批合成训练数据
# ======================================================================
def _build_default_cfg() -> RendererConfig:
    """使用 DiffRegistration 仓库自带资源构造默认 cfg"""
    return RendererConfig(
        obj_head_path=os.path.join(_DIFFREG_ROOT, "head", "headGTY", "headH.obj"),
        obj_helmet_path=os.path.join(_DIFFREG_ROOT, "helmet", "stdTK.obj"),
        texture_path=os.path.join(_DIFFREG_ROOT, "element", "qipan2.png"),
        texture_path_center=os.path.join(_DIFFREG_ROOT, "element", "qipan25.png"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="使用 DiffRegistration 渲染器批量生成 VFMReg 合成训练数据")
    parser.add_argument("--n", type=int, default=8, help="生成帧数")
    parser.add_argument("--out", type=str,
                        default=os.path.join(_THIS_DIR, "output", "synthetic"),
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = _build_default_cfg()
    renderer = ExternalReferenceRenderer(cfg)
    print(f"[ExternalRenderer] 输出目录: {args.out}")
    pairs = renderer.render_batch(args.n, save_dir=args.out, seed=args.seed)
    print(f"[ExternalRenderer] 已生成 {len(pairs)} 帧 (rgb + mask)")
