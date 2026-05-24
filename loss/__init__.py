"""
loss/__init__.py
统一导出本目录所有损失函数
"""

# 第3章 分割损失
from .segmentation_losses import (
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    BoundaryLoss,
    SobelEdgeLoss,
    MultiScaleCELoss,
    ComboSegLoss,
)

# 第4章 NeRF 损失
from .nerf_losses import (
    PhotoLoss,
    DensityLoss,
    LPIPSLossLite,
    TotalVariationLoss,
    DepthConsistencyLoss,
    NeRFRegLoss,
)

# 第5章 姿态损失
from .pose_losses import (
    TranslationLoss,
    GeodesicLoss,
    ChordalLoss,
    QuaternionLoss,
    Rotation6DLoss,
    AnglePenaltyLoss,
    VFMRegPoseLoss,
    rotation_6d_to_matrix,
    quaternion_to_matrix,
)

# 第5章 渲染损失
from .render_losses import (
    DifferentiableIoULoss,
    SilhouetteL1Loss,
    MaskedRGBLoss,
    MultiViewConsistencyLoss,
    VFMRegRenderLoss,
)

# 工具
from .utils import LossMeter, AdaptiveWeights, LossLogger

# 第5章 外部参考渲染器（DiffRegistration 集成）
# 仅在 numpy 可用时尝试导入；若 pyvista 缺失，模块本身仍可加载，
# 真正调用 render() 时才会报缺包错误。
try:
    from .external_renderer import (
        ExternalReferenceRenderer,
        RendererConfig,
        numpy_pair_to_torch,
    )
    _HAS_EXTERNAL_RENDERER = True
except Exception:  # pragma: no cover
    _HAS_EXTERNAL_RENDERER = False

__all__ = [
    # Ch.3
    "DiceLoss", "FocalLoss", "TverskyLoss", "BoundaryLoss",
    "SobelEdgeLoss", "MultiScaleCELoss", "ComboSegLoss",
    # Ch.4
    "PhotoLoss", "DensityLoss", "LPIPSLossLite",
    "TotalVariationLoss", "DepthConsistencyLoss", "NeRFRegLoss",
    # Ch.5 Pose
    "TranslationLoss", "GeodesicLoss", "ChordalLoss",
    "QuaternionLoss", "Rotation6DLoss", "AnglePenaltyLoss",
    "VFMRegPoseLoss",
    "rotation_6d_to_matrix", "quaternion_to_matrix",
    # Ch.5 Render
    "DifferentiableIoULoss", "SilhouetteL1Loss",
    "MaskedRGBLoss", "MultiViewConsistencyLoss", "VFMRegRenderLoss",
    # Utils
    "LossMeter", "AdaptiveWeights", "LossLogger",
]

if _HAS_EXTERNAL_RENDERER:
    __all__ += [
        "ExternalReferenceRenderer",
        "RendererConfig",
        "numpy_pair_to_torch",
    ]
