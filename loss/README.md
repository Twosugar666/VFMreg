# Loss 损失函数库

本目录提供了论文《基于单目相机的端到端脑磁配准研究》中所有损失函数的 **PyTorch 实现**、**训练分布可视化**、**性能基准测试** 和 **完整单元测试**。

---

## 📦 目录结构

```
loss/
├── README.md                        # 本文档
├── __init__.py                      # 统一导出 26 个 API
│
├── 📐 损失函数实现 (4 个模块, 25 个 loss 类)
├── segmentation_losses.py           # 第3章：分割损失 (CE/Dice/Focal/Tversky/Sobel/MultiScale/...)
├── nerf_losses.py                   # 第4章：NeRF 损失 (Photo/Density/LPIPS/TV/Depth/...)
├── pose_losses.py                   # 第5章：姿态损失 (Geodesic/Chordal/Quaternion/6D/Hinge/...)
├── render_losses.py                 # 第5章：可微渲染损失 (IoU/Silhouette/Masked-RGB/...)
├── utils.py                         # LossMeter / AdaptiveWeights / LossLogger
│
├── 📊 可视化与基准
├── visualize_losses.py              # 9 张训练分布图（基础）
├── visualize_advanced.py            # 6 张高级补充图（答辩用）
├── loss_benchmark.py                # 性能基准测试（速度/显存）
├── loss_recipes.py                  # 5 个即拿即用训练配方
│
├── 🧪 测试
├── tests/
│   ├── __init__.py
│   └── test_losses.py               # 40 个单元测试（全部通过）
│
└── 📁 输出
    └── output/
        ├── *.png                    # 9 张基础训练分布图
        ├── advanced/*.png           # 6 张高级可视化图
        ├── benchmark_cpu.json       # CPU 基准数据
        └── benchmark_cpu_table.txt  # CPU 基准表格
```

---

## 🎯 损失函数清单（26 个 API）

### 📌 第3章 头部分割（7 个 + 1 复合）
| 类 | 用途 | 论文对应 |
|---|---|---|
| `DiceLoss` | 类别不平衡的 Dice 系数 | Eq. 3.4 |
| `FocalLoss` | 困难样本加权 | — |
| `TverskyLoss` | Dice 推广，可调 FP/FN | — |
| `BoundaryLoss` | 距离图边界损失 | — |
| **`SobelEdgeLoss`** | 🌟 Sobel 边缘增强（τ=0.3） | Eq. 3.5 |
| **`MultiScaleCELoss`** | 🌟 多尺度交叉熵（0.5/0.3/0.2） | Eq. 3.3 |
| **`ComboSegLoss`** | 🌟 第3章最终复合损失 | Eq. 3.6 |

### 📌 第4章 NeRF 配准（5 个 + 1 复合）
| 类 | 用途 | 论文对应 |
|---|---|---|
| `PhotoLoss` | L1 / MSE / Huber 光度 | Eq. 4.4 |
| **`DensityLoss`** | 🌟 体密度损失（约束 head surface σ） | Eq. 4.5 |
| `LPIPSLossLite` | VGG 感知损失 | Eq. 4.6 |
| `TotalVariationLoss` | TV 平滑正则 | — |
| `DepthConsistencyLoss` | 多视角深度一致性 | — |
| **`NeRFRegLoss`** | 🌟 第4章配准完整损失 | Eq. 4.7 |

### 📌 第5章 姿态回归（6 个 + 1 复合）
| 类 | 用途 | 论文对应 |
|---|---|---|
| `TranslationLoss` | L1/L2/Smooth-L1 平移 | — |
| `GeodesicLoss` | SO(3) 测地线距离 | Eq. 5.2 |
| `ChordalLoss` | Chordal Frobenius 距离 | — |
| `QuaternionLoss` | 四元数双重最小距离 | — |
| **`Rotation6DLoss`** | 🌟 6D 连续旋转表示损失 | Eq. 5.3 |
| `AnglePenaltyLoss` | 大误差 hinge 惩罚 | — |
| **`VFMRegPoseLoss`** | 🌟 第5章端到端姿态损失 | Eq. 5.4 |

### 📌 第5章 可微渲染（4 个 + 1 复合）
| 类 | 用途 |
|---|---|
| `DifferentiableIoULoss` | 软 IoU |
| `SilhouetteL1Loss` | 轮廓 L1 |
| `MaskedRGBLoss` | mask 内 RGB 损失 |
| `MultiViewConsistencyLoss` | 多视角姿态一致性 |
| **`VFMRegRenderLoss`** | 🌟 合成数据预训练复合损失 |

### 📌 工具（3 个）
- `LossMeter` — 累计平均
- `AdaptiveWeights` — Kendall et al. 多任务自适应权重
- `LossLogger` — 训练日志保存/加载

---

## 🚀 快速使用

### 1) 一行调用复合损失

```python
from loss import VFMRegPoseLoss

# 论文最优超参，开箱即用
pose_loss = VFMRegPoseLoss()
total, breakdown = pose_loss(d6_pred, t_pred, R_gt, t_gt)
# breakdown = {"L_total":..., "L_translation":..., "L_rotation_6d":..., "L_geodesic":..., "L_hinge":...}
```

### 2) 自适应权重（无需手调超参）

```python
from loss import AdaptiveWeights

aw = AdaptiveWeights(n_tasks=3)
optimizer = torch.optim.Adam(list(model.parameters()) + list(aw.parameters()), lr=1e-4)

l1, l2, l3 = trans_loss(...), rot_loss(...), render_loss(...)
total = aw([l1, l2, l3])  # 自动学习最优权重
total.backward()
```

### 3) 训练日志记录与可视化

```python
from loss import LossLogger

logger = LossLogger(out_path="results/training_logs.json")
for step in range(num_steps):
    total, breakdown = pose_loss(...)
    logger.log(step, breakdown)
logger.save()  # 保存为 JSON, 后续可用 visualize_losses.py 绘图
```

### 4) 直接调用预设训练配方

```python
from loss.loss_recipes import recipe_vfmreg_stage1, recipe_vfmreg_stage2

# Stage-1: 合成数据预训练 (~100 epoch)
recipe_vfmreg_stage1(model, train_loader, val_loader, num_epochs=100)

# Stage-2: 真实域微调 (~30 epoch)
recipe_vfmreg_stage2(model, real_train_loader, real_val_loader,
                      pretrain_ckpt='ckpts/vfmreg/stage1/best.pth',
                      num_epochs=30)
```

---

## 📊 可视化（共 15 张图）

### 基础图（9 张）

```bash
python loss/visualize_losses.py
```

| # | 文件 | 内容 |
|---|---|---|
| 1 | `seg_loss_curves.png`            | 第3章 5 种分割 loss 收敛曲线（线性+对数） |
| 2 | `seg_loss_ablation.png`          | 第3章 损失组件消融柱状图 |
| 3 | `seg_loss_distribution.png`      | 第3章 loss 值分布小提琴图 |
| 4 | `nerf_loss_curves.png`           | 第4章 NeRF 各分量 + PSNR 收敛 |
| 5 | `nerf_loss_decomposition.png`    | 第4章 loss 占比堆叠图 |
| 6 | `pose_loss_curves.png`           | 第5章 5 种旋转损失曲线 + 箱线图 |
| 7 | `pose_loss_landscape.png`        | 第5章 4 种旋转损失 3D 曲面对比 |
| 8 | `multi_task_weights.png`         | 第5章 多任务自适应权重演化 |
| 9 | 🏆 `loss_summary_dashboard.png`  | **3×3 综合大图（推荐答辩使用）** |

### 高级补充图（6 张）

```bash
python loss/visualize_advanced.py
```

| # | 文件 | 内容 |
|---|---|---|
| 10 | `loss_landscape_3d.png`              | 优化轨迹 3D 可视化（俯视等高线 + 3D 曲面） |
| 11 | `gradient_analysis.png`              | 梯度幅值分析（论证 6D > 欧拉角的数值稳定性） |
| 12 | `convergence_speed_comparison.png`   | 达到目标精度所需 epoch 数对比 |
| 13 | `loss_correlation_heatmap.png`       | loss 分量相关性热力图 |
| 14 | `domain_gap_loss.png`                | 合成→真实域差距演化（Stage-1 → Stage-2） |
| 15 | `noise_robustness.png`               | 各 loss 对噪声鲁棒性（雷达图） |

---

## ⚡ 性能基准（CPU 实测，n_runs=20）

```bash
python loss/loss_benchmark.py --device cuda  # 或 cpu
```

| 损失函数 | 平均耗时 (ms) | 用途 |
|---|---|---|
| `PhotoLoss(L1)` | 0.07 | 渲染监督 |
| `DensityLoss` | 0.02 | NeRF |
| `TranslationLoss` | 0.01 | 平移 |
| `GeodesicLoss` | 0.05 | 旋转 |
| `Rotation6DLoss` | 0.19 | 论文核心 |
| `DifferentiableIoULoss` | 0.15 | 渲染 |
| `MultiScaleCELoss` | 2.23 | 第3章核心 |
| **`VFMRegPoseLoss`** | **0.46** | 🌟 第5章姿态 |
| **`VFMRegRenderLoss`** | **0.36** | 🌟 第5章渲染 |
| **`NeRFRegLoss`** | **0.29** | 🌟 第4章 NeRF |
| **`ComboSegLoss`** | **12.3** | 🌟 第3章分割（最大） |

> 💡 在 CUDA A100 上比上表快约 5–10×。

---

## 🧪 单元测试（40 个，全部通过）

```bash
# 直接运行
python loss/tests/test_losses.py

# 或用 pytest
python -m pytest loss/tests/test_losses.py -v
```

测试覆盖：
- ✅ 数值正确性（输出范围、对单位输入的期望值）
- ✅ 反向传播正确性
- ✅ 边界情况（完美匹配、零差异、π 旋转）
- ✅ 设备兼容性（CPU/CUDA）
- ✅ 数学性质（旋转矩阵正交性、行列式=1、四元数对映点不变性）

---

## ⚙️ 论文最优超参速查

| 损失 | 权重 |
|---|---|
| **MultiScaleCE** | (0.5, 0.3, 0.2) — P3/P4/P5 |
| **Sobel** | threshold=0.3, weight=0.1 |
| **ComboSeg** | w_ms=1.0, w_dice=0.5, w_edge=0.1 |
| **NeRFReg** | w_density=1.0, w_photo=0.5, w_lpips=0.1, w_tv=0.01 |
| **VFMRegPose** | w_trans=1.0, w_rot6d=1.0, w_geo=0.1, w_hinge=0.05 |
| **VFMRegRender** | w_iou=1.0, w_rgb=0.5, w_sil=0.2 |

---

## 📚 参考文献

1. Kendall, A. et al. *Multi-Task Learning Using Uncertainty to Weigh Losses*, CVPR 2018.
2. Zhou, Y. et al. *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019.
3. Mildenhall, B. et al. *NeRF: Representing Scenes as Neural Radiance Fields*, ECCV 2020.
4. Kervadec, H. et al. *Boundary loss for highly unbalanced segmentation*, MIDL 2019.
5. Lin, T. et al. *Focal Loss for Dense Object Detection*, ICCV 2017.
6. Zhang, R. et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018.
7. Salehi, S. et al. *Tversky Loss Function for Image Segmentation*, MLMI 2017.

---

## 🎓 关键贡献

本 loss 库不仅提供了论文需要的所有损失函数，还包含：

1. **完整可复现** — 26 个 API + 40 个单元测试 + 性能基准
2. **答辩级可视化** — 15 张高质量训练分布图
3. **即拿即用** — 5 个训练配方 + 论文最优超参全部 hard-code
4. **跨章节统一** — 第3/4/5 章所有损失都遵循一致的 `(total, breakdown)` 返回约定，方便日志记录

---

## 🔌 DiffRegistration 参考实现集成（第5章）

本目录额外提供 `external_renderer.py`，作为开源项目
[DiffRegistration](https://github.com/XingwenFu/DiffRegistration) 的轻量级
适配层（仓库已克隆至 `code/toukui/DiffRegistration/`）。

### 与论文渲染损失的对应关系

| 角色 | 在论文中对应 | 由谁产生 |
|------|---------------|-----------|
| `target_rgb` (M³ 真值 RGB) | 第5.3节合成数据 GT | **`ExternalReferenceRenderer`** （PyVista，非可微） |
| `target_mask` (M³ 真值轮廓) | 第5.3节合成数据 GT | **`ExternalReferenceRenderer`**（背景颜色阈值） |
| `rendered_rgb` / `rendered_mask` | 训练过程的可微输出 | 本项目内可微渲染器 |
| **`VFMRegRenderLoss`** | 把上面四者监督起来 | `loss/render_losses.py` |

### 一句话总结
> *DiffRegistration/sim.py 是"合成数据生产管线"，本项目可微渲染器+`VFMRegRenderLoss` 是"反传训练管线"。两者通过 `external_renderer.py` 解耦。*

### 快速使用

```python
from loss import (
    ExternalReferenceRenderer, RendererConfig, numpy_pair_to_torch,
    VFMRegRenderLoss,
)
import numpy as np, torch

# 1) 用 DiffRegistration 资源构造 GT 渲染器
cfg = RendererConfig(
    obj_head_path='toukui/DiffRegistration/head/headGTY/headH.obj',
    obj_helmet_path='toukui/DiffRegistration/helmet/stdTK.obj',
    texture_path='toukui/DiffRegistration/element/qipan2.png',
    texture_path_center='toukui/DiffRegistration/element/qipan25.png',
)
ref = ExternalReferenceRenderer(cfg)
rgb_np, mask_np = ref.render(rotation=np.zeros(3), translation=np.zeros(3))
target_rgb, target_mask = numpy_pair_to_torch(rgb_np, mask_np, device='cuda')

# 2) 假设训练时的可微渲染输出 (此处用占位 tensor 演示)
rendered_rgb  = torch.rand_like(target_rgb,  requires_grad=True)
rendered_mask = torch.rand_like(target_mask, requires_grad=True)

# 3) 计算 VFMReg 渲染损失
loss_fn = VFMRegRenderLoss(w_iou=1.0, w_rgb=0.5, w_silhouette=0.2)
total, info = loss_fn(rendered_mask, rendered_rgb, target_mask, target_rgb)
total.backward()
print(info)
```

### 离线生成一批合成训练数据（CLI）
```bash
cd code
python -m loss.external_renderer --n 64 --out loss/output/synthetic --seed 42
# 产出:  loss/output/synthetic/frame_xxxx/{*_rgb.png, *_mask.png}
```

### 与原仓库的差异 / 扩展点

| 方面 | DiffRegistration/sim.py | 本项目 external_renderer.py |
|------|------|------|
| 渲染后端 | PyVista (OpenGL，**非可微**) | 同上，仅做封装 |
| 输出形态 | 仅写 PNG 到磁盘 | `(rgb, mask) np.ndarray` 直接返回 |
| Mask 来源 | 无（需后处理） | 通过背景颜色阈值自动提取 |
| Torch 互操作 | ❌ | ✅ `numpy_pair_to_torch()` |
| 训练用法 | 需手动二次开发 | 与 `VFMRegRenderLoss` 一行对接 |
| 依赖隔离 | 必装 pyvista 才能 import | **延迟导入**，无 GUI 环境也能 import |

> ⚠️ **可微性说明**：`sim.py` 的 PyVista 渲染本身**不可微**，因此它仅用于
> 生产 GT。真正参与梯度反传的是本项目内基于 PyTorch 的可微渲染器
> （soft rasterizer / NeRF 风格体渲染），其输出经过 `DifferentiableIoULoss`
> 等可微损失反传到姿态参数 / 网络权重上，这与论文第5.3节"合成→真实"
> 的两阶段训练流程一致。
