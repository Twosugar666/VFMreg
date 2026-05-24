# 训练数据集 README

本目录存放论文《基于单目相机的端到端脑磁配准研究》训练所需的开源公开数据集。
受限于实验室真实数据的隐私性，使用以下公开数据集**模拟和补充** MEG-Head-360 数据集中的真实数据部分，用于演示与代码联调。

---

## 📂 目录结构

```
data/
├── head_seg/                       # 第3章 头部分割训练数据
│   ├── human_parsing/              # 人体解析（含头部分割mask）
│   ├── celebA_faces/               # 人脸图像
│   ├── cppe5/                      # 医疗防护装备
│   ├── protective_equipment/       # 安全帽 + 工业防护装备
│   └── afhqv2/                     # 动物面部（域外鲁棒性测试）
├── hdri/                           # 域随机化 - 背景图像
│   ├── dog_food_bg/                # 通用图像
│   └── wikiart_bg/                 # 艺术作品（多样化纹理）
├── nerf_test/                      # 第4章 NeRF 多视角图像参考
│   └── aloha_multiview/            # 双目机器人多视角（用于 NeRF 数据格式参考）
├── coco_subset/                    # 通用目标检测数据（占位）
├── helmet_3d/                      # 头盔 3D 模型（待补充 .obj/.glb）
└── smplx/                          # SMPL-X 参数化头部模型（待补充）
```

---

## 📊 数据集详细说明

### 🔹 1. 头部/人脸分割数据集（第3章 YOLOv8n-seg + Qwen2.5-VL）

| 数据集 | 来源 | 大小 | 样本量 | 用途 |
|--------|------|------|--------|------|
| `human_parsing` | [mattmdjaga/human_parsing_dataset](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) | 760 MB | 17,706 | 人体18类分割（含 face/hair/hat/upper-clothes 等），可裁剪头部mask |
| `celebA_faces` | [nielsr/CelebA-faces](https://huggingface.co/datasets/nielsr/CelebA-faces) | 442 MB | ~67,000 | 高质量人脸图像，多样化头型/肤色，用于训练分布扩展 |
| `cppe5` | [cppe-5](https://huggingface.co/datasets/cppe-5) | 230 MB | 1,029 | 医疗防护装备（口罩/护目镜/防护服），与 OPM-MEG 头盔场景同分布 |
| `protective_equipment` | [keremberke/protective-equipment-detection](https://huggingface.co/datasets/keremberke/protective-equipment-detection) | 2.1 GB | 6,000+ | **🌟最贴合论文场景**：佩戴安全帽人物图像（与脑磁头盔结构最相似） |
| `afhqv2` | [huggan/AFHQv2](https://huggingface.co/datasets/huggan/AFHQv2) | ~5 GB | 15,800 | 动物面部高质量图像，512×512，用于测试模型的域外泛化 |

### 🔹 2. 域随机化背景纹理（第3章&第5章 域随机化 / 论文 Table 5.1）

| 数据集 | 来源 | 大小 | 用途 |
|--------|------|------|------|
| `dog_food_bg` | [sasha/dog-food](https://huggingface.co/datasets/sasha/dog-food) | 272 MB | 一般场景图像，多样化光照与背景 |
| `wikiart_bg` | [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart) | ~3 GB | 多样化艺术风格图像，丰富纹理（论文中"texture"类背景） |

> 论文 5.1 节明确合成数据使用 **HDRI / 纯色 / 纹理** 三类共 100+ 环境进行域随机化。受限于公开 HDRI 资源访问，已用上述图像数据集作为"texture"类替代。
> 如需正版 HDRI（.exr 全景），请手动从 [Poly Haven](https://polyhaven.com/hdris)（CC0 授权）下载。

### 🔹 3. NeRF 多视角参考数据（第4章 NeRF 训练）

| 数据集 | 来源 | 用途 |
|--------|------|------|
| `aloha_multiview` | [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) | 双视角机器人数据，含相机标定参数，可作为 NeRF 数据格式参考 |

> 论文中真实 NeRF 数据为单被试 60 张多视角图像 + COLMAP 估计位姿。
> 经典 NeRF-Synthetic（lego/chair/ship/...）数据集需手动从 [bmild/nerf](https://github.com/bmild/nerf) 仓库 README 中的 Google Drive 链接下载（需翻墙）。

### 🔹 4. 待补充的辅助资源

以下因许可协议或下载源限制需手动补充：

| 资源 | 说明 | 下载地址 |
|------|------|---------|
| **SMPL-X 模型权重** | 论文 5.2 节用于参数化生成多样头型 | https://smpl-x.is.tue.mpg.de/ （需注册学术账号） |
| **Poly Haven HDRI** | 100+ 环境贴图（论文 Table 5.1） | https://polyhaven.com/hdris （CC0） |
| **Helmet 3D Mesh** | OPM-MEG 头盔的 CAD 模型 | 论文中为实验室自有，可用 [Free3D 头盔](https://free3d.com/3d-models/helmet) 替代 |
| **NeRF-Synthetic 经典8场景** | NeRF 论文 lego/chair/ship/... | https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 |
| **MEG-Head-360 真实数据** | 论文自采（5被试300张/20被试微调测试） | 因隐私无法公开，使用上述 `protective_equipment` + `human_parsing` 模拟 |

---

## 🚀 数据加载示例

### 加载人体解析数据（头部分割训练）

```python
import pandas as pd
from PIL import Image
import io

df = pd.read_parquet('data/head_seg/human_parsing/data/train-00000-of-00002-f3a663f7140ee7fd.parquet')
print(df.columns)  # ['image', 'mask']
img = Image.open(io.BytesIO(df.iloc[0]['image']['bytes']))
mask = Image.open(io.BytesIO(df.iloc[0]['mask']['bytes']))
print(f'Image: {img.size}, Mask: {mask.size}, classes: {set(mask.getdata())}')
```

### 加载防护装备数据（最贴合脑磁头盔场景）

```python
from datasets import load_dataset
ds = load_dataset('parquet', data_dir='data/head_seg/protective_equipment/data')
# 类别：helmet, head, person, ...
print(ds['train'][0])
```

### 转换为 YOLO 训练格式

```python
# 见 preprocess/dataset.py 中的 convert_to_yolo_format() 函数
from preprocess.dataset import convert_parquet_to_yolo
convert_parquet_to_yolo(
    parquet_path='data/head_seg/human_parsing/data/train-00000-of-00002-f3a663f7140ee7fd.parquet',
    output_dir='data/head_seg/human_parsing_yolo',
    head_class_ids=[1, 2, 11],  # face, hair, hat 三类合并为"head"
)
```

---

## 📈 数据规模统计

| 章节 | 数据来源 | 真实/合成 | 样本量 | 大小 |
|------|---------|----------|--------|------|
| 第3章 头部分割 | 论文自采 + 上述5个开源数据集 | 真实 | ~90,000 | ~9 GB |
| 第3章 数据增强 | wikiart_bg + dog_food_bg | 合成背景 | ~80,000 | ~3.3 GB |
| 第4章 NeRF 配准 | 论文自采 60 张 + aloha 参考 | 真实 | 60+ | <100 MB |
| 第5章 VFMReg | Blender 程序化生成（脚本见 [augmentation.py](../preprocess/augmentation.py)）| 合成 | 100,000 | 待生成 |

---

## 🔄 重新下载

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速

python3 -c "
from huggingface_hub import snapshot_download
# 头部分割
snapshot_download('mattmdjaga/human_parsing_dataset', repo_type='dataset',
                  local_dir='./head_seg/human_parsing')
snapshot_download('keremberke/protective-equipment-detection', repo_type='dataset',
                  local_dir='./head_seg/protective_equipment')
snapshot_download('cppe-5', repo_type='dataset', local_dir='./head_seg/cppe5')
snapshot_download('nielsr/CelebA-faces', repo_type='dataset',
                  local_dir='./head_seg/celebA_faces')
snapshot_download('huggan/AFHQv2', repo_type='dataset', local_dir='./head_seg/afhqv2')
# 背景图像
snapshot_download('sasha/dog-food', repo_type='dataset', local_dir='./hdri/dog_food_bg')
snapshot_download('huggan/wikiart', repo_type='dataset', local_dir='./hdri/wikiart_bg',
                  allow_patterns=['data/train-00000-*.parquet'])
"
```

---

## ⚖️ 数据集授权说明

| 数据集 | 协议 | 用途限制 |
|--------|------|---------|
| Human Parsing | MIT | 学术 + 商用 |
| CelebA | 仅限非商用研究 | **学术研究专用** |
| CPPE-5 | CC-BY-4.0 | 学术 + 商用（标注来源） |
| Protective Equipment | CC-BY-4.0 | 学术 + 商用 |
| AFHQv2 | CC BY-NC 4.0 | 仅限非商用 |
| WikiArt | 公开数据集 | 学术研究 |

⚠️ **本项目仅用于学术研究目的（论文复现演示）**，使用上述数据集时请遵守各自原始授权协议。
