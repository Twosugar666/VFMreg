# 数据集报告生成脚本 (`demo_reports/`)

本目录包含一组演示脚本，用于扫描和分析 [`code/data/`](../data/) 目录下已下载的训练数据集，
生成可视化总结报告，方便论文评审/答辩演示使用。

---

## 📂 文件清单

```
demo_reports/
├── 01_dataset_overview.py       # 数据集体积/样本量统计
├── 02_sample_visualization.py   # 样本图像网格展示
├── 03_data_distribution.py      # 图像分辨率/亮度/RGB 分布分析
├── 04_html_report.py            # 汇总生成 HTML 报告
├── run_all.py                   # 一键运行 01→04
└── output/                      # 输出目录（自动创建）
    ├── 01_overview/
    │   ├── dataset_size_chart.png    # 体积柱状图
    │   ├── dataset_count_chart.png   # 样本量柱状图
    │   ├── dataset_pie.png           # 章节占比饼图
    │   └── overview_stats.json
    ├── 02_samples/
    │   ├── grid_human_parsing.png    # 人体解析样本+mask
    │   ├── grid_celebA_faces.png     # 人脸样本
    │   ├── grid_cppe5.png            # 医疗防护装备
    │   ├── grid_protective_equipment.png  # 安全帽（最贴合脑磁头盔🌟）
    │   ├── grid_afhqv2.png           # 动物面部
    │   ├── grid_dog_food_bg.png      # 通用图像
    │   └── grid_combined.png         # 跨数据集对比
    ├── 03_distribution/
    │   ├── resolution_scatter.png    # 分辨率散点
    │   ├── brightness_hist.png       # 亮度直方图
    │   ├── rgb_histogram.png         # RGB通道分布
    │   ├── mask_class_distribution.png  # 类别像素占比
    │   └── distribution_stats.json
    └── 04_final_report/
        ├── index.html                # 📄 主报告（用浏览器打开）
        └── assets/                   # 图表副本
```

---

## 🚀 快速开始

### 方式一：一键生成（推荐）

```bash
cd /apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code
python demo_reports/run_all.py
```

完成后，用浏览器打开：
```
demo_reports/output/04_final_report/index.html
```

### 方式二：分步运行

```bash
cd /apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code

# 1. 统计每个数据集的体积/样本量
python demo_reports/01_dataset_overview.py

# 2. 从每个数据集采样图像生成网格图
python demo_reports/02_sample_visualization.py

# 3. 分析图像分布特征
python demo_reports/03_data_distribution.py

# 4. 汇总生成 HTML 报告
python demo_reports/04_html_report.py
```

---

## 📋 报告内容预览

生成的 HTML 报告包含以下章节：

### 1️⃣ 头部统计卡片
- 已下载数据集数量
- 总磁盘占用
- 总样本量
- 覆盖论文章节

### 2️⃣ 数据集清单表
| 数据集 | 用途分类 | 体积 | 样本数 | 状态 |
| ----- | ------- | ---- | ----- | ---- |
| human_parsing | Ch.3 头部分割 | 760 MB | 17,706 | ✅ |
| protective_equipment | Ch.3 头盔分割 | 2.1 GB | 6,000+ | ✅ |
| ... | ... | ... | ... | ... |

### 3️⃣ 统计可视化
- 数据集体积柱状图
- 样本量柱状图（log坐标）
- 论文章节占比饼图

### 4️⃣ 数据样本展示
- 每个数据集 6-8 张随机样本
- Human Parsing 含语义 mask 对比
- 跨数据集综合对比图

### 5️⃣ 分布特征分析
- 图像分辨率散点图
- 亮度分布直方图（评估域随机化覆盖度）
- RGB 通道分布
- Human Parsing 类别像素占比（红色高亮头部相关类别）

---

## 🛠️ 依赖

```bash
pip install pandas matplotlib pillow pyarrow numpy
```

> 已包含在项目 [`requirements.txt`](../requirements.txt) 中。

---

## 📸 演示效果

报告页面具备以下交互特性：
- 🖱️ **图像点击放大**：点击任意图表查看大图
- 📱 **响应式布局**：自适应桌面/平板
- 🎨 **现代化样式**：渐变色 + 卡片式设计
- ⌨️ **快捷键**：ESC 关闭灯箱

---

## 🔧 自定义配置

### 修改数据集路径
编辑各脚本顶部的 `DATA_ROOT` 常量。

### 调整采样数量
- `02_sample_visualization.py` 中各数据集的 `'n': 8` 字段
- `03_data_distribution.py` 中 `sample_parquet(path, n=200)` 的 n 参数

### 修改样式/配色
- 图表颜色：脚本中的 `colors` / `color` 字段
- HTML 主题：`04_html_report.py` 中的 `<style>` 部分
