# 学术风格项目主页 (`web/`)

仿照 [NeRF](https://www.matthewtancik.com/nerf) / [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) /
[SuperGlue](https://psarlin.com/superglue/) 等顶级会议的项目主页风格，
为本论文构建的多页面学术展示前端。

---

## 📐 目录结构

```
web/
├── index.html              # 🏠 主页 (论文导览)
├── pages/
│   ├── datasets.html       # 📁 数据集页面 (整合 demo_reports)
│   ├── results.html        # 📊 实验结果 (含交互式 Chart.js)
│   └── demo.html           # ▶️ 在线演示 (模拟推理)
├── css/
│   └── academic.css        # 🎨 统一学术风格样式
├── js/
│   ├── main.js             # 公共交互 (Tab/Lightbox/复制)
│   ├── charts.js           # Chart.js 图表配置
│   └── demo.js             # 演示页推理模拟
└── serve.py                # 🚀 本地启动脚本
```

---

## 🚀 快速启动

```bash
cd /apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code

# 启动本地 Web 服务器
python web/serve.py
```

然后用浏览器访问:
- **主页**: http://localhost:8000/web/
- **数据集**: http://localhost:8000/web/pages/datasets.html
- **实验结果**: http://localhost:8000/web/pages/results.html
- **在线演示**: http://localhost:8000/web/pages/demo.html

> 💡 也可以直接用浏览器打开 `web/index.html`（部分功能因 file:// 协议受限）。

---

## 🎨 设计特色

### 1. 学术配色
- 主色: `#1a3c6c` 北航深蓝
- 强调色: `#c9302c` 学术红 (类似论文图表中的最佳结果突出)
- 字体: Source Serif Pro (标题) + 系统中英文 sans (正文) + JetBrains Mono (代码/数字)

### 2. 论文级排版
- 期刊式表格（带 caption、最佳值高亮 ★）
- 图片说明 (Figure X.) 与正文衔接
- 公式/BibTeX 块一键复制
- 摘要框 / 关键指标卡 / 流程图组件化

### 3. 交互组件
- ✅ 顶部粘性导航 + 滚动联动高亮
- ✅ Tab 切换（章节内容、数据集样本）
- ✅ Chart.js 交互图表（雷达/柱状/折线/散点/饼）
- ✅ 图片点击 Lightbox 放大
- ✅ BibTeX 一键复制
- ✅ 移动端响应式布局

### 4. 在线演示 (demo.html)
- 模拟 4 视图输入界面（5 个预设场景: 标准/低光/遮挡/逆光/跨受试者）
- 逐步推理日志输出（5 阶段 pipeline）
- 6DoF 姿态矩阵实时展示
- 误差/延迟/置信度 4 项指标可视化

---

## 📊 页面内容映射

| 页面 | 数据来源 | 核心展示 |
|------|---------|---------|
| `index.html` | `results/evaluation_report.json` | 论文摘要 + 5 项核心指标 + Pipeline + 主结果对比表 |
| `pages/datasets.html` | `demo_reports/output/*.png` | 8 个数据集统计 + 7 张样本网格 + 4 张分布分析 |
| `pages/results.html` | `results/ch3/4/5_*.json` | 3 章实验结果 + 9 个交互图表 |
| `pages/demo.html` | `results/ch3_segmentation_results.json` | 模拟在线推理 (5 个回放样本) |

---

## 🛠️ 自定义修改

### 修改作者/标题
编辑 `index.html` 中 `<h1>` 与 `.authors` 部分。

### 替换主题色
编辑 `css/academic.css` 顶部 `:root` 变量:
```css
:root {
    --primary: #1a3c6c;     /* 改为你的学校/团队主色 */
    --accent: #c9302c;      /* 强调色 */
    ...
}
```

### 添加新数据集/实验
- 数据集表: 编辑 `pages/datasets.html` 的 `<table class="academic-table">`
- 新图表: 在 `js/charts.js` 中添加 `drawIfExists('chart-id', ...)`

### 修改演示样本
编辑 `js/demo.js` 顶部的 `PRESETS` 数组。

---

## 📦 依赖

- 纯静态 HTML/CSS/JS, **无需构建工具**
- CDN 加载 Chart.js v4.4.0
- 使用 Google Fonts (Source Serif Pro / JetBrains Mono)
- Python 内置 `http.server` 即可启动本地服务

---

## 🎯 适用场景

✅ 学位论文答辩演示
✅ 期刊投稿配套项目主页
✅ 开源代码 README 配套展示
✅ 个人学术作品集

---

## 📜 致谢

设计灵感来自:
- [NeRF Project Page](https://www.matthewtancik.com/nerf)
- [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Mip-NeRF](https://jonbarron.info/mipnerf/)
- [SuperGlue](https://psarlin.com/superglue/)
