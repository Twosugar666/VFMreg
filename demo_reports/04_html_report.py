"""
================================================================
脚本 04: 生成完整 HTML 总结报告
================================================================
功能:
    汇总 01/02/03 三个脚本生成的所有图表和 JSON 数据，
    生成一个交互式美观的 HTML 报告，可用浏览器打开演示
输出: ./demo_reports/output/04_final_report/index.html
用法:
    python demo_reports/04_html_report.py
    然后用浏览器打开 index.html 即可
================================================================
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path('/apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code/demo_reports')
OUTPUT_BASE = ROOT / 'output'
REPORT_DIR = OUTPUT_BASE / '04_final_report'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = REPORT_DIR / 'assets'
ASSETS_DIR.mkdir(exist_ok=True)


def copy_assets():
    """把所有图表复制到 report assets 目录"""
    sources = [
        OUTPUT_BASE / '01_overview',
        OUTPUT_BASE / '02_samples',
        OUTPUT_BASE / '03_distribution',
    ]
    copied = []
    for src in sources:
        if not src.exists():
            print(f'  ⚠ 不存在: {src}')
            continue
        for f in src.glob('*.png'):
            dst = ASSETS_DIR / f.name
            shutil.copy(f, dst)
            copied.append(dst.name)
    return copied


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def fmt_size(n):
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} PB'


def render_dataset_table(stats):
    """渲染数据集统计表格"""
    if not stats:
        return '<p>暂无数据</p>'
    rows = []
    for ds in stats['datasets']:
        status = '✅ 已下载' if ds['exists'] else '⏳ 未下载'
        status_class = 'badge-ok' if ds['exists'] else 'badge-pending'
        rows.append(f'''
        <tr>
            <td>{ds['name']}</td>
            <td><span class="chapter-tag">{ds['paper_chapter']}</span> {ds['category']}</td>
            <td>{ds['size_human']}</td>
            <td>{ds['sample_count']:,}</td>
            <td><span class="{status_class}">{status}</span></td>
        </tr>''')
    return ''.join(rows)


def render_image_card(img_name, title, description=''):
    rel_path = f'assets/{img_name}'
    full = ASSETS_DIR / img_name
    if not full.exists():
        return f'<div class="img-card empty"><h3>{title}</h3><p>(暂未生成)</p></div>'
    return f'''
    <div class="img-card">
        <h3>{title}</h3>
        {f'<p class="desc">{description}</p>' if description else ''}
        <img src="{rel_path}" alt="{title}" loading="lazy" onclick="openLightbox(this.src)"/>
    </div>'''


def render_html(overview_stats, dist_stats, copied_imgs):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_total = overview_stats['summary']['total_count'] if overview_stats else 0
    n_avail = overview_stats['summary']['available_count'] if overview_stats else 0
    total_size = overview_stats['summary']['total_size_human'] if overview_stats else 'N/A'
    total_samples = overview_stats['summary']['total_samples'] if overview_stats else 0

    # 检查图像是否存在
    imgs = set(copied_imgs)

    def has(name):
        return name in imgs

    table_rows = render_dataset_table(overview_stats) if overview_stats else ''

    # 图像卡片HTML
    overview_imgs = ''
    if has('dataset_size_chart.png'):
        overview_imgs += render_image_card('dataset_size_chart.png',
            '📊 数据集体积对比', '展示各数据集占用磁盘空间')
    if has('dataset_count_chart.png'):
        overview_imgs += render_image_card('dataset_count_chart.png',
            '📈 数据集样本量对比', '对数坐标显示，便于观察数量级差异')
    if has('dataset_pie.png'):
        overview_imgs += render_image_card('dataset_pie.png',
            '🥧 论文章节占比', '按论文章节(Ch.3/4/5)分组的数据占比')

    sample_imgs = ''
    sample_files = [
        ('grid_human_parsing.png', '👤 Human Parsing 样本（含语义mask）',
         '人体18类语义解析，用于生成头部分割训练标签'),
        ('grid_celebA_faces.png', '😀 CelebA 人脸样本',
         '高质量人脸图像，多样化头型与肤色'),
        ('grid_cppe5.png', '😷 CPPE-5 医疗防护装备',
         '与脑磁头盔同分布的医疗装备图像'),
        ('grid_protective_equipment.png',
         '⛑️ Protective Equipment 安全帽（最贴合脑磁头盔场景）',
         '佩戴安全帽人物图像，与论文 OPM-MEG 场景结构最相似'),
        ('grid_afhqv2.png', '🐕 AFHQv2 动物面部',
         '用于测试模型的域外（OOD）泛化能力'),
        ('grid_dog_food_bg.png', '🖼️ 通用图像（背景域随机化）',
         '用作合成数据的背景纹理替换'),
        ('grid_combined.png', '🎨 跨数据集综合对比', '所有数据集样本对比'),
    ]
    for fname, title, desc in sample_files:
        if has(fname):
            sample_imgs += render_image_card(fname, title, desc)

    dist_imgs = ''
    dist_files = [
        ('resolution_scatter.png', '📐 图像分辨率分布',
         '宽×高散点图，反映各数据集图像规格'),
        ('brightness_hist.png', '💡 亮度分布直方图',
         '用于评估域随机化的亮度覆盖范围'),
        ('rgb_histogram.png', '🎨 RGB 通道分布',
         '各数据集颜色通道直方图对比'),
        ('mask_class_distribution.png', '🏷️ 语义类别像素占比',
         '红色为头部相关类别（Hat/Hair/Face/Sunglasses/Scarf）'),
    ]
    for fname, title, desc in dist_files:
        if has(fname):
            dist_imgs += render_image_card(fname, title, desc)

    # 分布统计表
    dist_table = ''
    if dist_stats:
        for name, s in dist_stats.items():
            res = s['resolution']
            br = s['brightness']
            row = (
                '<tr>'
                f'<td>{name}</td>'
                f'<td>{s["n_samples_analyzed"]}</td>'
                f'<td>{res["width_min"]:.0f} ~ {res["width_max"]:.0f}<br><small>'
                f'mean: {res["width_mean"]:.0f}</small></td>'
                f'<td>{res["height_min"]:.0f} ~ {res["height_max"]:.0f}<br><small>'
                f'mean: {res["height_mean"]:.0f}</small></td>'
                f'<td>{br["mean"]:.1f} +/- {br["std"]:.1f}</td>'
                '</tr>'
            )
            dist_table += row

    dist_table_section = ''
    if dist_table:
        dist_table_section = (
            '<h3 style="margin-top:30px;">数值统计明细</h3>'
            '<table><thead><tr>'
            '<th>数据集</th><th>采样数</th>'
            '<th>宽度范围 (px)</th><th>高度范围 (px)</th>'
            '<th>亮度 (mean +/- std)</th>'
            '</tr></thead><tbody>'
            + dist_table +
            '</tbody></table>'
        )

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>训练数据集总结报告 - 单目脑磁配准</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: linear-gradient(135deg, #f0f4f8 0%, #e8eef5 100%);
    color: #2c3e50;
    line-height: 1.6;
}}
.header {{
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 50px 40px 40px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}}
.header h1 {{ font-size: 2.4em; margin-bottom: 8px; }}
.header .subtitle {{ font-size: 1.1em; opacity: 0.9; margin-bottom: 20px; }}
.header .meta {{ font-size: 0.95em; opacity: 0.85; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}

.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
    margin: -50px 30px 30px;
    position: relative;
    z-index: 10;
}}
.stat-card {{
    background: white;
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    border-left: 4px solid #3498db;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.stat-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.12); }}
.stat-card .label {{ font-size: 0.88em; color: #7f8c8d; margin-bottom: 6px; }}
.stat-card .value {{ font-size: 1.9em; font-weight: 700; color: #2c3e50; }}
.stat-card.success {{ border-left-color: #27ae60; }}
.stat-card.warning {{ border-left-color: #e67e22; }}
.stat-card.info {{ border-left-color: #9b59b6; }}

.section {{
    background: white;
    margin-bottom: 30px;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}}
.section h2 {{
    font-size: 1.6em;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 3px solid #3498db;
    color: #1e3c72;
}}
.section .desc {{ color: #7f8c8d; margin-bottom: 18px; font-size: 0.95em; }}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 14px;
    font-size: 0.92em;
}}
th, td {{
    padding: 12px 14px;
    text-align: left;
    border-bottom: 1px solid #ecf0f1;
}}
th {{
    background: linear-gradient(135deg, #ecf0f1, #d5dbdb);
    font-weight: 600;
    color: #2c3e50;
}}
tr:hover {{ background: #f8f9fa; }}

.badge-ok {{ background: #d5f5e3; color: #1e8449; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; }}
.badge-pending {{ background: #fdebd0; color: #b9770e; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; }}
.chapter-tag {{ background: #ebf5fb; color: #2874a6; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 600; }}

.gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 18px;
    margin-top: 18px;
}}
.img-card {{
    background: #fafafa;
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.2s;
}}
.img-card:hover {{ box-shadow: 0 6px 20px rgba(0,0,0,0.1); transform: translateY(-2px); }}
.img-card h3 {{ font-size: 1.05em; margin-bottom: 6px; color: #1e3c72; }}
.img-card p.desc {{ font-size: 0.85em; color: #7f8c8d; margin-bottom: 10px; }}
.img-card img {{
    width: 100%;
    border-radius: 6px;
    cursor: zoom-in;
    border: 1px solid #ecf0f1;
}}
.img-card.empty {{ background: #f9f9f9; color: #95a5a6; text-align: center; padding: 40px; }}

.footer {{
    text-align: center;
    color: #7f8c8d;
    padding: 30px;
    font-size: 0.9em;
}}

/* 灯箱效果 */
.lightbox {{
    display: none;
    position: fixed;
    z-index: 999;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.92);
    justify-content: center;
    align-items: center;
    cursor: zoom-out;
}}
.lightbox.active {{ display: flex; }}
.lightbox img {{ max-width: 95%; max-height: 95%; border-radius: 6px; }}

@media (max-width: 768px) {{
    .header h1 {{ font-size: 1.6em; }}
    .stats-grid {{ margin: -30px 10px 20px; }}
    .container {{ padding: 15px; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>📑 训练数据集总结报告</h1>
    <p class="subtitle">基于单目相机的端到端脑磁配准研究 — 训练数据准备情况</p>
    <p class="meta">📅 生成时间: {now} &nbsp;&nbsp;|&nbsp;&nbsp; 🏫 学位论文演示用</p>
</div>

<div class="stats-grid">
    <div class="stat-card success">
        <div class="label">已下载数据集</div>
        <div class="value">{n_avail} / {n_total}</div>
    </div>
    <div class="stat-card">
        <div class="label">总磁盘占用</div>
        <div class="value">{total_size}</div>
    </div>
    <div class="stat-card info">
        <div class="label">总样本量</div>
        <div class="value">{total_samples:,}</div>
    </div>
    <div class="stat-card warning">
        <div class="label">覆盖论文章节</div>
        <div class="value">Ch.3 / 4 / 5</div>
    </div>
</div>

<div class="container">

    <div class="section">
        <h2>📋 1. 数据集清单</h2>
        <p class="desc">所有训练数据集均来自 HuggingFace 开源社区，使用 hf-mirror 加速下载。
        受论文实验数据隐私性限制，使用以下公开数据集模拟和补充论文中提到的 MEG-Head-360 真实数据集。</p>
        <table>
            <thead>
            <tr>
                <th>数据集</th>
                <th>用途分类</th>
                <th>体积</th>
                <th>样本数</th>
                <th>状态</th>
            </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>

    <div class="section">
        <h2>📊 2. 数据集统计可视化</h2>
        <p class="desc">从体积、样本量、章节分布三个维度对比各数据集情况。</p>
        <div class="gallery">{overview_imgs}</div>
    </div>

    <div class="section">
        <h2>🖼️ 3. 数据样本示例</h2>
        <p class="desc">从每个数据集中随机抽取若干样本展示，含原图与语义 mask（如有）。</p>
        <div class="gallery">{sample_imgs}</div>
    </div>

    <div class="section">
        <h2>📈 4. 数据分布特征分析</h2>
        <p class="desc">对图像分辨率、亮度、颜色通道等关键统计特征进行分析，
        用于评估数据集的多样性与论文中"域随机化"策略的覆盖度。</p>
        <div class="gallery">{dist_imgs}</div>
        {dist_table_section}
    </div>

    <div class="section">
        <h2>📝 5. 数据使用说明</h2>
        <p class="desc">数据集与论文章节的对应关系如下：</p>
        <ul style="margin-left: 20px; line-height: 2;">
            <li><strong>第3章 - 头部分割</strong>:
                使用 <code>human_parsing</code>(主) + <code>celebA_faces</code> + <code>cppe5</code>
                + <code>protective_equipment</code> 作为 YOLOv8n-seg 训练数据，
                <code>afhqv2</code> 用于域外测试鲁棒性。</li>
            <li><strong>第4章 - NeRF 配准</strong>:
                使用 <code>aloha_multiview</code> 作为多视角数据格式参考；
                经典 NeRF-Synthetic 8场景需手动从原 Google Drive 下载。</li>
            <li><strong>第5章 - VFMReg 视觉基础模型配准</strong>:
                合成数据通过 Blender 程序化生成（脚本见 [augmentation.py](../preprocess/augmentation.py)）；
                <code>dog_food_bg</code> 与 <code>wikiart_bg</code> 提供背景域随机化。</li>
        </ul>
        <p class="desc" style="margin-top: 15px;">
            ⚠️ <strong>授权声明</strong>: 数据集仅用于学术研究目的（论文复现演示），
            CelebA 与 AFHQv2 限非商用，CPPE-5 与 Protective Equipment 为 CC-BY-4.0 协议。
        </p>
    </div>

    <div class="section">
        <h2>🚀 6. 复现步骤</h2>
        <pre style="background: #2c3e50; color: #ecf0f1; padding: 18px; border-radius: 8px; overflow-x: auto; font-size: 0.9em;">
# 1) 下载数据集 (使用国内镜像加速)
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; \\
    snapshot_download('mattmdjaga/human_parsing_dataset', repo_type='dataset', \\
                      local_dir='./data/head_seg/human_parsing')"

# 2) 转换为 YOLO 训练格式
python -m preprocess.convert_datasets --dataset human_parsing

# 3) 重新生成本报告
python demo_reports/01_dataset_overview.py
python demo_reports/02_sample_visualization.py
python demo_reports/03_data_distribution.py
python demo_reports/04_html_report.py</pre>
    </div>
</div>

<div class="footer">
    <p>💡 论文：基于单目相机的端到端脑磁配准研究</p>
    <p>© 学位论文配套代码 · 训练数据集说明文档 · 生成于 {now}</p>
</div>

<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
    <img id="lightbox-img" src="" alt="">
</div>

<script>
function openLightbox(src) {{
    document.getElementById('lightbox-img').src = src;
    document.getElementById('lightbox').classList.add('active');
}}
document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('active');
}});
</script>

</body>
</html>'''
    return html


def main():
    print('=' * 70)
    print('  生成 HTML 总结报告')
    print('=' * 70)

    print('\n📁 复制图表资源...')
    copied = copy_assets()
    print(f'  ✓ 复制 {len(copied)} 个图表')

    print('\n📄 加载统计数据...')
    overview_stats = load_json(OUTPUT_BASE / '01_overview/overview_stats.json')
    dist_stats = load_json(OUTPUT_BASE / '03_distribution/distribution_stats.json')
    if not overview_stats:
        print('  ⚠ 未找到 overview_stats.json，请先运行 01_dataset_overview.py')
    if not dist_stats:
        print('  ⚠ 未找到 distribution_stats.json，请先运行 03_data_distribution.py')

    print('\n📝 渲染 HTML...')
    html = render_html(overview_stats, dist_stats, copied)
    out = REPORT_DIR / 'index.html'
    out.write_text(html, encoding='utf-8')
    print(f'  ✓ 保存: {out}')

    print(f'\n✅ HTML 报告生成完成！')
    print(f'   📄 主文件: {out}')
    print(f'   🖼️ 资源: {ASSETS_DIR}')
    print(f'\n   👉 用浏览器打开 file://{out.resolve()} 查看\n')


if __name__ == '__main__':
    main()
