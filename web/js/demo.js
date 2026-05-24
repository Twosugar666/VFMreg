/* =========================================================
   demo.js - 在线演示模拟推理逻辑
   基于真实测试集结果回放，展示 VFMReg 推理过程
   ========================================================= */

// 5 个预设样本数据 (参考 results/ch3_segmentation_results.json)
const PRESETS = [
    {
        name: 'test_0001 (标准场景)',
        scene: 'standard',
        subject: 'subject_01',
        // 模拟真实结果
        R: [
            [0.99985, -0.01520, 0.00891],
            [0.01498,  0.99970, 0.01843],
            [-0.00922, -0.01823, 0.99979]
        ],
        t: [0.21, -0.18, 0.32],
        rot_err: 0.62,    // °
        trans_err: 0.55,  // mm
        time_ms: 14.8,
        seg_conf: 0.958,
        seg_miou: 95.7,
        seg_bf1: 90.3,
        seg_time: 8.5,
        // 视图占位（CSS 渐变模拟真实图像）
        views_color: ['#a8c5e6', '#c5d6e9', '#bdd0e6', '#a3bfdc'],
    },
    {
        name: 'test_0014 (低光)',
        scene: 'low_light',
        subject: 'subject_02',
        R: [
            [0.99962, 0.02431, -0.01302],
            [-0.02389, 0.99950, 0.02129],
            [0.01371, -0.02085, 0.99969]
        ],
        t: [0.39, 0.42, -0.51],
        rot_err: 0.71,
        trans_err: 0.68,
        time_ms: 15.2,
        seg_conf: 0.975,
        seg_miou: 93.9,
        seg_bf1: 89.1,
        seg_time: 7.25,
        views_color: ['#3a4a5e', '#465467', '#3f4d5e', '#384556'],
    },
    {
        name: 'test_0028 (遮挡)',
        scene: 'occlusion',
        subject: 'subject_03',
        R: [
            [0.99973, -0.01892, 0.01362],
            [0.01863, 0.99967, 0.02103],
            [-0.01401, -0.02077, 0.99975]
        ],
        t: [-0.32, 0.28, 0.18],
        rot_err: 0.68,
        trans_err: 0.62,
        time_ms: 15.5,
        seg_conf: 0.886,
        seg_miou: 95.3,
        seg_bf1: 94.6,
        seg_time: 8.24,
        views_color: ['#7a8a9e', '#8c9ab0', '#7e8c9e', '#74849a'],
    },
    {
        name: 'test_0046 (逆光)',
        scene: 'backlight',
        subject: 'subject_05',
        R: [
            [0.99981, 0.01452, -0.01290],
            [-0.01421, 0.99975, 0.02389],
            [0.01324, -0.02370, 0.99963]
        ],
        t: [0.15, -0.27, 0.41],
        rot_err: 0.65,
        trans_err: 0.58,
        time_ms: 14.5,
        seg_conf: 0.901,
        seg_miou: 94.8,
        seg_bf1: 95.1,
        seg_time: 7.52,
        views_color: ['#f0e8d0', '#ebe2c8', '#ede4cc', '#e8dfc5'],
    },
    {
        name: 'test_0040 (跨受试者)',
        scene: 'standard',
        subject: 'subject_05',
        R: [
            [0.99970, -0.02103, 0.01231],
            [0.02075, 0.99965, 0.02265],
            [-0.01278, -0.02239, 0.99967]
        ],
        t: [0.41, 0.18, -0.29],
        rot_err: 0.74,
        trans_err: 0.65,
        time_ms: 15.0,
        seg_conf: 0.881,
        seg_miou: 95.6,
        seg_bf1: 88.3,
        seg_time: 8.09,
        views_color: ['#b0c5d6', '#c2d3e0', '#b8c9d6', '#a8bdcf'],
    }
];

let currentPreset = 0;
let isRunning = false;

function loadPreset(idx) {
    if (isRunning) return;
    currentPreset = idx;
    document.querySelectorAll('.preset-btn').forEach((b, i) => {
        b.classList.toggle('active', i === idx);
    });

    // 更新视图卡片
    const preset = PRESETS[idx];
    const cards = document.querySelectorAll('#input-views .view-card');
    const viewLabels = ['Front', 'Left', 'Right', 'Top'];
    cards.forEach((c, i) => {
        const color = preset.views_color[i];
        // 简单的"模拟人头"圆形渐变
        c.style.background = `radial-gradient(circle at 50% 45%,
            ${shade(color, 30)} 0%,
            ${color} 35%,
            ${shade(color, -20)} 80%,
            ${shade(color, -40)} 100%)`;
        c.innerHTML = `<span class="view-label">${viewLabels[i]} - ${preset.scene}</span>`;
    });

    // 重置面板
    resetPanel();
    log('info', `已加载样本: ${preset.name}`);
    log('info', `场景: ${preset.scene} | 受试者: ${preset.subject}`);
}

function shade(hex, percent) {
    // 简单调色: percent 正为变亮，负为变暗
    const num = parseInt(hex.replace('#', ''), 16);
    let r = (num >> 16) + percent;
    let g = ((num >> 8) & 0x00FF) + percent;
    let b = (num & 0x0000FF) + percent;
    r = Math.min(255, Math.max(0, r));
    g = Math.min(255, Math.max(0, g));
    b = Math.min(255, Math.max(0, b));
    return '#' + ((r << 16) | (g << 8) | b).toString(16).padStart(6, '0');
}

function resetPanel() {
    setMetric('rot', 0, '— °');
    setMetric('trans', 0, '— mm');
    setMetric('time', 0, '— ms');
    setMetric('conf', 0, '— %');
    document.getElementById('status-bar').textContent = '等待运行...';
    document.getElementById('status-bar').style.background = 'var(--accent-soft)';
    document.getElementById('status-bar').style.color = 'var(--accent)';
}

function setMetric(name, pct, text) {
    document.getElementById(`m-${name}`).textContent = text;
    document.getElementById(`bar-${name}`).style.width = `${pct}%`;
}

function log(level, msg) {
    const console_ = document.getElementById('console');
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    const cls = { info: 'info', ok: 'ok', warn: 'warn', err: 'err' }[level] || 'info';
    console_.innerHTML += `\n<span class="ts">[${ts}]</span> <span class="${cls}">${msg}</span>`;
    console_.scrollTop = console_.scrollHeight;
}

function clearLog() {
    document.getElementById('console').innerHTML =
        '<span class="ts">[reset]</span> <span class="info">控制台已清空, 开始新一轮推理...</span>';
}

async function runInference() {
    if (isRunning) return;
    const btn = document.getElementById('run-btn');
    isRunning = true;
    btn.disabled = true;
    btn.textContent = '⏳ 推理中...';

    clearLog();
    resetPanel();

    const preset = PRESETS[currentPreset];
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));

    // 模拟推理 pipeline
    log('info', `▶ 启动 VFMReg 推理 pipeline (${preset.name})`);
    await sleep(300);

    log('info', '[1/5] 图像预处理 (resize 224×224, normalize)...');
    await sleep(200);
    log('ok', '  ✓ 4 个视图已就绪 (耗时 0.5 ms)');

    log('info', '[2/5] YOLOv8n-seg 头部分割...');
    await sleep(400);
    log('ok', `  ✓ 分割完成: mIoU=<span class="hl">${preset.seg_miou}</span>%, BF1=${preset.seg_bf1}%, conf=${(preset.seg_conf*100).toFixed(1)}% (耗时 ${preset.seg_time} ms)`);

    log('info', '[3/5] DINOv3 ViT-L/14 特征提取 (4 视图并行)...');
    await sleep(600);
    log('ok', '  ✓ 特征 shape: [4, 256, 1024] (耗时 10.0 ms)');

    log('info', '[4/5] 跨视角注意力融合 (Cross-View Attention)...');
    await sleep(300);
    log('ok', '  ✓ 融合特征 shape: [1, 1024] (耗时 1.5 ms)');

    log('info', '[5/5] 6DoF 姿态回归 (6D 旋转 + 3D 平移)...');
    await sleep(200);
    log('ok', `  ✓ 预测完成 (耗时 0.5 ms)`);

    await sleep(200);

    // 渲染结果
    log('info', '─────────────────────────────────────────');
    const m = preset.R;
    const matrixStr =
`R (3×3):
  [${m[0].map(v => v.toFixed(4).padStart(7)).join('  ')}]
  [${m[1].map(v => v.toFixed(4).padStart(7)).join('  ')}]
  [${m[2].map(v => v.toFixed(4).padStart(7)).join('  ')}]

t (mm):
  [${preset.t.map(v => v.toFixed(2).padStart(6)).join(', ')}]`;
    document.getElementById('pose-matrix').textContent = matrixStr;

    // 旋转误差 (越小越好, 取对数变换为百分比, 0.7° 对应约 87%)
    const rotPct = Math.max(0, 100 - preset.rot_err * 60);
    const transPct = Math.max(0, 100 - preset.trans_err * 70);
    const timePct = Math.max(0, 100 - preset.time_ms * 4);
    const confPct = preset.seg_conf * 100;

    setMetric('rot', rotPct, `${preset.rot_err.toFixed(2)} °`);
    setMetric('trans', transPct, `${preset.trans_err.toFixed(2)} mm`);
    setMetric('time', timePct, `${preset.time_ms.toFixed(1)} ms`);
    setMetric('conf', confPct, `${(preset.seg_conf*100).toFixed(1)} %`);

    log('ok', `🎯 配准成功 | 旋转误差: <span class="hl">${preset.rot_err.toFixed(2)}°</span> | 平移误差: <span class="hl">${preset.trans_err.toFixed(2)} mm</span>`);
    log('ok', `⚡ 总延迟: <span class="hl">${preset.time_ms.toFixed(1)} ms</span> (含分割), 满足实时性要求 (>50 FPS)`);

    // 评判
    const sb = document.getElementById('status-bar');
    if (preset.rot_err < 1.0 && preset.trans_err < 1.0) {
        sb.textContent = '✅ 配准成功 (满足亚毫米/亚度精度要求)';
        sb.style.background = '#d4edda';
        sb.style.color = 'var(--success)';
    } else {
        sb.textContent = '⚠ 精度退化 (建议检查输入图像质量)';
        sb.style.background = '#fff3cd';
        sb.style.color = 'var(--warning)';
    }

    isRunning = false;
    btn.disabled = false;
    btn.textContent = '▶ 运行 VFMReg 推理';
}

// 初始加载第一个样本
document.addEventListener('DOMContentLoaded', () => {
    loadPreset(0);
});
