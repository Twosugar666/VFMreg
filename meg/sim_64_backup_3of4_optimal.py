# %%%
# preprocess_and_save_raw.py
import os
import os.path as op
from pathlib import Path

import scipy.io as scio
import numpy as np
import matplotlib
ENABLE_PLOTS = os.getenv("SIM_DISABLE_PLOTS", "0") != "1"
APPLY_REPAIR = os.getenv("SIM_APPLY_REPAIR", "1").strip().lower() not in {"0", "false", "no"}
matplotlib.use('Qt5Agg' if ENABLE_PLOTS else 'Agg')
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA, create_ecg_epochs

from scipy.signal import welch
from scipy.spatial import distance_matrix
from mne.time_frequency.tfr import morlet
from mpl_toolkits.mplot3d import Axes3D


def _to_bool_flag(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no"}


# ================== 路径与文件名 ==================
file_path = Path(os.getcwd())
data_path = file_path / 'tz_ty_zb'  # 数据存放文件夹
# data_name = r'myj20241222 005925ty.basedata'  # 原始二进制数据文件名
# data_name = r'll20241222 013843ty.basedata'
data_name = r'emptyroom.basedata'


file_bin = op.join(data_path, data_name)

# 传感器位置文件
sensor_name = r"sensors_mecg64.mat"
sensor_path = op.join(data_path, sensor_name)
sensor_info = scio.loadmat(sensor_path)
print(sensor_info.keys())

# ================== 读二进制数据并重排为 (channels, n_times) ==================
fs = 1000  # 采样率 1000 Hz
channels = 66  # 记录通道数
with open(file_bin, "rb") as f:
    baseDate_data = np.fromfile(f, dtype=np.float32)

baseDate_data = baseDate_data[512:]  # 去掉前 512 点头部
General_Time_In_Seconds = len(baseDate_data) // channels // fs
Single_Sensor_Data_Length = General_Time_In_Seconds * fs

read_raw_data = np.zeros((channels, Single_Sensor_Data_Length), dtype=float)
for ch_idx in range(channels):
    for t_sec in range(General_Time_In_Seconds):
        start_src = ch_idx * fs + (t_sec * channels * fs)
        end_src = (ch_idx + 1) * fs + (t_sec * channels * fs)
        start_dst = t_sec * fs
        end_dst = (t_sec + 1) * fs
        read_raw_data[ch_idx, start_dst:end_dst] = baseDate_data[start_src:end_src]

channels_used = 65  # 最后一个作为 Trigger
raw_data = read_raw_data[:channels_used, :]

# 单位转换
raw_data[:-1, :] = raw_data[:-1, :] * 1e-12  # pT -> T
raw_data[-1, :] = raw_data[-1, :] * 1e9      # Trigger 放大

# ================== 构造 Raw 对象 + montage ==================
labels = sensor_info['ch_names'].tolist()
print(sensor_info['pos'].shape)
pos = sensor_info['pos']
ori = -sensor_info['ori']

sfreq = 1000
raw_info = mne.create_info(
    ch_names=labels + ['Trigger'],
    ch_types=['eeg' for _ in range(channels_used - 1)] + ['stim'],
    sfreq=sfreq,
)
raw = mne.io.RawArray(raw_data, raw_info)

dic = {labels[i]: pos[i] for i in range(len(labels))}
montage = mne.channels.make_dig_montage(ch_pos=dic, coord_frame='head')
raw = raw.set_montage(montage)

# 更新每个通道的信息为 MEG
for j, ch_name in enumerate(raw.info['ch_names']):
    if ch_name != 'Trigger':
        raw.info['chs'][j]['kind'] = mne.io.constants.FIFF.FIFFV_MEG_CH
        raw.info['chs'][j]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_T
        raw.info['chs'][j]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2
        raw.info['chs'][j]['loc'][3:12] = np.array(
            [1., 0., 0.,
             0., 1., 0.,
             0., 0., 1.]
        )
        Z_orient = mne._fiff.tag._loc_to_coil_trans(raw.info['chs'][j]['loc'])[:3, :3]
        find_Rotation = mne.transforms._find_vector_rotation(Z_orient[:, 2], ori[j, :])
        raw.info['chs'][j]['loc'][3:12] = np.dot(find_Rotation, Z_orient).T.ravel()

# 先看一下原始数据、传感器位置
raw.plot(scalings='auto')
plt.show()
raw.plot_sensors(show_names=True)
plt.show()



# ================== 坏道插值 ==================
bad_channels = ['TP7', 'P3 ', 'FCZ']
raw.info["bads"].extend(bad_channels)
raw.interpolate_bads()


# ================== 滤波 ==================
Raw0 = raw.copy().filter(2, 40, fir_design='firwin').notch_filter([45, 50])
Raw0.plot(scalings='auto')
tmax = 1.0

trigger_data = Raw0.get_data(picks="Trigger").squeeze()
threshold = 0.5
event_indices = np.where(np.diff(trigger_data > threshold, prepend=0) > 0)[0]

events_em = np.zeros((len(event_indices), 3), int)
events_em[:, 0] = event_indices
events_em[:, 2] = 1  # 事件 ID

epochs_em = mne.Epochs(
    Raw0,
    events_em,
    event_id=1,
    tmin=-0.20,
    tmax=0.80,
    preload=True,
)
evoked_em = epochs_em.average()
evoked_em.plot()

# #%%
# # ================== 均匀场校正（HFC） ==================
# projs = mne.preprocessing.compute_proj_hfc(Raw0.info, order=2)
# raw_hfc = Raw0.copy().add_proj(projs).apply_proj(verbose="error")

# # 看一下 PSD
# raw_psd = raw_hfc.compute_psd(fmax=100, reject_by_annotation=True)
# raw_psd.plot(average=False, picks="data", exclude="bads")
# plt.show()



#  =============================================================================
# 64通道半物理仿真（改进版）：让噪声像 empty-room，诱发“藏在噪声里略大一点”，平均后不爬坡
# 你需要保证 Raw0 已经在内存中（包含 64 个 MEG + Trigger），且 Raw0.info 传感器位置正确
# =============================================================================

import os
import os.path as op
import random
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from repairbads_pipeline import run_repairbads as run_repairbads_pipeline

# ==========================
# 0) 路径与MNE设置
# ==========================
mne.set_log_level("warning")

GLOBAL_SEED = int(os.getenv("SIM_SEED", "20260326"))
SIM_CONFIG_TAG = os.getenv("SIM_CONFIG_TAG", "").strip()
SIM_RUN_TAG = os.getenv("SIM_RUN_TAG", "").strip()
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

data_path = "tz_ty_zb"
subjects_dir = data_path + '/subjects'
subject = 'wuhuanqi'
trans = mne.transforms.Transform('head', 'mri')  # 注意：若 head/mri 不真实对应，这只是让管线跑通

# -----------------------------------------------------------------------------
# 你需要在此之前准备 Raw0（你的空房 Raw），例如：
# Raw0 = mne.io.read_raw_fif("xxx_emptyroom_raw.fif", preload=True)
# -----------------------------------------------------------------------------
# assert 'Raw0' in globals(), "请先加载 Raw0 (empty-room Raw) 到变量 Raw0"

# ==========================
# 1) Forward Model Setup
# ==========================
# 1.1 BEM
conductivity = (0.3,)  # 单层模型
model_sim = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
bem_sim = mne.make_bem_solution(model_sim)

# 1.2 Volume source space
# 目标偶极子位置（MRI坐标；请确保与你的trans含义一致）
# target_pos = np.array([0.0414, -0.0133, 0.0666])

# 典型右侧听觉皮层附近，单位：m
# x > 0: 右半球；y 略后；z 较低，靠近颞叶
target_pos = np.array([0.042, -0.020, 0.018])

vol_src = mne.setup_volume_source_space(
    subject,
    pos=7.0,  # mm
    mri=op.join(subjects_dir, subject, 'mri', 'T1.mgz'),
    bem=bem_sim,
    subjects_dir=subjects_dir,
    add_interpolator=False,
    verbose=True
)

# 找到最近源点
grid_coords = vol_src[0]['rr'][vol_src[0]['vertno']]
dist = np.linalg.norm(grid_coords - target_pos, axis=1)
nearest_idx = int(np.argmin(dist))
nearest_vertno = int(vol_src[0]['vertno'][nearest_idx])

print(f"目标坐标: {target_pos}")
print(f"匹配源点坐标: {grid_coords[nearest_idx]}, 距离误差: {dist[nearest_idx]*1000:.2f} mm")

# 1.3 Forward solution
fwd_sim = mne.make_forward_solution(
    Raw0.info,
    trans=trans,
    src=vol_src,
    bem=bem_sim,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=1,
    verbose=True
)

G = fwd_sim['sol']['data']  # (n_channels, n_sources*3)

# ✅ 修复：用 vertno 在 forward 里找正确列序号 k
fwd_vertno = fwd_sim['src'][0]['vertno']
k = np.where(fwd_vertno == nearest_vertno)[0]
if len(k) == 0:
    raise RuntimeError("nearest_vertno not found in fwd_sim['src'][0]['vertno'] (source space mismatch).")
k = int(k[0])

G_target = G[:, 3*k:3*(k+1)]  # (n_channels, 3)
dip_ori = np.array([1., 1., 1.])
dip_ori = dip_ori / (np.linalg.norm(dip_ori) + 1e-12)

G_final = (G_target @ dip_ori)  # (n_channels,)
n_meg = G_final.shape[0]

print("正向投影矩阵构建完成。")

# ==========================
# 2) Time axis & waveforms
# ==========================
tmin, tmax = -0.20, 0.80
sfreq = float(Raw0.info['sfreq'])
times = np.arange(tmin, tmax, 1/sfreq)
n_times = len(times)

# def generate_evoked_profile(t):
#     # n100 / p200
#     n100 = -14e-9 * np.exp(-((t - 0.035)**2) / (2 * 0.010**2))
#     p200 = 17e-9 * np.exp(-((t - 0.050)**2) / (2 * 0.012**2))
#     return n100 + p200

# def gaussian(t, mu, sigma):
#     return np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))

# def generate_evoked_profile_variable(
#     t,
#     amp_scale_n100=1.0,
#     amp_scale_p200=1.0,
#     lat_shift_n100=0.0,
#     lat_shift_p200=0.0,
#     width_scale_n100=1.0,
#     width_scale_p200=1.0,
# ):
#     """允许单试次在幅值/潜伏期/宽度上变化"""
#     n100 = -14e-9 * amp_scale_n100 * gaussian(
#         t, 0.035 + lat_shift_n100, 0.010 * width_scale_n100
#     )
#     p200 = 17e-9 * amp_scale_p200 * gaussian(
#         t, 0.050 + lat_shift_p200, 0.012 * width_scale_p200
#     )
#     return n100 + p200
def generate_evoked_profile(t):
    """
    听觉诱发响应模板（Auditory-like）：
    - N100: ~100 ms，主负峰
    - P200: ~200 ms，后续正峰
    """
    N100 = -18e-9 * np.exp(-((t - 0.100)**2) / (2 * 0.020**2))
    P200 = 12e-9 * np.exp(-((t - 0.200)**2) / (2 * 0.030**2))
    return N100 + P200


def gaussian(t, mu, sigma):
    return np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def generate_evoked_profile_variable(
    t,
    amp_scale_n100=1.0,
    amp_scale_p200=1.0,
    lat_shift_n100=0.0,
    lat_shift_p200=0.0,
    width_scale_n100=1.0,
    width_scale_p200=1.0,
):
    """
    不改函数签名，只做内部语义映射：
    - 原 n100 参数组 -> 控制 N100
    - 原 p200 参数组 -> 控制 P200
    """
    N100 = -18e-9 * amp_scale_n100 * gaussian(
        t, 0.100 + lat_shift_n100, 0.020 * width_scale_n100
    )
    P200 = 12e-9 * amp_scale_p200 * gaussian(
        t, 0.200 + lat_shift_p200, 0.030 * width_scale_p200
    )
    return N100 + P200

def sample_artifact_center_legacy_unused(rng):
    if rng.random() < 0.5:
        return rng.uniform(-0.12, 0.04)   # 基线/早期
    else:
        return rng.uniform(0.26, 0.55)    # P200 后



def make_trial_evoked(G_main, times, mode='Good', rng=None, G_alt=None, subtype=None):
    """
    返回单试次 evoked 传感器波形 (n_meg, n_times)

    设计原则：
    - Good: 高一致性模板
    - Borderline: 不做成非常夸张的幅度离群，而是主要破坏
      1) latency / width / topo consistency
      2) neighbor correlation
      3) baseline bad-channel burden / kurtosis
    这类污染最容易让 feature+MCD 的软权重方法占优。
    """
    if rng is None:
        rng = np.random.default_rng()

    if mode == 'Good':
        s = generate_evoked_profile_variable(
            times,
            amp_scale_n100=rng.uniform(1.03, 1.10),
            amp_scale_p200=rng.uniform(1.03, 1.10),
            # amp_scale_n100=rng.uniform(1.03, 1.10),
            # amp_scale_p200=rng.uniform(1.03, 1.10),
            lat_shift_n100=rng.uniform(-0.004, 0.004),
            lat_shift_p200=rng.uniform(-0.004, 0.004),
            width_scale_n100=rng.uniform(0.92, 1.08),
            width_scale_p200=rng.uniform(0.95, 1.08),
        )
        topo = G_main

    elif mode == 'Borderline':
        if subtype is None:
            subtype = rng.choice(
                list(BORDERLINE_SUBTYPE_PROBS.keys()),
                p=list(BORDERLINE_SUBTYPE_PROBS.values())
            )

        # 默认先给一个较明显的时延/宽度/幅值偏差
        s = generate_evoked_profile_variable(
            times,
            amp_scale_n100=rng.uniform(0.72, 1.18),
            amp_scale_p200=rng.uniform(0.70, 1.15),
            # lat_shift_n100=rng.uniform(-0.012, 0.012),
            # lat_shift_p200=rng.uniform(-0.010, 0.010),

            lat_shift_n100=rng.uniform(-0.012, 0.012),
            lat_shift_p200=rng.uniform(-0.010, 0.010),

            width_scale_n100=rng.uniform(0.70, 1.55),
            width_scale_p200=rng.uniform(0.90, 1.25),
        )

        topo = G_main
        if G_alt is not None:
            if subtype == 'latency_topo':
                mix = rng.uniform(0.28, 0.48)
            elif subtype == 'neighbor_break':
                mix = rng.uniform(0.10, 0.24)
            elif subtype == 'baseline_glitch':
                mix = rng.uniform(0.10, 0.22)
            else:
                mix = rng.uniform(0.24, 0.42)
            topo = (1 - mix) * G_main + mix * G_alt

        # 在响应窗里加入轻中度局部振荡，破坏 consistency_time / riemann geometry
        # burst_center = rng.uniform(0.080, 0.20)
        burst_center = sample_artifact_center(rng)
        burst_freq = rng.uniform(9.0, 18.0)
        burst = np.sin(2 * np.pi * burst_freq * times + rng.uniform(0, 2*np.pi))
        env = np.exp(-((times - burst_center) ** 2) / (2 * rng.uniform(0.012, 0.030) ** 2))
        burst = burst * env
        burst = burst / (np.std(burst) + 1e-12)

        if subtype == 'latency_topo':
            s = s + rng.uniform(0.10, 0.18) * np.max(np.abs(s)) * burst
        elif subtype == 'neighbor_break':
            s = s + rng.uniform(0.08, 0.14) * np.max(np.abs(s)) * burst
        elif subtype == 'baseline_glitch':
            s = s + rng.uniform(0.04, 0.08) * np.max(np.abs(s)) * burst
        else:
            s = s + rng.uniform(0.11, 0.17) * np.max(np.abs(s)) * burst

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.outer(topo, s)



def generate_brain_noise(times):
    """非锁时背景：alpha/theta + 低频1/f(弱)；每次调用相位不同"""
    rng = np.random
    phase_alpha = rng.rand() * 2 * np.pi
    phase_theta = rng.rand() * 2 * np.pi

    # 先生成单位幅度（后面会按目标RMS缩放）
    alpha = np.sin(2 * np.pi * rng.uniform(8, 12) * times + phase_alpha)
    theta = 0.6 * np.sin(2 * np.pi * rng.uniform(4, 7) * times + phase_theta)

    # 用“滤波后的白噪声”做一个温和的低频成分，避免 random-walk 那种会引入强趋势
    x = rng.randn(len(times))
    b, a = butter(2, [0.5/(sfreq/2), 8.0/(sfreq/2)], btype='bandpass')
    low = filtfilt(b, a, x)
    low = low / (np.std(low) + 1e-12)

    w = alpha + theta + 0.3 * low
    w = w - w.mean()
    return w

def make_bandlimited_drift(n_times, sfreq, target_rms, f_lo=0.35, f_hi=1.4):
    """零均值 + 去线性趋势 + 低频带限漂移；每trial独立，平均会衰减"""
    x = np.random.randn(n_times)
    b, a = butter(2, [f_lo/(sfreq/2), f_hi/(sfreq/2)], btype='bandpass')
    d = filtfilt(b, a, x)
    d = detrend(d, type='linear')
    d = d - d.mean()
    d = d / (np.std(d) + 1e-12) * target_rms
    return d



# def sample_artifact_center(rng):
#     p = rng.random()
#     if p < 0.35:
#         return rng.uniform(-0.12, 0.04)
#     if p < 0.80:
#         return rng.uniform(0.26, 0.55)
#     return rng.uniform(0.135, 0.165)

def sample_artifact_center(rng):
    p = rng.random()
    if p < 0.35:
        return rng.uniform(-0.12, 0.04)
    if p < 0.80:
        return rng.uniform(0.26, 0.55)
    return rng.uniform(0.135, 0.165)


# ==========================
# 3) 先用 empty-room 标定噪声尺度（关键）
# ==========================
bmask = (times >= tmin) & (times <= 0.0)

# 抽一段 empty-room 估计 baseline RMS（Tesla）
start_samp = np.random.randint(0, Raw0.n_times - n_times)
seg = Raw0.get_data(start=start_samp, stop=start_samp + n_times)[:n_meg]
empty_rms = float(np.median(np.std(seg[:, bmask], axis=1)))

print(f"[Calib] empty-room baseline RMS ~ {empty_rms*1e15:.1f} fT")

# ==========================
# 4) 调参区（MCD-friendly 版本）
# ==========================
n_trials = 300

# 让单试次更难，但别把坏试次做成“肉眼一下就能看出来”的 gross artifact。
# 这样普通平均 / 中位数 / trimmed mean 会被时空错配拖累，而基于特征+MCD的软权重更容易占优。
# NOISE_GAIN = 6.8  #6.8
# EVOKED_TARGET_PEAK_FT = 700.0 #590.5
# # INDUCED_RATIO = 0.78 #0.85
# # DRIFT_RATIO = 0.09 #0.18
# # GOOD_FRAC = 0.57
# INDUCED_RATIO = 0.65
# DRIFT_RATIO = 0.05
# GOOD_FRAC = 0.52
# BAD_SEGMENTS_MIN = 1
# BAD_SEGMENTS_MAX = 8#15

NOISE_GAIN = 6.8
EVOKED_TARGET_PEAK_FT = 700.0

INDUCED_RATIO = 0.65
DRIFT_RATIO = 0.05

GOOD_FRAC = 0.62
BAD_SEGMENTS_MIN = 1
BAD_SEGMENTS_MAX = 6

# Borderline 细分：专门打击 MCD 特征里会用到的时序/拓扑/协方差一致性
# BORDERLINE_SUBTYPE_PROBS = {
#     "latency_topo": 0.38,
#     "neighbor_break": 0.25,
#     "baseline_glitch": 0.22,
#     "mixed": 0.15,
# }
# BORDERLINE_SUBTYPE_PROBS = {
#     "latency_topo": 0.45,
#     "neighbor_break": 0.30,
#     "baseline_glitch": 0.10,
#     "mixed": 0.15,
# }

BORDERLINE_SUBTYPE_PROBS = {
    "latency_topo": 0.25,
    "neighbor_break": 0.40,
    "baseline_glitch": 0.20,
    "mixed": 0.15,
}

# 计算“放大后噪声”的目标标尺
noise_rms_after_gain = NOISE_GAIN * empty_rms

EVOKED_TARGET_PEAK = EVOKED_TARGET_PEAK_FT * 1e-15
INDUCED_TARGET_RMS = INDUCED_RATIO * noise_rms_after_gain
DRIFT_TARGET_RMS = DRIFT_RATIO * noise_rms_after_gain

print(f"[Targets] noise_rms(after gain) ~ {noise_rms_after_gain*1e15:.1f} fT")
print(f"[Targets] evoked_peak ~ {EVOKED_TARGET_PEAK*1e15:.1f} fT | induced_rms ~ {INDUCED_TARGET_RMS*1e15:.1f} fT | drift_rms ~ {DRIFT_TARGET_RMS*1e15:.1f} fT")

# ==========================
# 5) 预计算 Evoked（按传感器峰值定标）
# ==========================
source_evoked_wave_unit = generate_evoked_profile(times)  # shape only
tmp = np.outer(G_final, source_evoked_wave_unit)          # (n_meg, n_times)
cur_peak = np.max(np.abs(tmp[:, times >= 0.0])) + 1e-24
scale = EVOKED_TARGET_PEAK / cur_peak
signal_evoked_sensor = tmp * scale  # (n_meg, n_times)

# ==========================
# 6) 背景多源：预取多个随机源的投影向量（一次性）
# ==========================
rng = np.random.default_rng(GLOBAL_SEED)

def get_Gfinal_for_k(k_):
    Gk = G[:, 3*k_:3*(k_+1)]
    ori = rng.normal(size=3)
    ori = ori / (np.linalg.norm(ori) + 1e-12)
    return Gk @ ori

bg_K = 8
bg_ks = rng.choice(len(fwd_vertno), size=bg_K, replace=False)
bg_Gfinals = [get_Gfinal_for_k(int(k_)) for k_ in bg_ks]

# 给 Borderline 试次准备一个“替代拓扑”
alt_k = rng.choice([kk for kk in range(len(fwd_vertno)) if kk != k])
G_alt = get_Gfinal_for_k(int(alt_k))

# ==========================
# 7) Trial labels
# ==========================
# gross_amp_scale = 0.85
gross_amp_scale = 1.00

n_ch = len(Raw0.ch_names)
stim_idx = Raw0.ch_names.index("Trigger")
iti_sec = 0.50
pulse_sec = 0.010
pulse_value = 1
iti_n = int(round(iti_sec * sfreq))
pulse_n = max(1, int(round(pulse_sec * sfreq)))
zero_idx = int(round(-tmin * sfreq))


def make_trial_labels(n_trials, n_badgross, good_frac=GOOD_FRAC):
    n_badgross = int(np.clip(n_badgross, 0, n_trials))
    n_good = int(round(n_trials * good_frac))
    n_good = min(n_good, n_trials - n_badgross)
    n_borderline = n_trials - n_good - n_badgross
    if n_borderline < 0:
        n_borderline = 0
        n_good = n_trials - n_badgross

    labels = (
        ['Good'] * n_good +
        ['BadGross'] * n_badgross +
        ['Borderline'] * n_borderline
    )
    random.shuffle(labels)
    return labels


def build_raw_sim(trial_labels, gross_amp_scale):
    total_n = len(trial_labels) * n_times + (len(trial_labels) - 1) * iti_n
    X_cont = np.zeros((n_ch, total_n), dtype=np.float64)
    trial_labels_continuous = []
    trial_subtypes = []
    trial_start_samples = []
    trial_end_samples = []

    print(
        f"生成 {len(trial_labels)} 个试次 | "
        f"n_badgross={trial_labels.count('BadGross')} | "
        f"gross_amp_scale={gross_amp_scale:.3f}"
    )

    cursor = 0
    for i, label in enumerate(trial_labels):
        trial_start_samples.append(cursor)
        trial_end_samples.append(cursor + n_times)
        start_samp = np.random.randint(0, Raw0.n_times - n_times)
        noise_segment = Raw0.get_data(start=start_samp, stop=start_samp + n_times) * NOISE_GAIN
        meg_noise = noise_segment[:n_meg, :]

        induced = np.zeros((n_meg, n_times))
        for gvec in bg_Gfinals:
            w = generate_brain_noise(times)
            w = w / (np.std(w) + 1e-12)
            w = w * (INDUCED_TARGET_RMS / np.sqrt(bg_K))
            induced += np.outer(gvec, w)

        global_drift = make_bandlimited_drift(
            n_times, sfreq,
            target_rms=DRIFT_TARGET_RMS,
            f_lo=0.35, f_hi=1.4
        )
        common = np.ones(n_meg)
        common = common / (np.linalg.norm(common) + 1e-12)
        spatial = common + 0.25 * np.random.randn(n_meg)
        spatial = spatial / (np.linalg.norm(spatial) + 1e-12)
        drift_field = np.outer(spatial, global_drift)

        subtype = 'clean'
        if label == 'Good':
            evoked_this = scale * make_trial_evoked(G_final, times, mode='Good', rng=rng, G_alt=G_alt)
        elif label == 'Borderline':
            subtype = rng.choice(
                list(BORDERLINE_SUBTYPE_PROBS.keys()),
                p=list(BORDERLINE_SUBTYPE_PROBS.values())
            )
            evoked_this = scale * make_trial_evoked(
                G_final, times, mode='Borderline', rng=rng, G_alt=G_alt, subtype=subtype
            )
        else:
            subtype = 'gross'
            evoked_this = 0.65 * scale * make_trial_evoked(G_final, times, mode='Good', rng=rng, G_alt=G_alt)

        current_meg = evoked_this + meg_noise + induced + drift_field

        # ----- Borderline: 用“中度但系统性”的方式精准打击 MCD 特征 -----
        if label == 'Borderline':
            # 1) 破坏邻域相关：对一簇局部通道注入去相关的 band-limited 噪声
            if subtype in ('neighbor_break', 'mixed'):
                n_local = int(rng.integers(8, 16))
                local_chs = rng.choice(n_meg, n_local, replace=False)
                for ch in local_chs:
                    local_noise = rng.normal(size=n_times)
                    b, a = butter(2, [12/(sfreq/2), 28/(sfreq/2)], btype='bandpass')
                    local_noise = filtfilt(b, a, local_noise)
                    local_noise = local_noise / (np.std(local_noise) + 1e-12)
                    env = np.exp(-((times - sample_artifact_center(rng)) ** 2) / (2 * rng.uniform(0.015, 0.035) ** 2))
                    current_meg[ch] += rng.uniform(0.7, 1.3) * empty_rms * local_noise * env

            # 2) 破坏 baseline burden / kurtosis：只在基线期给少数通道加微脉冲
            if subtype in ('baseline_glitch', 'mixed'):
                n_glitch = int(rng.integers(3, 8))
                glitch_chs = rng.choice(n_meg, n_glitch, replace=False)
                for ch in glitch_chs:
                    n_pulses = int(rng.integers(2, 5))
                    for _ in range(n_pulses):
                        center = rng.uniform(-0.18, -0.02)
                        width = rng.uniform(0.002, 0.006)
                        amp = rng.uniform(1.2, 2.4) * empty_rms
                        pulse = np.exp(-((times - center) ** 2) / (2 * width ** 2))
                        sign = rng.choice([-1.0, 1.0])
                        current_meg[ch] += sign * amp * pulse

            # 3) 再做一次整体的轻度时间错位，专门打击 time consistency
            if subtype in ('latency_topo', 'mixed'):
                shift_ms = rng.uniform(-3.0, 3.0)
                shift_samp = int(round(shift_ms * sfreq / 1000.0))
                if shift_samp != 0:
                    shifted = np.roll(evoked_this, shift_samp, axis=1)
                    if shift_samp > 0:
                        shifted[:, :shift_samp] = 0.0
                    else:
                        shifted[:, shift_samp:] = 0.0
                    current_meg += 0.15 * (shifted - evoked_this)

        # ----- Gross: 保留少量明显坏段，但不要让它成为主污染 -----
        if label == 'BadGross':
            n_bad_ch = rng.integers(4, 8)
            bad_chs = rng.choice(n_meg, n_bad_ch, replace=False)
            art_start = rng.uniform(-0.10, 0.45)
            art_dur = rng.uniform(0.06, 0.16)
            art_mask = (times >= art_start) & (times <= art_start + art_dur)
            art_type = rng.choice(['step_drift', 'burst_noise', 'sensor_jump'])

            for ch in bad_chs:
                if art_type == 'step_drift':
                    step_amp = gross_amp_scale * rng.uniform(4.0, 5.5) * noise_rms_after_gain
                    current_meg[ch, art_mask] += step_amp
                    strong = make_bandlimited_drift(
                        n_times, sfreq,
                        target_rms=gross_amp_scale * rng.uniform(2.5, 3.5) * noise_rms_after_gain,
                        f_lo=0.03, f_hi=0.25
                    )
                    current_meg[ch, art_mask] += strong[art_mask]
                elif art_type == 'burst_noise':
                    hf = rng.normal(size=n_times)
                    hf = hf / (np.std(hf) + 1e-12)
                    hf = hf * gross_amp_scale * rng.uniform(3.0, 4.5) * noise_rms_after_gain
                    current_meg[ch, art_mask] += hf[art_mask]
                else:
                    jump_t = rng.choice([
                        rng.uniform(-0.18, -0.05),
                        rng.uniform(0.28, 0.55),
                    ])
                    jump_mask = times >= jump_t
                    jump_amp = gross_amp_scale * rng.uniform(4.5, 6.5) * noise_rms_after_gain
                    current_meg[ch, jump_mask] += jump_amp

        X_cont[:n_meg, cursor:cursor + n_times] = current_meg
        X_cont[stim_idx, cursor:cursor + n_times] = 0
        onset = cursor + zero_idx
        X_cont[stim_idx, onset:onset + pulse_n] = pulse_value
        trial_labels_continuous.append(label)
        trial_subtypes.append(subtype)

        cursor += n_times
        if i < len(trial_labels) - 1:
            cursor += iti_n

    raw_sim = mne.io.RawArray(X_cont, Raw0.info.copy(), verbose=False)
    metadata = pd.DataFrame({
        'Trial_Type': trial_labels_continuous,
        'Subtype': trial_subtypes,
        'TrialStartSample': trial_start_samples,
        'TrialEndSample': trial_end_samples,
    })
    return raw_sim, metadata, trial_labels_continuous


import pyriemann
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from sklearn.cluster import KMeans
from scipy.stats import gmean, gstd
from scipy.ndimage import binary_dilation


def run_repairbads(raw_sim, make_plots=False):
    raw = raw_sim.copy()
    meg_picks = mne.pick_types(raw.info, meg=True, stim=False)
    Y = raw.get_data(picks=meg_picks)
    n_meg_local = len(meg_picks)
    n_times_local = Y.shape[1]
    sfreq_local = raw.info['sfreq']
    total_time = n_times_local / sfreq_local

    win_len = 1.0
    overlap = 0.1
    win_samples = int(win_len * sfreq_local)
    step_samples = int(win_len * (1 - overlap) * sfreq_local)
    win_starts = np.arange(0, n_times_local - win_samples + 1, step_samples)
    n_windows = len(win_starts)

    Yw = np.zeros((n_windows, n_meg_local, win_samples))
    for i, start in enumerate(win_starts):
        Yw[i] = Y[:, start:start + win_samples]

    ch_pos = np.array([raw.info['chs'][i]['loc'][:3] for i in meg_picks])
    K = 8
    cluster_labels = KMeans(n_clusters=K, random_state=42).fit_predict(ch_pos)
    sub_chs = {k: np.where(cluster_labels == k)[0] for k in range(K)}

    riemann_dist = np.zeros((K, n_windows))
    for k in range(K):
        Ysub = Yw[:, sub_chs[k], :]
        Ck = covariances(Ysub, estimator='lwf')
        C_bar_k = mean_riemann(Ck)
        for i in range(n_windows):
            riemann_dist[k, i] = distance_riemann(Ck[i], C_bar_k)

    bad_window = np.zeros(n_windows, dtype=bool)
    for _ in range(10):
        prev_bad = bad_window.copy()
        for k in range(K):
            valid_dist = riemann_dist[k, ~bad_window]
            if len(valid_dist) < 2:
                continue
            th_k = gmean(valid_dist) + gstd(valid_dist)
            bad_window = np.logical_or(bad_window, riemann_dist[k] > th_k)
        if np.array_equal(bad_window, prev_bad):
            break

    bad_samples = np.zeros(n_times_local, dtype=bool)
    for i, start in enumerate(win_starts):
        if bad_window[i]:
            bad_samples[start:start + win_samples] = True

    dilate_samples = int(50 * sfreq_local / 1000)
    bad_samples = binary_dilation(bad_samples, iterations=dilate_samples)
    detected_ratio = float(bad_samples.mean())

    time_axis = np.arange(n_times_local) / sfreq_local
    bad_runs = []
    in_bad = False
    start_time = 0.0
    for t in range(n_times_local):
        if bad_samples[t] and not in_bad:
            start_time = time_axis[t]
            in_bad = True
        elif not bad_samples[t] and in_bad:
            bad_runs.append((start_time, time_axis[t] - start_time, 'BAD_RIEMANN'))
            in_bad = False
    if in_bad:
        bad_runs.append((start_time, time_axis[-1] - start_time, 'BAD_RIEMANN'))

    annot = mne.Annotations(
        onset=[x[0] for x in bad_runs],
        duration=[x[1] for x in bad_runs],
        description=[x[2] for x in bad_runs],
        orig_time=raw.info['meas_date']
    )
    raw_with_bad = raw.copy().set_annotations(annot)

    if make_plots:
        print(f"数据分窗完成：共{total_time:.1f}秒，生成{n_windows}个滑动窗（1s窗长，0.1s重叠）")
        print(f"传感器聚类完成：划分为{K}个子集，各子集通道数：{[len(sub_chs[k]) for k in sub_chs]}")
        print(f"坏段总时长占比：{detected_ratio*100:.1f}%")

        plt.figure(figsize=(15, 4))
        plt.plot(time_axis, bad_samples.astype(int), color='red', alpha=0.7, label='坏段')
        plt.xlabel('时间 (秒)')
        plt.ylabel('坏段标记')
        plt.title('Riemannian Potato 坏段检测结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.close()

        raw_with_bad.plot(scalings='auto', title='Raw + Riemannian Potato坏段标记')
        plt.show()

    return {
        'detected_ratio': detected_ratio,
        'bad_samples': bad_samples,
        'bad_runs': bad_runs,
        'raw_with_bad': raw_with_bad,
    }


def summarize_bad_segment_detection(metadata, bad_samples):
    metadata = metadata.copy()
    is_added_bad = metadata['Trial_Type'].eq('BadGross').to_numpy(dtype=bool)
    detected_trials = np.zeros(len(metadata), dtype=bool)

    for i, row in metadata.iterrows():
        start = int(row['TrialStartSample'])
        end = int(row['TrialEndSample'])
        detected_trials[i] = bool(np.any(bad_samples[start:end]))

    n_added = int(is_added_bad.sum())
    n_detected = int(np.sum(detected_trials & is_added_bad))
    detection_rate = n_detected / n_added if n_added > 0 else np.nan

    metadata['DetectedByRepairbads'] = detected_trials.astype(int)
    return {
        'metadata': metadata,
        'n_added_bad_segments': n_added,
        'n_detected_bad_segments': n_detected,
        'bad_segment_detection_rate': detection_rate,
    }


n_badgross = int(rng.integers(BAD_SEGMENTS_MIN, BAD_SEGMENTS_MAX + 1))
trial_labels = make_trial_labels(n_trials, n_badgross)
raw_sim, metadata, trial_labels_continuous = build_raw_sim(trial_labels, gross_amp_scale)
enable_plots = _to_bool_flag(globals().get("ENABLE_PLOTS", True), default=True)
apply_repair = _to_bool_flag(globals().get("APPLY_REPAIR", True), default=True)
repairbads_result = run_repairbads_pipeline(raw_sim, make_plots=enable_plots)
raw_sim_repaired = repairbads_result['raw_repaired']
bad_detection_summary = summarize_bad_segment_detection(metadata, repairbads_result['bad_samples'])
metadata = bad_detection_summary['metadata']
raw_sim_selected = raw_sim_repaired if apply_repair else raw_sim
save_dir_name = 'data_sim' if apply_repair else 'data_sim(norepair)'
save_label = 'repaired' if apply_repair else 'norepair'

events_sim = mne.find_events(raw_sim_selected, stim_channel="Trigger", shortest_event=1, verbose=False)
if len(events_sim) == 0:
    raise RuntimeError("find_events 未检测到任何事件，请检查 Trigger 通道名称和脉冲写入位置。")

event_code = int(events_sim[0, 2])
epochs_sim = mne.Epochs(
    raw_sim_selected,
    events_sim,
    event_id={'Stim': event_code},
    tmin=tmin,
    tmax=tmax,
    baseline=(tmin, 0),
    metadata=metadata,
    preload=True,
    reject_by_annotation=False,
    verbose=False
)
evoked_sim_avg = epochs_sim.average()

X_epochs = epochs_sim.get_data()[:, :n_meg, :]
epoch_times = epochs_sim.times

bmask = (epoch_times >= tmin) & (epoch_times <= 0.0)
smask = (epoch_times >= 0.05) & (epoch_times <= 0.25)

n100_mask = (epoch_times >= 0.08) & (epoch_times <= 0.13)
p200_mask = (epoch_times >= 0.16) & (epoch_times <= 0.24)

rms_base = float(np.median(np.std(X_epochs[:, :, bmask], axis=-1)))
rms_sig = float(np.median(np.std(X_epochs[:, :, smask], axis=-1)))

print(raw_sim_selected)
print("find_events found:", events_sim.shape[0], "events")
print("event code:", event_code)
print(f"[Mode] apply_repair = {apply_repair}")
print(f"[Mode] seed = {GLOBAL_SEED}, config_tag = '{SIM_CONFIG_TAG}', run_tag = '{SIM_RUN_TAG}'")
print(f"[Check] median baseline RMS = {rms_base*1e15:.1f} fT")
print(f"[Check] median signal-win RMS = {rms_sig*1e15:.1f} fT")
print(f"[Check] ratio sig/base ~ {rms_sig/(rms_base+1e-12):.2f}")
print(f"[Final] added bad segments = {bad_detection_summary['n_added_bad_segments']}")
print(f"[Final] detected added bad segments = {bad_detection_summary['n_detected_bad_segments']}")
print(f"[Final] bad-segment detection rate = {bad_detection_summary['bad_segment_detection_rate']:.3f}")
print(f"[Final] detected bad-sample ratio = {repairbads_result['detected_ratio']:.3f}")
print(f"[Final] repaired components = {repairbads_result['artifact_components']}")
#%%
# ==========================
# 13) 保存数据到 data_sim 文件夹，文件名含实际坏段比例
# ==========================
out_dir = Path(save_dir_name)
out_dir.mkdir(parents=True, exist_ok=True)

bad_ratio_pct = int(np.round(repairbads_result['detected_ratio'] * 100))
name_suffix = ""
if SIM_CONFIG_TAG:
    name_suffix += f"__cfg-{SIM_CONFIG_TAG}"
if SIM_RUN_TAG:
    name_suffix += f"__rep-{SIM_RUN_TAG}"

raw_fname = out_dir / f"raw_sim_repaired_badratio_{bad_ratio_pct:02d}pct{name_suffix}.fif"
# epochs_fname = out_dir / f"epochs_sim_repaired_badratio_{bad_ratio_pct:02d}pct{name_suffix}-epo.fif"
gt_fname = out_dir / f"ground_truth_badratio_{bad_ratio_pct:02d}pct{name_suffix}-ave.fif"
trial_meta_fname = out_dir / f"raw_sim_repaired_badratio_{bad_ratio_pct:02d}pct{name_suffix}__trial_metadata.csv"

raw_sim_selected.save(raw_fname, overwrite=True)
print(f"Saved {save_label} raw_sim to {raw_fname}")

# epochs_sim.save(epochs_fname, overwrite=True)
# print(f"Saved epochs_sim to {epochs_fname}")

metadata.to_csv(trial_meta_fname, index=False)
print(f"Saved trial metadata to {trial_meta_fname}")

info_pure = mne.pick_info(raw_sim_selected.info.copy(), np.arange(n_meg))
evoked_gt = mne.EvokedArray(signal_evoked_sensor, info_pure, tmin=tmin, nave=1, comment='Ground Truth')
evoked_gt.save(gt_fname, overwrite=True)
print(f"Saved ground truth evoked to {gt_fname}")

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
evoked_gt.plot(axes=axes[0], show=False)
axes[0].set_title("Ground Truth (Evoked Only)")

trial_idx = np.random.randint(0, len(epochs_sim))
axes[1].plot(epoch_times, X_epochs[trial_idx].T, color='gray', alpha=0.45, lw=0.6)
axes[1].set_title(f"Single Trial #{trial_idx} (Should look noisy, like empty-room)")
axes[1].set_ylabel("Tesla")

evoked_sim_avg.plot(axes=axes[2], show=False)
axes[2].set_title(f"Averaged Result (N={len(epochs_sim)})")

plt.tight_layout()
if ENABLE_PLOTS:
    plt.show()
else:
    plt.close('all')


# %%
