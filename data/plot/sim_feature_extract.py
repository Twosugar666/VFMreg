#%%
import os
import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, gmean, gstd, kurtosis
from scipy.ndimage import binary_dilation
from sklearn.covariance import MinCovDet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.tangentspace import TangentSpace
from mne.time_frequency import psd_array_multitaper


DATA_DIR = "data_sim"
OUTPUT_DIR = "result"
FEATURE_DIR = os.path.join(OUTPUT_DIR, "features_sim")
FIG_DIR = os.path.join(OUTPUT_DIR, "feature_figs_sim")

TMIN = -0.2
TMAX = 0.8
BASELINE = None
PICKS = "mag"
STIM_CHANNEL = "Trigger"
LOW_Q = 0.2
HIGH_Q = 0.8
SNR_BASELINE = (-0.2, 0.0)
AUDITORY_ANALYSIS_WIN = (0.08, 0.25)
AUDITORY_TEMPLATE_WIN = (0.08, 0.25)
# SNR_SIGNAL = (0.0, 0.5)#原来是0.5
SNR_SIGNAL = AUDITORY_ANALYSIS_WIN
NEIGH_K = 5
EPS = 1e-12
MCD_SUPPORT_FRACTION = 0.75
MCD_CHI2_P = 0.95

MCD_TEMPLATE_PRUNE_REDUCE = "mean_abs"
MCD_TEMPLATE_PRUNE_RESIDUAL_WIN = AUDITORY_ANALYSIS_WIN
MCD_TEMPLATE_PRUNE_QUANTILE = 0.70
MCD_WEIGHT_TIME_CONSISTENCY = 0.3
MCD_WEIGHT_TOPO_CONSISTENCY = 0.7
MCD_USE_Q_SCORE = True
MCD_USE_RESIDUAL_SCORE = True
MCD_SCORE_MODE = "product"
MCD_SCORE_EPS = 1e-6
MCD_SOFTMAX_RHO = 0.95 #0.96

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def find_sim_raw_files(data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, "raw_sim*.fif")
    files = sorted(glob.glob(pattern))
    print(f"找到 {len(files)} 个仿真 raw 文件")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    return files


def load_raw_and_epochs(fif_path: str):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    events = mne.find_events(raw, stim_channel=STIM_CHANNEL, shortest_event=1, verbose=False)
    if len(events) == 0:
        raise RuntimeError(f"{fif_path} 未检测到事件")
    event_code = int(events[0, 2])
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"Stim": event_code},
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )
    name = Path(fif_path).stem
    return raw, events, epochs, name


def _safe_log1p(x, eps=EPS):
    x = np.asarray(x, float)
    mn = np.nanmin(x)
    if np.isfinite(mn) and mn < 0:
        x = x - mn
    return np.log1p(np.maximum(x, 0.0) + eps)


def build_penalty_features(df: pd.DataFrame, cfg: dict, do_log: bool = True):
    df_pen = df.copy()
    pen_cols = []
    for feat, meta in cfg.items():
        if feat not in df_pen.columns:
            continue
        x = df_pen[feat].to_numpy(dtype=float)
        polarity = meta.get("polarity", "lower_better")
        transform = meta.get("transform", "identity")
        use_log = meta.get("log", False) and do_log

        if polarity == "higher_better":
            if transform == "1-r":
                x = np.clip(x, -1.0, 1.0)
                pen = 1.0 - x
            elif transform == "inv":
                pen = 1.0 / (x + EPS)
            else:
                pen = -x
        else:
            if transform == "abs_dev":
                pen = np.abs(x - meta.get("center", 0.5))
            else:
                pen = x

        if use_log:
            pen = _safe_log1p(pen)

        col = f"Pen_{feat}"
        df_pen[col] = pen
        pen_cols.append(col)

    X = df_pen[pen_cols].to_numpy(dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        good = np.isfinite(col)
        if np.any(good):
            col[~good] = np.nanmedian(col[good])
        else:
            col[:] = 0.0
        X[:, j] = col
    return X, pen_cols, df_pen


def mcd_quality_score(X: np.ndarray, support_fraction: float = 0.75, random_state: int = 42):
    mcd = MinCovDet(support_fraction=support_fraction, random_state=random_state)
    mcd.fit(X)
    q = mcd.mahalanobis(X)
    return q, mcd


def chi2_threshold(k: int, p: float = 0.95) -> float:
    return float(chi2.ppf(p, df=k))


def robust_minmax_rank(x):
    x = np.asarray(x, float)
    order = np.argsort(np.argsort(x))
    if len(x) <= 1:
        return np.ones_like(x, dtype=float)
    return order / (len(x) - 1)


def build_template_from_support(epoch_data, q_i, support_mask, eps=1e-12):
    q_i = np.asarray(q_i, float)
    support_mask = np.asarray(support_mask, bool)
    a = np.zeros_like(q_i, dtype=float)
    a[support_mask] = 1.0 / (q_i[support_mask] + eps)
    s = a.sum()
    if s <= 0:
        a = support_mask.astype(float)
        s = a.sum()
    if s <= 0:
        a = np.ones_like(q_i, dtype=float)
        s = a.sum()
    a_norm = a / s
    template = np.tensordot(a_norm, epoch_data, axes=(0, 0))
    return template, a_norm


def compute_time_consistency(epoch_data, template, times, signal_win=AUDITORY_ANALYSIS_WIN):
    smask = (times >= signal_win[0]) & (times <= signal_win[1])
    tpl_1d = template[:, smask].mean(axis=0)
    dt = float(np.median(np.diff(times)))
    max_lag = max(0, int(round((10.0 / 1000.0) / dt)))
    out = np.zeros(epoch_data.shape[0], dtype=float)

    for i in range(epoch_data.shape[0]):
        x_1d = epoch_data[i][:, smask].mean(axis=0)
        best_r = np.nan
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                a, b = x_1d[lag:], tpl_1d[:-lag]
            elif lag < 0:
                a, b = x_1d[:lag], tpl_1d[-lag:]
            else:
                a, b = x_1d, tpl_1d
            if len(a) < 3 or len(b) < 3:
                continue
            sa = a.std(ddof=1)
            sb = b.std(ddof=1)
            if sa < 1e-20 or sb < 1e-20:
                continue
            r = np.corrcoef(a, b)[0, 1]
            rmse = np.sqrt(np.mean((a - b)**2))
            if np.isfinite(r) and ((not np.isfinite(best_r)) or (r > best_r)):
                best_r = r
        out[i] = best_r if np.isfinite(best_r) else 0.0
    return out


def compute_topo_consistency(epoch_data, template, times, signal_win=AUDITORY_ANALYSIS_WIN):
    smask = (times >= signal_win[0]) & (times <= signal_win[1])
    tpl_topo = template[:, smask].mean(axis=1)
    tpl_std = tpl_topo.std(ddof=1)
    out = np.zeros(epoch_data.shape[0], dtype=float)
    for i in range(epoch_data.shape[0]):
        x_topo = epoch_data[i][:, smask].mean(axis=1)
        x_std = x_topo.std(ddof=1)
        if tpl_std < 1e-20 or x_std < 1e-20:
            out[i] = 0.0
        else:
            r = np.corrcoef(x_topo, tpl_topo)[0, 1]
            out[i] = r if np.isfinite(r) else 0.0
    return out


def softmax_with_temperature(x, tau):
    x = np.asarray(x, float)
    z = (x - np.max(x)) / max(tau, 1e-12)
    ez = np.exp(z)
    s = ez.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(x) / len(x)
    return ez / s


def effective_trial_count(w):
    denom = np.sum(np.asarray(w, float) ** 2)
    return 0.0 if denom <= 0 else 1.0 / denom


def auto_temperature_by_neff(score, rho=0.75):
    tau_grid = np.logspace(-2, 2, 200)
    target_neff = rho * len(score)
    best_tau = tau_grid[-1]
    best_w = softmax_with_temperature(score, best_tau)
    for tau in tau_grid:
        w = softmax_with_temperature(score, tau)
        if effective_trial_count(w) >= target_neff:
            best_tau = tau
            best_w = w
            break
    return best_w, best_tau, effective_trial_count(best_w)


def combine_quality_components(
    components: dict[str, np.ndarray],
    *,
    mode: str = "product",
    eps: float = 1e-6,
) -> np.ndarray:
    active = [np.asarray(v, float) for v in components.values() if v is not None]
    if not active:
        raise ValueError("At least one quality component must be provided")

    if mode == "product":
        out = np.ones_like(active[0], dtype=float)
        for v in active:
            out *= np.clip(v, eps, None)
        return out

    if mode == "geometric_mean":
        out = np.ones_like(active[0], dtype=float)
        for v in active:
            out *= np.clip(v, eps, None)
        return out ** (1.0 / len(active))

    if mode == "mean":
        return np.mean(np.vstack(active), axis=0)

    raise ValueError(f"Unsupported MCD_SCORE_MODE: {mode}")


def compute_template_residual(epoch_data, template, times=None, residual_win=None, reduce="mean_abs"):
    """
    计算每个 trial 相对模板的残差，值越大表示越偏离模板。
    """
    X = np.asarray(epoch_data, float)
    T = np.asarray(template, float)

    if times is None or residual_win is None:
        mask = np.ones(T.shape[1], dtype=bool)
    else:
        mask = (times >= residual_win[0]) & (times <= residual_win[1])

    diff = X[:, :, mask] - T[None, :, mask]

    if reduce == "mean_abs":
        residual = np.mean(np.abs(diff), axis=(1, 2))
    elif reduce == "rmse":
        residual = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    else:
        raise ValueError("reduce must be 'mean_abs' or 'rmse'")

    return residual




def build_template_from_weights(epoch_data, w_i, eps=1e-12):
    """
    直接按给定权重构建模板。
    """
    w = np.asarray(w_i, float).copy()
    w = np.where(np.isfinite(w), w, 0.0)

    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / len(w)
    else:
        w /= s

    template = np.tensordot(w, epoch_data, axes=(0, 0))
    return template


def _get_ch_xyz(info, picks):
    pick_inds = mne.pick_types(info, meg=picks, eeg=False, stim=False, eog=False, ecg=False, misc=False)
    if len(pick_inds) == 0:
        raise RuntimeError("没有选到任何 MEG(mag) 通道")
    xyz = np.zeros((len(pick_inds), 3), dtype=float)
    for i, ch_idx in enumerate(pick_inds):
        xyz[i] = info["chs"][ch_idx]["loc"][:3]
    if np.allclose(xyz, 0):
        raise RuntimeError("通道 loc[:3] 全为 0，无法计算邻域相关")
    return xyz, pick_inds


def _nearest_neighbors(xyz, k=5):
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))
    neigh = []
    for i in range(xyz.shape[0]):
        order = np.argsort(d[i])
        neigh.append(order[order != i][:k])
    return neigh


def compute_trial_snr_robust(epoch_data, sfreq, times, baseline_win, signal_win, noise_band=(20, 40)):
    bmask = (times >= baseline_win[0]) & (times <= baseline_win[1])
    smask = (times >= signal_win[0]) & (times <= signal_win[1])
    baseline_seg = epoch_data[:, :, bmask]
    signal_seg = epoch_data[:, :, smask]
    psds, _ = psd_array_multitaper(baseline_seg, sfreq=sfreq, fmin=noise_band[0], fmax=noise_band[1], verbose=False)
    noise_rms = np.sqrt(psds.mean(axis=2))
    trial_noise_level = noise_rms.mean(axis=1)
    peak = np.max(np.abs(signal_seg), axis=(1, 2))
    return peak / (trial_noise_level + 1e-20)


def snr_grouping(snr, low_q=0.2, high_q=0.8):
    lo = np.quantile(snr, low_q)
    hi = np.quantile(snr, high_q)
    grp = np.full_like(snr, fill_value="Medium", dtype=object)
    grp[snr <= lo] = "Low"
    grp[snr >= hi] = "High"
    return grp, lo, hi


def compute_stat_features(epoch_data):
    kurt_ch = kurtosis(epoch_data, axis=2, fisher=False, bias=False)
    return np.nanmean(kurt_ch, axis=1)


def compute_neighbor_corr(epoch_data, neigh_idx):
    n_trials, n_ch, _ = epoch_data.shape
    out = np.zeros(n_trials, float)
    for ti in range(n_trials):
        X = epoch_data[ti]
        ch_corr_means = np.zeros(n_ch, float)
        for ci in range(n_ch):
            x = X[ci]
            corrs = []
            for nj in neigh_idx[ci]:
                y = X[nj]
                sx = x.std(ddof=1)
                sy = y.std(ddof=1)
                if sx < 1e-20 or sy < 1e-20:
                    continue
                r = np.corrcoef(x, y)[0, 1]
                if np.isfinite(r):
                    corrs.append(r)
            ch_corr_means[ci] = np.median(corrs) if corrs else np.nan
        out[ti] = np.nanmean(ch_corr_means)
    return out


def compute_baseline_bad_channel_burden(epoch_data, times, baseline_win=(-0.2, 0.0), q=0.9):
    bmask = (times >= baseline_win[0]) & (times <= baseline_win[1])
    Xb = epoch_data[:, :, bmask]
    a_ic = Xb.max(axis=2) - Xb.min(axis=2)
    med_c = np.median(a_ic, axis=0)
    mad_c = np.median(np.abs(a_ic - med_c[None, :]), axis=0)
    denom = 1.4826 * mad_c + EPS
    z_ic = (a_ic - med_c[None, :]) / denom[None, :]
    burden = np.quantile(np.maximum(z_ic, 0.0), q=q, axis=1)
    return burden


def _force_spd(C, eig_floor_rel=1e-6):
    C = np.asarray(C, dtype=float)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = 0.5 * (C + C.T)

    eigvals, eigvecs = np.linalg.eigh(C)
    finite_pos = eigvals[np.isfinite(eigvals) & (eigvals > 0)]
    if finite_pos.size:
        scale = float(np.median(finite_pos))
    else:
        scale = 1.0
    floor = max(EPS, eig_floor_rel * scale)
    eigvals = np.clip(np.nan_to_num(eigvals, nan=floor, posinf=floor, neginf=floor), floor, None)
    C_spd = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (C_spd + C_spd.T)


def _spd_cov(trial_X):
    X = np.asarray(trial_X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.cov(X)
    return _force_spd(C)


def compute_riemann_distance(epoch_data, variance_threshold=0.95, max_components=8):
    covs = np.stack([_spd_cov(epoch_data[i]) for i in range(epoch_data.shape[0])])
    try:
        ts = TangentSpace(metric='riemann')
        tangent_vectors = ts.fit_transform(covs)
    except ValueError:
        covs = np.stack([_force_spd(C, eig_floor_rel=1e-4) for C in covs])
        ts = TangentSpace(metric='logeuclid')
        tangent_vectors = ts.fit_transform(covs)
    dists = np.linalg.norm(tangent_vectors, axis=1)
    n_pca_max = max(1, min(tangent_vectors.shape[0], tangent_vectors.shape[1], max_components))
    temp_vecs = PCA(n_components=variance_threshold).fit_transform(tangent_vectors)
    if temp_vecs.shape[1] > n_pca_max:
        tangent_pca = PCA(n_components=n_pca_max).fit_transform(tangent_vectors)
    else:
        tangent_pca = temp_vecs
    return dists, tangent_pca

def compute_template_residual(epoch_data, template, times=None, residual_win=None, reduce="mean_abs"):
    """
    epoch_data: (n_trials, n_channels, n_times)
    template:   (n_channels, n_times)

    返回每个 trial 相对模板的残差，值越大表示越偏离模板。
    """
    X = np.asarray(epoch_data, float)
    T = np.asarray(template, float)

    if times is None or residual_win is None:
        mask = np.ones(T.shape[1], dtype=bool)
    else:
        mask = (times >= residual_win[0]) & (times <= residual_win[1])

    diff = X[:, :, mask] - T[None, :, mask]

    if reduce == "mean_abs":
        # 推荐：对长段慢漂比较稳
        residual = np.mean(np.abs(diff), axis=(1, 2))
    elif reduce == "rmse":
        residual = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    else:
        raise ValueError("reduce must be 'mean_abs' or 'rmse'")

    return residual

def build_template_with_residual_pruning(
    epoch_data,
    q_i,
    support_mask,
    times=None,
    residual_win=None,
    reduce="mean_abs",
    prune_quantile=0.75,
    eps=1e-12,
):
    """
    两步法建模板：
    1) 先用原 support_mask 建一个初始模板
    2) 计算每个 trial 到初始模板的残差
    3) 在 support 内剔除残差较大的 trial
    4) 用剩余 trial 重建模板
    """
    # 初始模板
    template0, a0 = build_template_from_support(epoch_data, q_i, support_mask, eps=eps)

    # 计算 support 内 trial 的模板残差
    residual = compute_template_residual(
        epoch_data,
        template0,
        times=times,
        residual_win=residual_win,
        reduce=reduce,
    )

    support_idx = np.where(support_mask)[0]
    if len(support_idx) == 0:
        return template0, support_mask.copy(), residual

    support_res = residual[support_idx]
    thr = np.quantile(support_res, prune_quantile)

    refined_support_mask = support_mask.copy()
    refined_support_mask[support_idx[support_res > thr]] = False

    # 防止删太狠
    if refined_support_mask.sum() < max(5, int(0.2 * len(support_mask))):
        refined_support_mask = support_mask.copy()
        print("剪枝保护机制触发，保留原 support_mask")

    template1, a1 = build_template_from_support(
        epoch_data, q_i, refined_support_mask, eps=eps
    )

    return template1, refined_support_mask, residual

def plot_q_hist(df_features: pd.DataFrame, chi_thr: float, k: int, name: str):
    plt.figure()
    plt.hist(df_features["q_i"].values, bins=40)
    plt.axvline(chi_thr, linestyle="--")
    plt.xlabel("q_i (squared Mahalanobis)")
    plt.ylabel("Count")
    plt.title(f"{name}: q_i distribution (Chi2 p95 thr={chi_thr:.2f}, K={k})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, f"{name}_q_mcd_hist.png")
    plt.savefig(save_path, dpi=200)
    plt.close()






def process_one_dataset(fif_path: str) -> pd.DataFrame:
    raw, events, epochs, name = load_raw_and_epochs(fif_path)
    print(f"\n========== Dataset: {name} ==========")

    if "Trigger" in epochs.ch_names:
        epochs = epochs.copy().drop_channels(["Trigger"])

    data = epochs.get_data(picks=PICKS)
    times = epochs.times

    sfreq = epochs.info["sfreq"]
    n_trials = data.shape[0]

    snr = compute_trial_snr_robust(data, sfreq, times, SNR_BASELINE, SNR_SIGNAL)
    snr_group, _, _ = snr_grouping(snr, low_q=LOW_Q, high_q=HIGH_Q)
    kurt_mean = compute_stat_features(data)
    baseline_bad_burden = compute_baseline_bad_channel_burden(data, times, baseline_win=SNR_BASELINE, q=0.9)
    xyz, _ = _get_ch_xyz(epochs.info, picks=PICKS)
    neigh_idx = _nearest_neighbors(xyz, k=NEIGH_K)
    neigh_corr = compute_neighbor_corr(data, neigh_idx)
    riem_dist, riem_pca = compute_riemann_distance(data, variance_threshold=0.95, max_components=8)

    df_features = pd.DataFrame({
        "Dataset": name,
        "Trial": np.arange(n_trials),
        "SNR": snr,
        "SNR_Group": snr_group,
        "Kurtosis": kurt_mean,
        "NeighborCorr": neigh_corr,
        "RiemannDist": riem_dist,
        "BaselineBadBurden": baseline_bad_burden,
    })

    penalty_cfg = {
        "NeighborCorr": {"polarity": "higher_better", "transform": "1-r", "log": False},
        "Kurtosis": {"polarity": "lower_better", "transform": "identity", "log": True},
        "RiemannDist": {"polarity": "lower_better", "transform": "identity", "log": True},
        "BaselineBadBurden": {"polarity": "lower_better", "transform": "identity", "log": True},
    }
    X_pen, _, df_features = build_penalty_features(df_features, cfg=penalty_cfg, do_log=True)
    k = X_pen.shape[1]
    q_i, mcd_model = mcd_quality_score(
        X_pen,
        support_fraction=MCD_SUPPORT_FRACTION,
        random_state=42,
    )
    chi_thr = chi2_threshold(k, p=MCD_CHI2_P)
    is_bad = q_i > chi_thr

    df_features["q_i"] = q_i
    df_features["q_mcd_sqrt"] = np.sqrt(q_i)
    df_features["chi2_thr_p95"] = chi_thr
    df_features["Bad_by_chi2_p05"] = is_bad.astype(int)
    df_features["MCD_Support"] = mcd_model.support_.astype(int)

    support_mask = df_features["MCD_Support"].to_numpy(dtype=int).astype(bool)
    # template, template_w = build_template_from_support(data, q_i, support_mask=support_mask)
    template, refined_support_mask, template_residual0 = build_template_with_residual_pruning(
    data,
    q_i,
    support_mask=support_mask,
    times=times,
    residual_win=SNR_SIGNAL,
    reduce="mean_abs",
    prune_quantile=0.75,    # 先试 0.75，再试 0.70 / 0.80
    )


    c_time = compute_time_consistency(data, template, times, signal_win=SNR_SIGNAL)
    c_topo = compute_topo_consistency(data, template, times, signal_win=SNR_SIGNAL)


    template_residual = compute_template_residual(
    data,
    template,
    times=times,
    residual_win=SNR_SIGNAL,
    reduce="mean_abs",
    )



    residual_conf = 1.0 / (template_residual + 1e-12)
    consistency_mean = (
        MCD_WEIGHT_TIME_CONSISTENCY * c_time
        + MCD_WEIGHT_TOPO_CONSISTENCY * c_topo
    )

    b_conf = 1.0 / (baseline_bad_burden + 1e-12)
    q_conf = 1.0 / (q_i + 1e-12)

    score_cons = robust_minmax_rank(consistency_mean)
    score_q = robust_minmax_rank(q_conf)
    score_b = robust_minmax_rank(b_conf)
    score_r = robust_minmax_rank(residual_conf)
    quality_score = score_r * score_b * score_q * score_cons

    w_i, tau_auto, neff = auto_temperature_by_neff(quality_score, rho=MCD_SOFTMAX_RHO)
    # #这里是0.9好



    df_features["TemplateSet"] = support_mask.astype(int)
    df_features["RefinedTemplateSet"] = refined_support_mask.astype(int)
    df_features["TemplateResidual0"] = template_residual0
    # df_features["TemplateWeight_q"] = template_w
    df_features["Consistency_Time"] = c_time
    df_features["Consistency_Topo"] = c_topo
    df_features["Consistency_Mean"] = consistency_mean
    df_features["QualityScore"] = quality_score
    df_features["tau_auto"] = tau_auto
    df_features["N_eff"] = neff
    df_features["w_i"] = w_i
    df_features["MCDSupportFraction"] = MCD_SUPPORT_FRACTION
    df_features["MCDChi2P"] = MCD_CHI2_P
    # df_features["MCDPruneQuantile"] = MCD_TEMPLATE_PRUNE_QUANTILE
    df_features["MCDSoftmaxRho"] = MCD_SOFTMAX_RHO


 

    for i in range(riem_pca.shape[1]):
        df_features[f"Feat_GeoPC_{i}"] = riem_pca[:, i]

    csv_path = os.path.join(FEATURE_DIR, f"{name}_trial_features.csv")
    df_features.to_csv(csv_path, index=False)
    print(f"Saved feature CSV: {csv_path}")
    plot_q_hist(df_features, chi_thr, k, name)
    return df_features

# #%%交互式调用
# fif_files = find_sim_raw_files(DATA_DIR)
# if not fif_files:
#     raise FileNotFoundError(f"{DATA_DIR} 下未找到 raw_sim*.fif")

# all_dfs = []
# for fif_path in fif_files:
#     df_features = process_one_dataset(fif_path)
#     all_dfs.append(df_features)

# df_all = pd.concat(all_dfs, ignore_index=True)
# merged_path = os.path.join(FEATURE_DIR, "ALL_sim_trial_features.csv")
# df_all.to_csv(merged_path, index=False)
# print(f"Saved merged feature CSV: {merged_path}")




# %%
