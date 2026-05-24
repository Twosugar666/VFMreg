#%%
import os
import glob
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne import Epochs, find_events

from robust_averaging_methods import (
    robust_abs_wacfm,
    robust_cor_wacfm,  # temporarily disabled
    robust_median_average,
    robust_tanh_mean,
    robust_trimmed_mean,

    robust_modified_dtw_average,
)
from robust_averaging_tools import (
    _load_trial_weights_csv,

    _prepare_epochs_data,
    _save_method_outputs,
    mcd_weighted_average,
)


DATA_DIR = "data_sim"
OUTPUT_DIR = "result"
WEIGHT_DIR = os.path.join(OUTPUT_DIR, "features_sim")
ROBUST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "sim_robust_average")
COMPARE_FIG_DIR = os.path.join(OUTPUT_DIR, "sim_robust_compare_figs")
METRIC_DIR = os.path.join(OUTPUT_DIR, "sim_robust_metrics")

os.makedirs(ROBUST_OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPARE_FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

REJECT_CRITERIA = None
TMIN, TMAX = -0.2, 0.8
# BASELINE = None
STIM_CHANNEL = "Trigger"
# ROBUST_BASELINE = None
ROBUST_BASELINE = (-0.2, 0.0)
POST_EVOKED_BASELINE_ENABLED = True
POST_EVOKED_BASELINE = (-0.2, 0.0)


TRIM_PROP = 0.08
TANH_K = 0.35
TANH_S = None
TANH_TRIM_COUNT = 1
WACFM_M = 2.0
WACFM_XI = 1e-10
WACFM_MAX_ITER = 200
WACFM_REJECT_C = 100.0
WACFM_METRIC_FLOOR = None
WACFM_INIT = "median"

AUDITORY_RESPONSE_WIN = (0.05, 0.25)
AUDITORY_COMPARE_WIN = (0.0, 0.25)
CORR_WINDOW = AUDITORY_RESPONSE_WIN

DTW_LPF_CUTOFF_HZ = 40.0
DTW_LPF_TRANSITION_HZ = 5.0
DTW_LPF_ATTEN_DB = 60.0
DTW_WINDOW = AUDITORY_RESPONSE_WIN
DTW_MAX_WARP_MS = 30.0

SIGNAL_WIN = AUDITORY_RESPONSE_WIN
BASELINE_WIN = (-0.2, 0.0)
COMPARE_TIME_WIN = AUDITORY_COMPARE_WIN
EVOKED_PLOT_TIME_WIN = (-0.2, 0.8)

# MCD 相关参数
MCD_REMOVE_SLOW_DRIFT = True
MCD_SLOW_DRIFT_WIN_LEN = 0.20
MCD_SLOW_DRIFT_WIN_STEP = 0.15

COMPARE_COLORS = {
    "Ground Truth": "green",
    "Conventional": "tab:blue",
    "Median": "tab:purple",
    "Trimmed": "tab:orange",
    "Tanh": "tab:brown",
    "DTW": "tab:cyan",
    "WACFM": "tab:olive",
    "corWACFM": "tab:gray",
    "MCD-weighted": "red",

}

SUMMARY_METRICS = [
    "snr_like",
    "signal_peak_ratio_to_reference",
    "signal_peak_abs_error_to_reference",
    "baseline_rms",
    "corr_to_reference",
    "rmse_to_reference",
    "peak_latency_error",
]

REPEAT_TAG_RE = re.compile(r"__rep-(?P<rep>\d+)$")
CONFIG_TAG_RE = re.compile(r"__cfg-(?P<cfg>[A-Za-z0-9_.-]+)")


def find_fif_files(data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, "raw_sim*.fif")
    files = sorted(glob.glob(pattern))
    print(f"找到 {len(files)} 个仿真 raw 文件:")
    for i, f in enumerate(files, start=1):
        print(f"{i}. {os.path.basename(f)}")
    return files

def parse_dataset_tags(dataset_name: str) -> dict:
    dataset_name = str(dataset_name)
    rep_match = REPEAT_TAG_RE.search(dataset_name)
    cfg_match = CONFIG_TAG_RE.search(dataset_name)

    repeat_index = int(rep_match.group("rep")) if rep_match else np.nan
    repeat_label = rep_match.group("rep") if rep_match else None
    config_tag = cfg_match.group("cfg") if cfg_match else None

    config_name = dataset_name
    if rep_match:
        config_name = dataset_name[:rep_match.start()]

    return {
        "dataset": dataset_name,
        "config_name": config_name,
        "config_tag": config_tag,
        "repeat_index": repeat_index,
        "repeat_label": repeat_label,
        "has_repeat_tag": bool(rep_match),
    }


def annotate_metrics_with_dataset_tags(df_metrics: pd.DataFrame) -> pd.DataFrame:
    if df_metrics.empty:
        return df_metrics.copy()

    tag_df = pd.DataFrame([parse_dataset_tags(name) for name in df_metrics["dataset"]])
    out = df_metrics.copy().reset_index(drop=True)
    for col in tag_df.columns:
        if col != "dataset":
            out[col] = tag_df[col].values
    return out


def _format_mean_pm_std(mean_value, std_value) -> str:
    if not np.isfinite(mean_value):
        return "nan"
    return f"{mean_value:.6g} +/- {std_value:.6g}"


def summarize_metrics_by_config(df_metrics: pd.DataFrame) -> pd.DataFrame:
    if df_metrics.empty:
        return pd.DataFrame()

    rows = []
    for (config_name, method), df_group in df_metrics.groupby(["config_name", "method"], dropna=False):
        row = {
            "config_name": config_name,
            "method": method,
            "n_repeats": int(df_group["dataset"].nunique()),
            "config_tag": df_group["config_tag"].dropna().iloc[0] if df_group["config_tag"].notna().any() else None,
        }
        for metric in SUMMARY_METRICS:
            values = df_group[metric].to_numpy(dtype=float)
            mean_value = float(np.nanmean(values))
            std_value = float(np.nanstd(values, ddof=1)) if np.sum(np.isfinite(values)) >= 2 else np.nan
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_median"] = float(np.nanmedian(values))
            row[f"{metric}_mean_pm_std"] = _format_mean_pm_std(mean_value, std_value)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["config_name", "method"]).reset_index(drop=True)


def load_one_dataset(
    fif_path: str,
    *,
    stim_channel: str,
    tmin: float,
    tmax: float,
    baseline,
    reject_criteria,
    post_evoked_baseline=POST_EVOKED_BASELINE,
    post_evoked_baseline_enabled=POST_EVOKED_BASELINE_ENABLED,
):
    raw0 = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    raw = raw0.copy().filter(1.0, 40, fir_design="firwin", verbose=False)
    filename = os.path.splitext(os.path.basename(fif_path))[0]
    events = find_events(raw, stim_channel=stim_channel, shortest_event=1, verbose=False)
    if len(events) == 0:
        raise RuntimeError(f"{filename} 鏈娴嬪埌浜嬩欢")
    event_id = int(events[0, 2])
    epochs = Epochs(
        raw,
        events,
        event_id={"Stim": event_id},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        detrend=None,
        reject=reject_criteria,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )
    evoked = maybe_apply_post_evoked_baseline(
        epochs.average(),
        baseline=post_evoked_baseline,
        enabled=post_evoked_baseline_enabled,
    )
    return raw, events, epochs, evoked, filename


def load_ground_truth_evoked(dataset_name: str):
    gt_name = dataset_name.replace("raw_sim_", "ground_truth_")
    gt_path = os.path.join(DATA_DIR, f"{gt_name}-ave.fif")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"鎵句笉鍒?Ground Truth 鏂囦欢: {gt_path}")
    gt_list = mne.read_evokeds(gt_path, condition=None, baseline=None, verbose=False)
    if len(gt_list) == 0:
        raise RuntimeError(f"Ground Truth 鏂囦欢涓虹┖: {gt_path}")
    return gt_list[0], gt_path


def align_evoked_to_reference(evoked, ref_evoked):
    ref = ref_evoked.copy().pick("data")
    src = evoked.copy().pick(ref.ch_names)
    if len(src.times) == len(ref.times) and np.allclose(src.times, ref.times, atol=1e-7):
        return src

    aligned_data = np.vstack([
        np.interp(ref.times, src.times, src.data[ch_idx])
        for ch_idx in range(src.data.shape[0])
    ])
    aligned = mne.EvokedArray(
        aligned_data,
        ref.info.copy(),
        tmin=ref.times[0],
        nave=src.nave,
        comment=src.comment,
        verbose=False,
    )
    return aligned


def maybe_apply_post_evoked_baseline(
    evoked,
    baseline=POST_EVOKED_BASELINE,
    enabled=POST_EVOKED_BASELINE_ENABLED,
):
    if not enabled or baseline is None:
        return evoked
    return evoked.copy().apply_baseline(baseline)


def run_robust_averaging_for_one_subject(
    epochs,
    subject_name,
    out_root,
    baseline=None,
    selected_methods=None,
    post_evoked_baseline=POST_EVOKED_BASELINE,
    post_evoked_baseline_enabled=POST_EVOKED_BASELINE_ENABLED,
):
    subject_dir = os.path.join(out_root, subject_name)
    os.makedirs(subject_dir, exist_ok=True)

    ep_used, X = _prepare_epochs_data(epochs, baseline=baseline, picks="data")
    times = ep_used.times
    results = {}
    selected_methods = None if selected_methods is None else set(selected_methods)

    if selected_methods is None or "median" in selected_methods:
        avg_median = robust_median_average(X)
        results["median"] = _save_method_outputs(
            subject_dir, subject_name, "median", avg_median, ep_used,
            extra_meta={"baseline_used_before_robust_avg": baseline},
            post_evoked_baseline=post_evoked_baseline if post_evoked_baseline_enabled else None,
        )

    if selected_methods is None or "trimmed_mean" in selected_methods:
        avg_trim = robust_trimmed_mean(X, trim_prop=TRIM_PROP)
        results["trimmed_mean"] = _save_method_outputs(
            subject_dir, subject_name, "trimmed_mean", avg_trim, ep_used,
            extra_meta={"baseline_used_before_robust_avg": baseline, "trim_prop_each_tail": TRIM_PROP},
            post_evoked_baseline=post_evoked_baseline if post_evoked_baseline_enabled else None,
        )






    if selected_methods is None or "tanh_mean" in selected_methods:
        avg_tanh, tanh_weights = robust_tanh_mean(X, k=TANH_K, s=TANH_S, trim_count=TANH_TRIM_COUNT)
        results["tanh_mean"] = _save_method_outputs(
            subject_dir, subject_name, "tanh_mean", avg_tanh, ep_used,
            extra_meta={
                "baseline_used_before_robust_avg": baseline,
                "tanh_k": TANH_K,
                "tanh_s": TANH_S,
                "tanh_trim_count": TANH_TRIM_COUNT,
                "tanh_rank_weights": tanh_weights.tolist(),
            },
            post_evoked_baseline=post_evoked_baseline if post_evoked_baseline_enabled else None,
        )

    # avg_dtw, _, dtw_meta = robust_modified_dtw_average(
    #     X,
    #     times=times,
    #     sfreq=ep_used.info["sfreq"],
    #     do_filter=False,
    #     dtw_window=DTW_WINDOW,
    #     max_warp_ms=DTW_MAX_WARP_MS,
    #     lpf_cutoff_hz=DTW_LPF_CUTOFF_HZ,
    #     lpf_transition_hz=DTW_LPF_TRANSITION_HZ,
    #     lpf_atten_db=DTW_LPF_ATTEN_DB,
    # )
    # results["dtw_average"] = _save_method_outputs(
    #     subject_dir, subject_name, "dtw_average", avg_dtw, ep_used,
    #     extra_meta={"baseline_used_before_robust_avg": baseline, **dtw_meta},
    # )

    # avg_fdtw, _, fdtw_meta = robust_modified_dtw_average(
    #     X,
    #     times=times,
    #     sfreq=ep_used.info["sfreq"],
    #     do_filter=True,
    #     dtw_window=DTW_WINDOW,
    #     max_warp_ms=DTW_MAX_WARP_MS,
    #     lpf_cutoff_hz=DTW_LPF_CUTOFF_HZ,
    #     lpf_transition_hz=DTW_LPF_TRANSITION_HZ,
    #     lpf_atten_db=DTW_LPF_ATTEN_DB,
    # )
    # results["filtered_dtw_average"] = _save_method_outputs(
    #     subject_dir, subject_name, "filtered_dtw_average", avg_fdtw, ep_used,
    #     extra_meta={"baseline_used_before_robust_avg": baseline, **fdtw_meta},
    # )

    if selected_methods is None or "wacfm" in selected_methods:
        avg_wacfm, wacfm_weights, wacfm_meta = robust_abs_wacfm(
            X,
            m=WACFM_M,
            xi=WACFM_XI,
            max_iter=WACFM_MAX_ITER,
            reject_c=WACFM_REJECT_C,
            metric_floor=WACFM_METRIC_FLOOR,
            init_mode=WACFM_INIT,
        )
        results["wacfm"] = _save_method_outputs(
            subject_dir, subject_name, "wacfm", avg_wacfm, ep_used,
            extra_meta={"baseline_used_before_robust_avg": baseline, "final_weights": wacfm_weights.tolist(), **wacfm_meta},
            post_evoked_baseline=post_evoked_baseline if post_evoked_baseline_enabled else None,
        )

    if selected_methods is None or "cor_wacfm" in selected_methods:
        avg_corwacfm, corwacfm_weights, corwacfm_meta = robust_cor_wacfm(
            X,
            times=times,
            corr_window=CORR_WINDOW,
            m=WACFM_M,
            xi=WACFM_XI,
            max_iter=WACFM_MAX_ITER,
            reject_c=WACFM_REJECT_C,
            metric_floor=WACFM_METRIC_FLOOR,
            init_mode=WACFM_INIT,
        )
        results["cor_wacfm"] = _save_method_outputs(
            subject_dir, subject_name, "cor_wacfm", avg_corwacfm, ep_used,
            extra_meta={"baseline_used_before_robust_avg": baseline, "final_weights": corwacfm_weights.tolist(), **corwacfm_meta},
            post_evoked_baseline=post_evoked_baseline if post_evoked_baseline_enabled else None,
        )

    if selected_methods is None or "mcd_weighted" in selected_methods:
        try:
            mcd_weights = _load_trial_weights_csv(WEIGHT_DIR, subject_name, n_trials=len(ep_used))
            # avg_mcd, mcd_weights_normed, mcd_neff = mcd_weighted_average(X, mcd_weights)
            avg_mcd, mcd_weights_normed, mcd_neff = mcd_weighted_average(
            X=X,          # 鍘熸湁鍙傛暟1
            w_i=mcd_weights,      # 鍘熸湁鍙傛暟2
            info=ep_used.info,  # 鏂板鍙傛暟1
            times=times, # 鏂板鍙傛暟2
            remove_slow_drift=MCD_REMOVE_SLOW_DRIFT, # 鏂板鍙傛暟3锛堝惎鐢ㄥ幓婕傦級
            win_len=MCD_SLOW_DRIFT_WIN_LEN, # 鍙€夛細璋冧紭绐楀彛闀垮害
            win_step=MCD_SLOW_DRIFT_WIN_STEP # 鍙€夛細璋冧紭绐楀彛姝ラ暱
            )


            extra_meta = {
            "baseline_used_before_robust_avg": baseline,
            "mcd_weights": mcd_weights_normed.tolist(),
            "mcd_n_eff": float(mcd_neff),
            "weight_file": os.path.join(WEIGHT_DIR, f"{subject_name}_trial_features.csv"),
            "mcd_remove_slow_drift": bool(MCD_REMOVE_SLOW_DRIFT),
            "mcd_slow_drift_win_len": float(MCD_SLOW_DRIFT_WIN_LEN),
            "mcd_slow_drift_win_step": float(MCD_SLOW_DRIFT_WIN_STEP),
            }
            results["mcd_weighted"] = _save_method_outputs(
            subject_dir, subject_name, "mcd_weighted", avg_mcd, ep_used,
            extra_meta=extra_meta,
            post_evoked_baseline=post_evoked_baseline if post_evoked_baseline_enabled else None,
        )
        except Exception as e:
            print(f"MCD-weighted failed ({subject_name}): {e}")

    return results


def _resolve_evoked_picks(evoked, picks="data"):
    if picks == "data":
        return mne.pick_types(
            evoked.info,
            meg=True,
            eeg=True,
            stim=False,
            eog=False,
            ecg=False,
            misc=False,
            exclude=[],
        )
    return np.asarray(picks, dtype=int)


def _get_evoked_scale_and_ylabel(evoked, picks_idx):
    ch_types = {evoked.get_channel_types(picks=[int(p)])[0] for p in picks_idx}
    if "mag" in ch_types:
        return 1e15, "fT"
    elif "grad" in ch_types:
        return 1e13, "fT/cm"
    elif "eeg" in ch_types:
        return 1e6, "uV"
    return 1.0, "Amplitude"


def _select_response_peak_channel(evoked, time_window=COMPARE_TIME_WIN, picks="data"):
    picks_idx = _resolve_evoked_picks(evoked, picks=picks)
    if len(picks_idx) == 0:
        raise ValueError("No data channels available for compare plot.")

    tmask = (evoked.times >= time_window[0]) & (evoked.times <= time_window[1])
    if not np.any(tmask):
        raise ValueError(f"No samples found inside compare window {time_window}.")

    window_data = evoked.data[picks_idx][:, tmask]
    peak_amplitudes = np.max(np.abs(window_data), axis=1)
    return evoked.ch_names[int(picks_idx[np.argmax(peak_amplitudes)])]


def plot_evoked_colored_on_ax(evoked, ax, title, picks="data", cmap_name="gist_ncar"):
    picks_idx = _resolve_evoked_picks(evoked, picks=picks)
    if len(picks_idx) == 0:
        ax.set_title(title)
        return

    data_plot = evoked.data[picks_idx]
    times = evoked.times
    scale, ylabel = _get_evoked_scale_and_ylabel(evoked, picks_idx)

    data_plot = data_plot * scale
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / max(data_plot.shape[0] - 1, 1)) for i in range(data_plot.shape[0])]
    for ch_idx in range(data_plot.shape[0]):
        ax.plot(times, data_plot[ch_idx], color=colors[ch_idx], linewidth=0.8)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlim(*EVOKED_PLOT_TIME_WIN)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _compute_shared_evoked_ylim(evokeds, picks="data"):
    y_min = np.inf
    y_max = -np.inf

    for evoked in evokeds:
        picks_idx = _resolve_evoked_picks(evoked, picks=picks)
        if len(picks_idx) == 0:
            continue
        scale, _ = _get_evoked_scale_and_ylabel(evoked, picks_idx)
        data_plot = evoked.data[picks_idx] * scale
        y_min = min(y_min, float(np.min(data_plot)))
        y_max = max(y_max, float(np.max(data_plot)))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None
    if np.isclose(y_min, y_max):
        pad = 0.1 * max(abs(y_min), 1.0)
    else:
        pad = 0.05 * (y_max - y_min)
    return y_min - pad, y_max + pad


# def plot_compare_on_ax(evoked_dict, reference_evoked, ax, title, time_window=COMPARE_TIME_WIN):
#     # 1. 鑾峰彇鎵€鏈夋暟鎹€氶亾鐨勭储寮曪紙鏇夸唬鍘熸潵瀵绘壘鍗曚竴璐熷嘲鍊奸€氶亾鐨勯€昏緫锛?#     picks_idx = _resolve_evoked_picks(reference_evoked, picks="data")
#     if len(picks_idx) == 0:
#         ax.set_title(title + "\n(No data channels found)")
#         return
        
#     # 2. 鑾峰彇缂╂斁姣斾緥鍜?Y 杞存爣绛?#     scale, ylabel = _get_evoked_scale_and_ylabel(reference_evoked, picks_idx)
    
#     # 3. 鏃堕棿鎺╃爜
#     tmask = (reference_evoked.times >= time_window[0]) & (reference_evoked.times <= time_window[1])
#     times_plot = reference_evoked.times[tmask]

#     y_min = np.inf
#     y_max = -np.inf
    
#     # 4. 閬嶅巻鎵€鏈夋柟娉曪紝鎻愬彇鍏ㄩ€氶亾鏁版嵁骞舵眰鍧囧€?(妯℃嫙 combine="mean")
#     for label, evoked in evoked_dict.items():
#         aligned = align_evoked_to_reference(evoked, reference_evoked)
        
#         # 鎻愬彇閫変腑鐨勬墍鏈夋暟鎹€氶亾锛?n_channels, n_times)
#         data_picked = aligned.data[picks_idx]
        
#         # 鍦ㄩ€氶亾缁村害 (axis=0) 姹傚钩鍧囷細寰楀埌 (n_times,) 鐨?1D 鏁扮粍锛屼絾mean鍙兘浼氬洜涓烘璐熸姷娑堣€岃繃灏忥紝鎵€浠ユ敼鐢ㄧ被浼?GFP 鐨勬柟寮忥細鍏堝钩鏂瑰啀骞冲潎鏈€鍚庡紑鏍瑰彿
#         # trace_mean = np.mean(data_picked, axis=0) * scale
#         trace_mean = np.sqrt(np.mean(data_picked**2, axis=0)) * scale  # GFP-like average
        
#         trace_plot = trace_mean[tmask]
        
#         ax.plot(
#             times_plot,
#             trace_plot,
#             linewidth=1.5,
#             label=label,
#             color=COMPARE_COLORS.get(label),
#         )
#         y_min = min(y_min, float(np.min(trace_plot)))
#         y_max = max(y_max, float(np.max(trace_plot)))

#     ax.axvline(0, color="k", linestyle="--", linewidth=1)
#     ax.axhline(0, color="k", linewidth=0.8, alpha=0.35)
#     ax.set_xlim(*time_window)
#     if np.isfinite(y_min) and np.isfinite(y_max):
#         if np.isclose(y_min, y_max):
#             pad = 0.1 * max(abs(y_min), 1.0)
#         else:
#             pad = 0.08 * (y_max - y_min)
#         ax.set_ylim(y_min - pad, y_max + pad)
        
#     ax.set_title(f"{title}\nCombine: Mean across {len(picks_idx)} channels", fontsize=12)
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel(ylabel)
#     ax.grid(alpha=0.25)

def plot_compare_on_ax(evoked_dict, reference_evoked, ax, title, time_window=COMPARE_TIME_WIN):
    channel_name = _select_response_peak_channel(reference_evoked, time_window=time_window, picks="data")
    compare_pick = reference_evoked.ch_names.index(channel_name)
    scale, ylabel = _get_evoked_scale_and_ylabel(reference_evoked, [compare_pick])
    tmask = (reference_evoked.times >= time_window[0]) & (reference_evoked.times <= time_window[1])
    times_plot = reference_evoked.times[tmask]

    y_min = np.inf
    y_max = -np.inf
    for label, evoked in evoked_dict.items():
        aligned = align_evoked_to_reference(evoked, reference_evoked)
        trace = aligned.data[aligned.ch_names.index(channel_name)] * scale
        trace_plot = trace[tmask]
        ax.plot(
            times_plot,
            trace_plot,
            linewidth=1.5,
            label=label,
            color=COMPARE_COLORS.get(label),
        )
        y_min = min(y_min, float(np.min(trace_plot)))
        y_max = max(y_max, float(np.max(trace_plot)))

    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.35)
    ax.set_xlim(*time_window)
    if np.isfinite(y_min) and np.isfinite(y_max):
        if np.isclose(y_min, y_max):
            pad = 0.1 * max(abs(y_min), 1.0)
        else:
            pad = 0.08 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_title(f"{title}\nMax |response| channel: {channel_name}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_one_subject_robust_comparison(subject_name, conventional_evoked, ground_truth_evoked, rr, show_figure=False):
    method_items = [("Ground Truth", ground_truth_evoked), ("Conventional average", conventional_evoked)]
    evoked_dict = {"Ground Truth": ground_truth_evoked, "Conventional": conventional_evoked}
    display_order = [
        ("median", "Median"),
        ("trimmed_mean", "Trimmed"),

        ("tanh_mean", "Tanh"),
        ("dtw_average", "DTW"),
        # ("filtered_dtw_average", "Filtered-DTW"),
        ("wacfm", "WACFM"),
        # ("cor_wacfm", "corWACFM"),
        ("mcd_weighted", "MCD-weighted"),
    ]
    for key, label in display_order:
        if key in rr:
            method_items.append((label, rr[key]["evoked"]))
            evoked_dict[label] = rr[key]["evoked"]

    n_rows = len(method_items)
    fig = plt.figure(figsize=(17, 2.2 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=n_rows, ncols=2, width_ratios=[1.15, 1.6])
    left_axes = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
    ax_right = fig.add_subplot(gs[:, 1])
    shared_ylim = _compute_shared_evoked_ylim([evk for _, evk in method_items], picks="data")

    for ax, (title, evk) in zip(left_axes, method_items):
        plot_evoked_colored_on_ax(evk, ax, title, picks="data")
        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)

    plot_compare_on_ax(evoked_dict, ground_truth_evoked, ax_right, f"Compare evokeds - {subject_name}")
    ax_right.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Robust averaging comparison - {subject_name}", fontsize=15)
    save_path = os.path.join(COMPARE_FIG_DIR, f"{subject_name}_robust_compare_all_methods.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_figure:
        plt.show()
    else:
        plt.close(fig)
    print(f"图已保存: {save_path}")


def compute_evoked_metrics(evoked, signal_win=SIGNAL_WIN, baseline_win=BASELINE_WIN):
    data = evoked.copy().pick("data").data
    times = evoked.times
    smask = (times >= signal_win[0]) & (times <= signal_win[1])
    bmask = (times >= baseline_win[0]) & (times <= baseline_win[1])

    gfp = np.sqrt(np.mean(data ** 2, axis=0))
    signal_peak = float(np.max(np.abs(gfp[smask])))
    baseline_rms = float(np.sqrt(np.mean(gfp[bmask] ** 2)))
    snr_like = signal_peak / (baseline_rms + 1e-20)
    peak_idx_local = np.argmax(np.abs(gfp[smask]))
    peak_latency = float(times[smask][peak_idx_local])
    return {
        "signal_peak": signal_peak,
        "baseline_rms": baseline_rms,
        "snr_like": snr_like,
        "peak_latency": peak_latency,
    }


def compare_to_reference(evoked, ref_evoked, signal_win=SIGNAL_WIN):
    evk = evoked.copy().pick("data")
    ref = ref_evoked.copy().pick("data")
    smask = (evk.times >= signal_win[0]) & (evk.times <= signal_win[1])
    x = evk.data[:, smask].ravel()
    y = ref.data[:, smask].ravel()
    if x.std(ddof=1) < 1e-20 or y.std(ddof=1) < 1e-20:
        corr = np.nan
    else:
        corr = float(np.corrcoef(x, y)[0, 1])
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    return {"corr_to_reference": corr, "rmse_to_reference": rmse}


def compare_scalar_metrics_to_reference(metrics: dict, ref_metrics: dict) -> dict:
    peak = float(metrics.get("signal_peak", np.nan))
    ref_peak = float(ref_metrics.get("signal_peak", np.nan))
    latency = float(metrics.get("peak_latency", np.nan))
    ref_latency = float(ref_metrics.get("peak_latency", np.nan))

    if np.isfinite(peak) and np.isfinite(ref_peak):
        peak_abs_error = abs(peak - ref_peak)
        peak_signed_error = peak - ref_peak
    else:
        peak_abs_error = np.nan
        peak_signed_error = np.nan

    if np.isfinite(peak) and np.isfinite(ref_peak) and abs(ref_peak) > 1e-30:
        peak_ratio = peak / ref_peak
        peak_relative_error = abs(peak - ref_peak) / abs(ref_peak)
    else:
        peak_ratio = np.nan
        peak_relative_error = np.nan

    if np.isfinite(latency) and np.isfinite(ref_latency):
        latency_error = abs(latency - ref_latency)
        latency_signed_error = latency - ref_latency
    else:
        latency_error = np.nan
        latency_signed_error = np.nan

    return {
        "signal_peak_ratio_to_reference": float(peak_ratio),
        "signal_peak_relative_error_to_reference": float(peak_relative_error),
        "signal_peak_abs_error_to_reference": float(peak_abs_error),
        "signal_peak_signed_error_to_reference": float(peak_signed_error),
        "peak_latency_error": float(latency_error),
        "peak_latency_signed_error": float(latency_signed_error),
    }


def plot_metric_summary(df_metrics: pd.DataFrame):
    metric_cols = [
        "snr_like",
        "signal_peak_ratio_to_reference",
        "signal_peak_abs_error_to_reference",
        "baseline_rms",
        "corr_to_reference",
        "rmse_to_reference",
        "peak_latency_error",
    ]
    for metric in metric_cols:
        plt.figure(figsize=(10, 5))
        pivot = df_metrics.pivot_table(index="dataset", columns="method", values=metric, aggfunc="first")
        pivot.plot(kind="bar", ax=plt.gca())
        plt.title(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_path = os.path.join(METRIC_DIR, f"{metric}_comparison.png")
        plt.savefig(save_path, dpi=200)
        plt.show()


def plot_metric_boxplots(
    df_metrics: pd.DataFrame,
    *,
    config_tag: str | None = None,
    mode_tag: str = "repair",
) -> list[str]:
    if df_metrics.empty:
        return []

    df_plot = annotate_metrics_with_dataset_tags(df_metrics)
    if config_tag:
        df_plot = df_plot[df_plot["config_tag"] == config_tag].copy()
    if df_plot.empty:
        return []

    metric_cols = [
        "snr_like",
        "signal_peak_ratio_to_reference",
        "signal_peak_abs_error_to_reference",
        "baseline_rms",
        "corr_to_reference",
        "rmse_to_reference",
        "peak_latency_error",
    ]
    


    method_order = [m for m in ["conventional", "median", "trimmed_mean", "tanh_mean", "wacfm", "mcd_weighted", "ground_truth"] if m in df_plot["method"].unique()]
    if not method_order:
        method_order = sorted(df_plot["method"].dropna().unique())

    saved_paths = []
    title_cfg = config_tag if config_tag else "all"
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        data_by_method = []
        labels = []
        for method in method_order:
            vals = df_plot.loc[df_plot["method"] == method, metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            data_by_method.append(vals)
            labels.append(method)

        if not data_by_method:
            plt.close(fig)
            continue

        bp = ax.boxplot(data_by_method, labels=labels, patch_artist=True, showfliers=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        y_min, y_max = ax.get_ylim()
        y_span = max(y_max - y_min, 1e-12)
        for idx, vals in enumerate(data_by_method, start=1):
            mean_value = float(np.nanmean(vals))
            std_value = float(np.nanstd(vals, ddof=1)) if len(vals) >= 2 else np.nan
            median_value = float(np.nanmedian(vals))
            text = f"{_format_mean_pm_std(mean_value, std_value)}\nmed={median_value:.6g}"
            ax.text(
                idx,
                np.nanmax(vals) + 0.04 * y_span,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

        ax.set_title(f"{metric} boxplot | cfg={title_cfg} | {mode_tag}")
        ax.set_xlabel("Method")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        suffix = f"__cfg-{title_cfg}__{mode_tag}" if title_cfg else f"__{mode_tag}"
        save_path = os.path.join(METRIC_DIR, f"{metric}_boxplot{suffix}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        saved_paths.append(save_path)
        plt.close(fig)

    return saved_paths


def load_ground_truth_evoked(dataset_name: str):
    candidate_names = []
    if dataset_name.startswith("raw_sim_repaired_"):
        candidate_names.append(dataset_name.replace("raw_sim_repaired_", "ground_truth_repaired_"))
        candidate_names.append(dataset_name.replace("raw_sim_repaired_", "ground_truth_"))
    elif dataset_name.startswith("raw_sim_"):
        candidate_names.append(dataset_name.replace("raw_sim_", "ground_truth_"))
    else:
        candidate_names.append(f"ground_truth_{dataset_name}")

    gt_path = None
    for gt_name in candidate_names:
        candidate_path = os.path.join(DATA_DIR, f"{gt_name}-ave.fif")
        if os.path.exists(candidate_path):
            gt_path = candidate_path
            break

    if gt_path is None:
        searched = ", ".join(os.path.join(DATA_DIR, f"{name}-ave.fif") for name in candidate_names)
        raise FileNotFoundError(f"鎵句笉鍒?Ground Truth 鏂囦欢锛屽凡妫€鏌? {searched}")

    gt_list = mne.read_evokeds(gt_path, condition=None, baseline=None, verbose=False)
    if len(gt_list) == 0:
        raise RuntimeError(f"Ground Truth 鏂囦欢涓虹┖: {gt_path}")
    return gt_list[0], gt_path


def prepare_sim_datasets(
    name_contains: str | None = None,
    name_exact: str | None = None,
    *,
    post_evoked_baseline=POST_EVOKED_BASELINE,
    post_evoked_baseline_enabled=POST_EVOKED_BASELINE_ENABLED,
):
    fif_files = find_fif_files(DATA_DIR)
    if name_exact:
        fif_files = [f for f in fif_files if Path(f).stem == name_exact]
    elif name_contains:
        fif_files = [f for f in fif_files if name_contains in Path(f).stem]
    if not fif_files:
        raise FileNotFoundError(f"{DATA_DIR} 涓嬫湭鎵惧埌 raw_sim*.fif")

    datasets = []
    for fif_path in fif_files:
        raw, events, epochs, evoked, filename = load_one_dataset(
            fif_path,
            stim_channel=STIM_CHANNEL,
            tmin=TMIN,
            tmax=TMAX,
            baseline=ROBUST_BASELINE,
            reject_criteria=REJECT_CRITERIA,
            post_evoked_baseline=post_evoked_baseline,
            post_evoked_baseline_enabled=post_evoked_baseline_enabled,
        )
        ground_truth_evoked, gt_path = load_ground_truth_evoked(filename)
        ground_truth_evoked = align_evoked_to_reference(ground_truth_evoked, evoked)
        ground_truth_evoked = maybe_apply_post_evoked_baseline(
            ground_truth_evoked,
            baseline=post_evoked_baseline,
            enabled=post_evoked_baseline_enabled,
        )
        print(f"\n已加载数据集: {filename}, n_epochs={len(epochs)}")
        print(f"Ground Truth: {gt_path}")
        datasets.append({
            "filename": filename,
            "raw": raw,
            "events": events,
            "epochs": epochs,
            "evoked": evoked,
            "ground_truth_evoked": ground_truth_evoked,
            "ground_truth_path": gt_path,
        })
    return datasets


def run_all_robust_methods(
    datasets,
    baseline=ROBUST_BASELINE,
    *,
    selected_methods=None,
    post_evoked_baseline=POST_EVOKED_BASELINE,
    post_evoked_baseline_enabled=POST_EVOKED_BASELINE_ENABLED,
):
    robust_results_all = {}
    for ds in datasets:
        filename = ds["filename"]
        rr = run_robust_averaging_for_one_subject(
            epochs=ds["epochs"],
            subject_name=filename,
            out_root=ROBUST_OUTPUT_DIR,
            baseline=baseline,
            selected_methods=selected_methods,
            post_evoked_baseline=post_evoked_baseline,
            post_evoked_baseline_enabled=post_evoked_baseline_enabled,
        )
        robust_results_all[filename] = rr
        print(f"已完成稳健平均: {filename}")
    return robust_results_all


def plot_all_subject_comparisons(datasets, robust_results_all, show_figures=False):
    for ds in datasets:
        filename = ds["filename"]
        plot_one_subject_robust_comparison(
            filename,
            ds["evoked"],
            ds["ground_truth_evoked"],
            robust_results_all[filename],
            show_figure=show_figures,
        )


def build_all_metrics(datasets, robust_results_all):
    all_metrics = []
    for ds in datasets:
        filename = ds["filename"]
        evoked = ds["evoked"]
        ground_truth_evoked = ds["ground_truth_evoked"]
        rr = robust_results_all[filename]
        gt_metrics = compute_evoked_metrics(ground_truth_evoked)
        gt_scalar_metrics = gt_metrics.copy()

        conventional_metrics = compute_evoked_metrics(evoked)
        conventional_metrics.update({
            "dataset": filename,
            "method": "conventional",
            **compare_to_reference(evoked, ground_truth_evoked),
            **compare_scalar_metrics_to_reference(conventional_metrics, gt_scalar_metrics),
        })
        all_metrics.append(conventional_metrics)

        gt_metrics.update({
            "dataset": filename,
            "method": "ground_truth",
            "corr_to_reference": 1.0,
            "rmse_to_reference": 0.0,
            "signal_peak_ratio_to_reference": 1.0,
            "signal_peak_relative_error_to_reference": 0.0,
            "signal_peak_abs_error_to_reference": 0.0,
            "signal_peak_signed_error_to_reference": 0.0,
            "peak_latency_error": 0.0,
            "peak_latency_signed_error": 0.0,
        })
        all_metrics.append(gt_metrics)

        for method_name, method_result in rr.items():
            evk = method_result["evoked"]
            metrics = compute_evoked_metrics(evk)
            metrics.update(compare_to_reference(evk, ground_truth_evoked))
            metrics.update(compare_scalar_metrics_to_reference(metrics, gt_scalar_metrics))
            metrics.update({"dataset": filename, "method": method_name})
            all_metrics.append(metrics)
            # print(metrics)

    return pd.DataFrame(all_metrics)


def save_metric_tables(df_metrics: pd.DataFrame):
    df_metrics = annotate_metrics_with_dataset_tags(df_metrics)
    metric_csv = os.path.join(METRIC_DIR, "sim_robust_metrics.csv")
    df_metrics.to_csv(metric_csv, index=False)
    print(df_metrics)
    print(f"指标表已保存: {metric_csv}")

    summary_csv = os.path.join(METRIC_DIR, "sim_robust_metrics_summary.csv")
    df_summary = summarize_metrics_by_config(df_metrics)
    if df_summary.empty:
        df_summary = df_metrics.groupby("method", as_index=False).mean(numeric_only=True)
    df_summary.to_csv(summary_csv, index=False)
    print(f"指标汇总表已保存: {summary_csv}")
    return df_summary


def run_pipeline(
    plot_subjects=True,
    plot_metrics=True,
    baseline=ROBUST_BASELINE,
    name_contains: str | None = None,
    *,
    post_evoked_baseline=POST_EVOKED_BASELINE,
    post_evoked_baseline_enabled=POST_EVOKED_BASELINE_ENABLED,
):
    datasets = prepare_sim_datasets(
        name_contains=name_contains,
        post_evoked_baseline=post_evoked_baseline,
        post_evoked_baseline_enabled=post_evoked_baseline_enabled,
    )
    robust_results_all = run_all_robust_methods(
        datasets,
        baseline=baseline,
        post_evoked_baseline=post_evoked_baseline,
        post_evoked_baseline_enabled=post_evoked_baseline_enabled,
    )

    if plot_subjects:
        plot_all_subject_comparisons(datasets, robust_results_all)

    df_metrics = build_all_metrics(datasets, robust_results_all)
    df_summary = save_metric_tables(df_metrics)

    if plot_metrics:
        plot_metric_summary(df_metrics)

    print(f"稳健平均结果输出目录: {ROBUST_OUTPUT_DIR}")
    return datasets, robust_results_all, df_metrics, df_summary


# def main():
#     return run_pipeline(plot_subjects=True, plot_metrics=True, baseline=ROBUST_BASELINE)


# 浜や簰寮忔帹鑽愯皟鐢ㄩ『搴忥細
# datasets = prepare_sim_datasets()
# robust_results_all = run_all_robust_methods(datasets)
# plot_one_subject_robust_comparison(
#     datasets[0]["filename"],
#     datasets[0]["evoked"],
#     datasets[0]["ground_truth_evoked"],
#     robust_results_all[datasets[0]["filename"]],
# )
# df_metrics = build_all_metrics(datasets, robust_results_all)
# df_summary = save_metric_tables(df_metrics)
# plot_metric_summary(df_metrics)
# if __name__ == "__main__":
#     main()




# # 浜や簰寮忔帹鑽愯皟鐢ㄩ『搴忥細
# datasets = prepare_sim_datasets()
# robust_results_all = run_all_robust_methods(datasets)
# plot_one_subject_robust_comparison(
#     datasets[0]["filename"],
#     datasets[0]["evoked"],
#     datasets[0]["ground_truth_evoked"],
#     robust_results_all[datasets[0]["filename"]],
# )
# #%%
# df_metrics = build_all_metrics(datasets, robust_results_all)
# df_summary = save_metric_tables(df_metrics)
# plot_metric_summary(df_metrics)

# %%
