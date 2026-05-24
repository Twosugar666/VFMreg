import json
import os

import mne
import numpy as np
import pandas as pd


def _prepare_epochs_data(epochs, baseline=None, picks="data"):
    """
    从 Epochs 中取出用于 robust averaging 的数据
    X shape = (n_epochs, n_channels, n_times)

    返回:
      ep_picked: 已经 pick 后、和 X 完全对应的 Epochs
      X:         (n_epochs, n_channels, n_times)
    """
    ep = epochs.copy()
    if baseline is not None:
        ep = ep.apply_baseline(baseline)

    ep_picked = ep.copy().pick(picks)
    X = ep_picked.get_data(copy=True)
    return ep_picked, X


def _make_evoked_from_data(avg_data, epochs_ref, comment="", nave=None):
    """
    根据平均后的二维数据 (n_channels, n_times) 构造 EvokedArray
    """
    info = epochs_ref.info.copy()
    evoked = mne.EvokedArray(
        avg_data,
        info,
        tmin=epochs_ref.tmin,
        comment=comment,
        nave=nave if nave is not None else 1,
        verbose=False,
    )
    return evoked


def _make_single_epoch_from_data(avg_data, epochs_ref, method_name="robust_avg"):
    """
    把平均后的二维数据 (n_channels, n_times) 包装成 1 个 epoch，
    方便后续统一按 epochs 接口做评估。
    """
    info = epochs_ref.info.copy()

    events = np.array([[0, 0, 1]], dtype=int)
    event_id = {method_name: 1}

    epochs_single = mne.EpochsArray(
        data=avg_data[np.newaxis, ...],
        info=info,
        events=events,
        event_id=event_id,
        tmin=epochs_ref.tmin,
        baseline=None,
        verbose=False,
    )
    return epochs_single


def _make_pseudo_raw_from_data(avg_data, epochs_ref):
    """
    由平均波形 (n_channels, n_times) 重构一个 pseudo-raw。
    注意: 这不是原始连续 raw，只是为了保存和接口兼容。
    """
    info = epochs_ref.info.copy()
    raw_pseudo = mne.io.RawArray(avg_data, info, verbose=False)
    return raw_pseudo


def _load_trial_weights_csv(weight_dir: str, fname: str, n_trials: int) -> np.ndarray:
    """从 trial 特征 csv 中加载 w_i 权重，并归一化。"""
    csv_path = os.path.join(weight_dir, f"{fname}_trial_features.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到特征文件: {csv_path}")

    dfw = pd.read_csv(csv_path)

    if "Trial" in dfw.columns:
        dfw = dfw.sort_values("Trial").reset_index(drop=True)

    if "w_i" not in dfw.columns:
        raise KeyError(f"{csv_path} 中没有 w_i 列。实际列: {dfw.columns.tolist()}")

    w = dfw["w_i"].to_numpy(dtype=float)

    if len(w) != n_trials:
        raise ValueError(f"trial 数不一致: csv={len(w)} vs epochs={n_trials}")

    w = np.where(np.isfinite(w), w, 0.0)
    s = w.sum()
    if s <= 0:
        raise ValueError("w_i 总和 <= 0")
    w = w / s
    return w




def sliding_baseline_correction(evoked, baseline_window=(-0.2, 0.0), win_len=0.05, win_step=0.01):
    """滑动窗口基线校正（核心函数，不变）"""
    corrected_evoked = evoked.copy()
    data = corrected_evoked.data.copy()
    data_ref = corrected_evoked.data.copy()
    sfreq = corrected_evoked.info['sfreq']
    times = corrected_evoked.times
    
    win_len_samples = int(win_len * sfreq)
    win_step_samples = int(win_step * sfreq)
    
    for t_idx in range(len(times)):
        win_start = max(0, t_idx - win_len_samples)
        win_end = t_idx
        if win_end <= win_start:
            continue
        baseline_mean = np.mean(data_ref[:, win_start:win_end], axis=1, keepdims=True)
        data[:, t_idx] -= baseline_mean[:, 0]
    
    # 基线期最终校准
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    if np.any(baseline_mask):
        final_baseline_mean = np.mean(data[:, baseline_mask], axis=1, keepdims=True)
        data -= final_baseline_mean
    
    corrected_evoked.data = data
    return corrected_evoked

def array_to_evoked(data, info, times, tmin=-0.2):
    """
    将加权后的数组转回mne.Evoked对象（适配滑动校正函数）
    参数：
        data: (n_channels, n_times) 加权后的数组（你的avg_data）
        info: mne.Info对象（从原始epochs/raw获取）
        times: 时间轴数组（如epochs.times）
        tmin: 时间起点（与你的数据一致，默认-0.2s）
    返回：
        evoked: mne.Evoked对象
    """
    evoked = mne.EvokedArray(data, info, tmin=tmin)
    evoked.times = times  # 确保时间轴完全匹配
    return evoked


def array_to_evoked(data, info, times, tmin=-0.2):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if data.shape[-1] != len(times):
        raise ValueError(f"data/times length mismatch: {data.shape[-1]} vs {len(times)}")

    inferred_tmin = float(times[0]) if len(times) else float(tmin)
    if len(times) > 1:
        dt = np.diff(times)
        expected_dt = 1.0 / float(info["sfreq"])
        if not np.allclose(dt, expected_dt, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"times are inconsistent with info['sfreq']: expected step {expected_dt}, got {dt[:3]}"
            )

    return mne.EvokedArray(data, info, tmin=inferred_tmin, verbose=False)


# def mcd_weighted_average(X: np.ndarray, w_i: np.ndarray):
#     """
#     对所有 trial 按给定权重做 soft weighted average。

#     X: (n_epochs, n_channels, n_times)
#     w_i: (n_epochs,) 权重，需要已归一化

#     返回 avg_data (n_channels, n_times) 以及有效样本数 neff
#     """
#     if X.shape[0] != len(w_i):
#         raise ValueError("w_i 长度与 trial 数不一致")

#     w_i = np.asarray(w_i, float)
#     s = w_i.sum()
#     if s <= 0:
#         raise ValueError("w_i 总和 <= 0")

#     w_i = w_i / s
#     avg_data = np.tensordot(w_i, X, axes=(0, 0))
#     neff = 1.0 / np.sum(w_i**2)


#     return avg_data, w_i, neff


import numpy as np
from robust_averaging_tools import sliding_baseline_correction, array_to_evoked

def mcd_weighted_average(X: np.ndarray, w_i: np.ndarray, 
                         # 以下为纯新增参数，全部后置且带默认值，不影响原有调用
                         info=None, 
                         times=None, 
                         baseline_window=(-0.2, 0.0),
                         remove_slow_drift=False, 
                         win_len=0.05, 
                         win_step=0.01):
    """
    对所有 trial 按给定权重做 soft weighted average，新增慢漂去除功能（兼容原有调用）。

    【原有参数（完全保留）】
    X: (n_epochs, n_channels, n_times)
    w_i: (n_epochs,) 权重，需要已归一化

    【新增参数（全部带默认值，不影响原有调用）】
    info: mne.Info对象 | None （remove_slow_drift=True时必填，从原始数据获取）
    times: np.ndarray | None （时间轴数组，如epochs.times，remove_slow_drift=True时必填）
    baseline_window: tuple 特征计算用的基线段（仅用于慢漂校正，默认(-0.2, 0.0)）
    remove_slow_drift: bool 是否启用滑动窗口去慢漂（默认False，即不启用）
    win_len: float 滑动窗口长度（秒，默认0.05）
    win_step: float 滑动窗口步长（秒，默认0.01）

    返回：
        avg_data (n_channels, n_times)：加权（+可选去漂）后的数据
        w_i (n_epochs,)：归一化后的权重
        neff：有效样本数
    """
    # 原始加权逻辑（一字不改，完全保留）
    if X.shape[0] != len(w_i):
        raise ValueError("w_i 长度与 trial 数不一致")

    w_i = np.asarray(w_i, float)
    s = w_i.sum()
    if s <= 0:
        raise ValueError("w_i 总和 <= 0")

    w_i = w_i / s
    avg_data = np.tensordot(w_i, X, axes=(0, 0))
    neff = 1.0 / np.sum(w_i**2)

    # 新增：滑动窗口去慢漂逻辑（仅当remove_slow_drift=True时触发）
    if remove_slow_drift:
        # 校验新增参数（仅在启用去漂时校验，不影响原有调用）
        if info is None or times is None:
            raise ValueError("启用remove_slow_drift时，必须传入info和times参数")
        # 步骤1：数组转Evoked（适配校正函数）
        evoked = array_to_evoked(avg_data, info, times, tmin=baseline_window[0])
        # 步骤2：滑动窗口校正去慢漂
        corrected_evoked = sliding_baseline_correction(
            evoked,
            baseline_window=baseline_window,
            win_len=win_len,
            win_step=win_step
        )
        # 步骤3：校正后的数据转回数组
        avg_data = corrected_evoked.data

    return avg_data, w_i, neff

def _save_method_outputs(
    subject_dir,
    subject_name,
    method_name,
    avg_data,
    epochs_ref,
    extra_meta=None,
    post_evoked_baseline=None,
):
    """
    保存:
      1) evoked
      2) single-epoch
      3) pseudo-raw
      4) meta json
    """
    method_dir = os.path.join(subject_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    evoked = _make_evoked_from_data(
        avg_data,
        epochs_ref,
        comment=f"{subject_name}_{method_name}",
        nave=len(epochs_ref),
    )
    if post_evoked_baseline is not None:
        evoked = evoked.copy().apply_baseline(post_evoked_baseline)

    corrected_data = evoked.data.copy()
    epochs_single = _make_single_epoch_from_data(
        corrected_data,
        epochs_ref,
        method_name=method_name,
    )
    raw_pseudo = _make_pseudo_raw_from_data(corrected_data, epochs_ref)

    # evoked_path = os.path.join(method_dir, f"{subject_name}_{method_name}-ave.fif")
    epo_path = os.path.join(method_dir, f"{subject_name}_{method_name}-epo.fif")
    # raw_path = os.path.join(method_dir, f"{subject_name}_{method_name}-raw.fif")
    meta_path = os.path.join(method_dir, f"{subject_name}_{method_name}_meta.json")

    # evoked.save(evoked_path, overwrite=True)
    if method_name == "mcd_weighted":
        epochs_single.save(epo_path, overwrite=True)
    # raw_pseudo.save(raw_path, overwrite=True)

    meta = {
        "subject": subject_name,
        "method": method_name,
        "n_original_epochs": int(len(epochs_ref)),
        "tmin": float(epochs_ref.tmin),
        "tmax": float(epochs_ref.tmax),
        "sfreq": float(epochs_ref.info["sfreq"]),
        "n_channels": int(len(epochs_ref.copy().pick("data").ch_names)),
        "note": (
            "raw file is pseudo-raw reconstructed from robust-averaged epoch; "
            "not the original continuous raw recording."
        ),
    }
    if extra_meta is not None:
        meta.update(extra_meta)
    meta["post_evoked_baseline"] = post_evoked_baseline

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "evoked": evoked,
        "epochs": epochs_single,
        "raw": raw_pseudo,
        "paths": {
            # "evoked": evoked_path,
            "epochs": epo_path if method_name == "mcd_weighted" else None,
            # "raw": raw_path,
            "meta": meta_path,
        },
    }


import numpy as np
import mne
