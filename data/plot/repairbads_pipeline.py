from __future__ import annotations

import numpy as np
import mne

from pyriemann.utils.covariance import covariances
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import eigh
from scipy.ndimage import binary_dilation
from scipy.stats import gmean, gstd
from sklearn.cluster import KMeans


def detect_bad_segments_riemann(
    raw_sim: mne.io.BaseRaw,
    *,
    win_len: float = 1.0,
    overlap: float = 0.1,
    n_clusters: int = 8,
    dilation_ms: float = 50.0,
):
    raw = raw_sim.copy()
    meg_picks = mne.pick_types(raw.info, meg=True, stim=False)
    Y = raw.get_data(picks=meg_picks)
    n_meg_local, n_times_local = Y.shape
    sfreq_local = raw.info["sfreq"]
    total_time = n_times_local / sfreq_local

    win_samples = int(win_len * sfreq_local)
    step_samples = int(win_len * (1 - overlap) * sfreq_local)
    win_starts = np.arange(0, n_times_local - win_samples + 1, step_samples)
    n_windows = len(win_starts)

    Yw = np.zeros((n_windows, n_meg_local, win_samples))
    for i, start in enumerate(win_starts):
        Yw[i] = Y[:, start:start + win_samples]

    ch_pos = np.array([raw.info["chs"][i]["loc"][:3] for i in meg_picks])
    n_clusters = max(1, min(n_clusters, n_meg_local))
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(ch_pos)
    sub_chs = {k: np.where(cluster_labels == k)[0] for k in range(n_clusters)}

    riemann_dist = np.zeros((n_clusters, n_windows))
    for k in range(n_clusters):
        Ysub = Yw[:, sub_chs[k], :]
        Ck = covariances(Ysub, estimator="lwf")
        C_bar_k = mean_riemann(Ck)
        for i in range(n_windows):
            riemann_dist[k, i] = distance_riemann(Ck[i], C_bar_k)

    bad_window = np.zeros(n_windows, dtype=bool)
    for _ in range(10):
        prev_bad = bad_window.copy()
        for k in range(n_clusters):
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

    dilate_samples = int(dilation_ms * sfreq_local / 1000)
    if dilate_samples > 0:
        bad_samples = binary_dilation(bad_samples, iterations=dilate_samples)

    detected_ratio = float(bad_samples.mean())
    time_axis = np.arange(n_times_local) / sfreq_local
    bad_runs = _bad_runs_from_mask(time_axis, bad_samples)

    annot = mne.Annotations(
        onset=[x[0] for x in bad_runs],
        duration=[x[1] for x in bad_runs],
        description=[x[2] for x in bad_runs],
        orig_time=raw.info["meas_date"],
    )
    raw_with_bad = raw.copy().set_annotations(annot)

    return {
        "detected_ratio": detected_ratio,
        "bad_samples": bad_samples,
        "bad_runs": bad_runs,
        "raw_with_bad": raw_with_bad,
        "time_axis": time_axis,
        "total_time": total_time,
        "n_windows": n_windows,
        "sub_chs": sub_chs,
    }


def repair_bad_segments_joint_decorrelation(
    raw_sim: mne.io.BaseRaw,
    bad_samples: np.ndarray,
    *,
    max_artifact_components: int | None = None,
    eig_ratio_floor: float = 1.15,
):
    raw_repaired = raw_sim.copy()
    meg_picks = mne.pick_types(raw_repaired.info, meg=True, stim=False)
    Y = raw_repaired.get_data()
    X_meg = Y[meg_picks].copy()

    clean_mask = ~bad_samples
    if bad_samples.sum() == 0 or clean_mask.sum() == 0:
        return {
            "raw_repaired": raw_repaired,
            "artifact_components": 0,
            "artifact_eigvals": np.array([]),
            "repair_applied": False,
        }

    C_clean = covariances(X_meg[:, clean_mask][None, :, :], estimator="lwf")[0]
    C_bad = covariances(X_meg[:, bad_samples][None, :, :], estimator="lwf")[0]

    reg = 1e-10 * np.trace(C_clean) / max(C_clean.shape[0], 1)
    C_clean_reg = C_clean + reg * np.eye(C_clean.shape[0])

    eigvals, filters = eigh(C_bad, C_clean_reg)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    filters = filters[:, order]

    artifact_mask = eigvals > eig_ratio_floor
    if max_artifact_components is not None:
        keep_n = max(0, min(int(max_artifact_components), len(eigvals)))
        forced = np.zeros_like(artifact_mask)
        forced[:keep_n] = True
        artifact_mask = np.logical_and(artifact_mask, forced)

    n_artifact = int(artifact_mask.sum())
    if n_artifact == 0:
        return {
            "raw_repaired": raw_repaired,
            "artifact_components": 0,
            "artifact_eigvals": eigvals,
            "repair_applied": False,
        }

    mixing = np.linalg.pinv(filters.T)
    keep_mask = ~artifact_mask
    projector = mixing[:, keep_mask] @ filters[:, keep_mask].T

    repaired_meg = X_meg.copy()
    repaired_meg[:, bad_samples] = projector @ X_meg[:, bad_samples]
    Y[meg_picks] = repaired_meg

    raw_repaired = mne.io.RawArray(Y, raw_repaired.info.copy(), verbose=False)
    bad_runs = _bad_runs_from_mask(
        np.arange(len(bad_samples)) / raw_repaired.info["sfreq"],
        bad_samples,
        description="BAD_RIEMANN_REPAIRED",
    )
    raw_repaired.set_annotations(
        mne.Annotations(
            onset=[x[0] for x in bad_runs],
            duration=[x[1] for x in bad_runs],
            description=[x[2] for x in bad_runs],
            orig_time=raw_repaired.info["meas_date"],
        )
    )

    return {
        "raw_repaired": raw_repaired,
        "artifact_components": n_artifact,
        "artifact_eigvals": eigvals,
        "repair_applied": True,
    }


def run_repairbads(raw_sim: mne.io.BaseRaw, make_plots: bool = False):
    detect_result = detect_bad_segments_riemann(raw_sim)
    repair_result = repair_bad_segments_joint_decorrelation(
        raw_sim,
        detect_result["bad_samples"],
    )

    result = {**detect_result, **repair_result}
    if make_plots:
        print(
            f"数据分窗完成：共{detect_result['total_time']:.1f}秒，"
            f"生成{detect_result['n_windows']}个滑动窗"
        )
        print(
            "传感器聚类完成："
            f"{len(detect_result['sub_chs'])}个子集，"
            f"各子集通道数 {[len(v) for v in detect_result['sub_chs'].values()]}"
        )
        print(f"坏段总时长占比：{detect_result['detected_ratio'] * 100:.1f}%")
        print(f"joint decorrelation 去除分量数：{repair_result['artifact_components']}")
    return result


def _bad_runs_from_mask(time_axis, bad_samples, description="BAD_RIEMANN"):
    bad_runs = []
    in_bad = False
    start_time = 0.0
    for t in range(len(bad_samples)):
        if bad_samples[t] and not in_bad:
            start_time = time_axis[t]
            in_bad = True
        elif not bad_samples[t] and in_bad:
            bad_runs.append((start_time, time_axis[t] - start_time, description))
            in_bad = False
    if in_bad:
        bad_runs.append((start_time, time_axis[-1] - start_time, description))
    return bad_runs
