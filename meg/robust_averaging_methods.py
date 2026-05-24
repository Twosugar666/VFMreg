import numpy as np


def robust_median_average(X):
    """
    X: (n_epochs, n_channels, n_times)
    return avg_data: (n_channels, n_times)
    """
    return np.median(X, axis=0)


def robust_trimmed_mean(X, trim_prop=0.1):
    """
    对每个 channel x time 点，沿 trial 维做对称 trimmed mean
    X: (n_epochs, n_channels, n_times)
    """
    n_epochs = X.shape[0]
    g = int(np.floor(trim_prop * n_epochs))

    if g <= 0:
        return X.mean(axis=0)

    if 2 * g >= n_epochs:
        raise ValueError(
            f"trim_prop={trim_prop} 过大，n_epochs={n_epochs} 时会把样本全部截掉。"
        )

    Xs = np.sort(X, axis=0)
    Xt = Xs[g : n_epochs - g, :, :]
    return Xt.mean(axis=0)


def robust_mcd_support_trimmed_mean(X, keep_mask, trim_prop=0.1):
    """
    先按 keep_mask 保留试次，再在保留集合内做 trimmed mean。

    X: (n_epochs, n_channels, n_times)
    keep_mask: (n_epochs,) bool
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape[0] != X.shape[0]:
        raise ValueError("keep_mask 长度与 trial 数不一致")

    X_keep = X[keep_mask]

    if X_keep.shape[0] == 0:
        raise ValueError("keep_mask 没有保留任何试次")

    # 保底：保留太少时，退回简单均值
    if X_keep.shape[0] < 5:
        return X_keep.mean(axis=0)

    # 如果 trim_prop 太大导致样本会被截光，也退回均值
    g = int(np.floor(trim_prop * X_keep.shape[0]))
    if 2 * g >= X_keep.shape[0]:
        return X_keep.mean(axis=0)

    return robust_trimmed_mean(X_keep, trim_prop=trim_prop)

def _tanh_rank_weights(n_epochs, k=0.35, s=None, trim_count=1):
    """
    构造 tanh mean 的秩权重。
    """
    ranks = np.arange(1, n_epochs + 1)
    d = np.minimum(ranks, n_epochs - ranks + 1).astype(float)

    if s is None:
        trim_count = max(0, int(trim_count))
        s = np.tanh(k * trim_count)

    w = np.tanh(k * d) - s
    w = np.clip(w, 0.0, None)

    if np.all(w == 0):
        w = np.ones_like(w)

    w = w / w.sum()
    return w


def robust_tanh_mean(X, k=0.35, s=None, trim_count=1):
    """
    对每个 channel x time 点:
      1) 沿 trial 排序
      2) 按秩赋 tanh 权重
      3) 做加权平均
    """
    n_epochs = X.shape[0]
    weights = _tanh_rank_weights(n_epochs, k=k, s=s, trim_count=trim_count)

    Xs = np.sort(X, axis=0)
    avg_data = np.tensordot(weights, Xs, axes=(0, 0))
    return avg_data, weights


def _mask_time_window(times, window):
    if window is None:
        return np.ones_like(times, dtype=bool)
    tmin_w, tmax_w = window
    return (times >= tmin_w) & (times <= tmax_w)


def _safe_pearsonr_1d(a, b, eps=1e-30):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size:
        raise ValueError("a 和 b 长度不一致")
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std < eps or b_std < eps:
        return 0.0
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    r = np.sum(a0 * b0) / (np.sqrt(np.sum(a0**2)) * np.sqrt(np.sum(b0**2)) + eps)
    return float(np.clip(r, -1.0, 1.0))


def _get_metric_floor_auto(X, v_ref, min_floor=1e-30):
    d = np.sum(np.abs(X - v_ref[None, :, :]), axis=(1, 2))
    med_d = np.median(d)
    floor = max(min_floor, 1e-6 * med_d)
    return float(floor)


def _wacfm_dissimilarity_l1(X, v, metric_floor=None):
    rho = np.sum(np.abs(X - v[None, :, :]), axis=(1, 2))

    if metric_floor is None:
        floor = _get_metric_floor_auto(X, v)
    else:
        floor = float(metric_floor)

    rho = rho + floor
    return rho, floor


def _wacfm_update_weights_from_rho(rho, m=2.0):
    exponent = 1.0 / (1.0 - m)
    w = np.power(rho, exponent)

    w = np.asarray(w, dtype=float)
    w[~np.isfinite(w)] = 0.0

    s = np.sum(w)
    if s <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w


def _wacfm_reject_small_weights(w, c=100.0):
    w = np.asarray(w, dtype=float).copy()
    n = w.size
    thr = 1.0 / (c * n)
    w[w < thr] = 0.0

    s = np.sum(w)
    if s <= 0:
        w[:] = 1.0 / n
    else:
        w /= s
    return w, thr


def _wacfm_update_v(X, w, m=2.0):
    wm = np.power(w, m)
    denom = np.sum(wm)
    if denom <= 0:
        return np.mean(X, axis=0)
    v = np.tensordot(wm, X, axes=(0, 0)) / denom
    return v


def _init_template(X, init_mode="median"):
    if init_mode == "mean":
        return np.mean(X, axis=0)
    if init_mode == "median":
        return np.median(X, axis=0)
    raise ValueError("init_mode 必须是 'mean' 或 'median'")


def _compute_epoch_correlations(X, v_ref, times=None, corr_window=None):
    if times is None:
        mask = np.ones(v_ref.shape[1], dtype=bool)
    else:
        mask = _mask_time_window(times, corr_window)

    v_use = v_ref[:, mask].ravel()

    u = np.zeros(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        xi = X[i, :, mask].ravel()
        r = _safe_pearsonr_1d(xi, v_use)
        u[i] = max(0.0, r)

    return u


def robust_abs_wacfm(
    X,
    *,
    m=2.0,
    xi=1e-5,
    max_iter=200,
    reject_c=100.0,
    metric_floor=None,
    init_mode="median",
):
    n_epochs = X.shape[0]

    v_prev = _init_template(X, init_mode=init_mode)

    rho0, floor_used = _wacfm_dissimilarity_l1(X, v_prev, metric_floor=metric_floor)
    w_prev = _wacfm_update_weights_from_rho(rho0, m=m)
    w_prev, reject_thr = _wacfm_reject_small_weights(w_prev, c=reject_c)

    v_prev = _wacfm_update_v(X, w_prev, m=m)

    n_iter_done = 0
    for it in range(1, max_iter + 1):
        rho, _ = _wacfm_dissimilarity_l1(X, v_prev, metric_floor=floor_used)
        w_new = _wacfm_update_weights_from_rho(rho, m=m)
        w_new, _ = _wacfm_reject_small_weights(w_new, c=reject_c)

        v_new = _wacfm_update_v(X, w_new, m=m)

        dw = np.linalg.norm(w_new - w_prev)
        n_iter_done = it

        w_prev = w_new
        v_prev = v_new

        if dw < xi:
            break

    meta = {
        "method_detail": "absWACFM",
        "m": float(m),
        "xi": float(xi),
        "max_iter": int(max_iter),
        "reject_c": float(reject_c),
        "metric_floor_used": float(floor_used),
        "reject_threshold": float(reject_thr),
        "n_iter_done": int(n_iter_done),
        "init_mode": init_mode,
        "n_epochs": int(n_epochs),
    }
    return v_prev, w_prev, meta


def robust_cor_wacfm(
    X,
    *,
    times=None,
    corr_window=None,
    m=2.0,
    xi=1e-5,
    max_iter=200,
    reject_c=100.0,
    metric_floor=None,
    init_mode="median",
):
    n_epochs = X.shape[0]

    v_prev = _init_template(X, init_mode=init_mode)

    rho0, floor_used = _wacfm_dissimilarity_l1(X, v_prev, metric_floor=metric_floor)
    w_prev = _wacfm_update_weights_from_rho(rho0, m=m)
    w_prev, reject_thr = _wacfm_reject_small_weights(w_prev, c=reject_c)

    v_prev = _wacfm_update_v(X, w_prev, m=m)

    n_iter_done = 0
    u_last = np.ones(n_epochs, dtype=float)

    for it in range(1, max_iter + 1):
        u = _compute_epoch_correlations(
            X, v_prev, times=times, corr_window=corr_window
        )

        rho, _ = _wacfm_dissimilarity_l1(X, v_prev, metric_floor=floor_used)
        w_new = _wacfm_update_weights_from_rho(rho, m=m)
        w_new = w_new * u

        s = np.sum(w_new)
        if s <= 0:
            w_new = np.ones_like(w_new) / len(w_new)
        else:
            w_new = w_new / s

        w_new, _ = _wacfm_reject_small_weights(w_new, c=reject_c)
        v_new = _wacfm_update_v(X, w_new, m=m)

        dw = np.linalg.norm(w_new - w_prev)
        n_iter_done = it

        w_prev = w_new
        v_prev = v_new
        u_last = u

        if dw < xi:
            break

    meta = {
        "method_detail": "corWACFM",
        "m": float(m),
        "xi": float(xi),
        "max_iter": int(max_iter),
        "reject_c": float(reject_c),
        "metric_floor_used": float(floor_used),
        "reject_threshold": float(reject_thr),
        "n_iter_done": int(n_iter_done),
        "init_mode": init_mode,
        "corr_window": corr_window,
        "correlation_weights_last_iter": u_last.tolist(),
        "n_epochs": int(n_epochs),
    }
    return v_prev, w_prev, meta


import numpy as np
from scipy.signal import kaiserord, firwin, filtfilt


# =========================================================
# Modified DTW average (Molina et al., 2024 style)
# =========================================================

def _dtw_min_cost_path_l1(ref, sig, max_warp_offset=None):
    """
    ref, sig: 1D arrays, shape (n_times,)
    local cost: |ref[i] - sig[j]|
    allowed steps: (1,1), (1,0), (0,1)
    """
    ref = np.asarray(ref, dtype=float)
    sig = np.asarray(sig, dtype=float)

    n = len(ref)
    m = len(sig)
    D = np.full((n, m), np.inf, dtype=float)
    P = np.full((n, m), -1, dtype=np.int8)

    if max_warp_offset is None:
        max_warp_offset = max(n, m)
    max_warp_offset = max(0, int(max_warp_offset))

    if abs(n - m) > max_warp_offset:
        raise ValueError("max_warp_offset 太小，无法完成 DTW 对齐。")

    for i in range(n):
        j_start = max(0, i - max_warp_offset)
        j_end = min(m, i + max_warp_offset + 1)
        for j in range(j_start, j_end):
            cost = abs(ref[i] - sig[j])

            if i == 0 and j == 0:
                D[i, j] = cost
                continue

            cand = (
                D[i - 1, j - 1] if (i > 0 and j > 0) else np.inf,
                D[i - 1, j] if i > 0 else np.inf,
                D[i, j - 1] if j > 0 else np.inf,
            )
            k = int(np.argmin(cand))
            best = cand[k]
            if not np.isfinite(best):
                continue
            D[i, j] = cost + best
            P[i, j] = k

    if not np.isfinite(D[n - 1, m - 1]):
        raise RuntimeError("DTW failed under the current max_warp_offset constraint.")

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        move = P[i, j]
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        elif move == 2:
            j -= 1
        else:
            raise RuntimeError("DTW backtracking failed.")
        path.append((i, j))

    path.reverse()
    return path, float(D[n - 1, m - 1])


def _restrict_path_remove_ref_stalls(path):
    """
    论文思路：去掉那些“不推进 reference 索引”的 (0,1) 步所对应的后续路径点，
    避免 warped signal 无约束拉长。

    实现上等价为：
    - 如果当前点和上一个保留点的 i 相同，说明 reference 索引未推进 -> 丢弃当前点
    - 仅保留 reference 索引前进的点

    返回 restricted_path
    """
    if len(path) == 0:
        return path

    restricted = [path[0]]
    for pt in path[1:]:
        if pt[0] != restricted[-1][0]:
            restricted.append(pt)

    return restricted


def _warp_signal_from_restricted_path(sig, restricted_path, target_len):
    """
    用 restricted path 重建 warped signal。
    取 restricted_path 中的 trial 索引 j，对应采样 sig[j]。
    如长度不足，则用末值补齐；如超出则裁剪。
    """
    if len(restricted_path) == 0:
        raise ValueError("restricted_path 为空")

    j_idx = [j for _, j in restricted_path]
    warped = np.asarray(sig[j_idx], dtype=float)

    if len(warped) < target_len:
        pad = np.full(target_len - len(warped), warped[-1], dtype=float)
        warped = np.concatenate([warped, pad], axis=0)
    elif len(warped) > target_len:
        warped = warped[:target_len]

    return warped


def _lowpass_fir_kaiser_1d(x, sfreq, cutoff_hz, transition_hz=5.0, atten_db=60.0):
    """
    低通 FIR + Kaiser 窗，尽量贴近文中描述。
    """
    x = np.asarray(x, dtype=float)
    nyq = sfreq / 2.0

    cutoff_hz = min(float(cutoff_hz), nyq * 0.95)
    if cutoff_hz <= 0:
        return x.copy()

    # 过渡带不能太小
    transition_hz = max(0.5, float(transition_hz))
    width = transition_hz / nyq

    numtaps, beta = kaiserord(atten_db, width)
    numtaps = max(numtaps, 5)
    if numtaps % 2 == 0:
        numtaps += 1

    taps = firwin(
        numtaps,
        cutoff=cutoff_hz / nyq,
        window=("kaiser", beta),
        pass_zero="lowpass"
    )

    # trial 很短时，filtfilt pad 可能失败，退化为 same 卷积
    padlen = 3 * (len(taps) - 1)
    if len(x) <= padlen:
        y_full = np.convolve(x, taps, mode="full")
        start = (len(y_full) - len(x)) // 2
        y = y_full[start : start + len(x)]
    else:
        y = filtfilt(taps, [1.0], x, method="pad")

    return y


def robust_modified_dtw_average(
    X,
    *,
    times=None,
    sfreq,
    do_filter=False,
    dtw_window=(0.0, 0.1),
    max_warp_ms=30.0,
    lpf_cutoff_hz=30.0,
    lpf_transition_hz=5.0,
    lpf_atten_db=60.0,
):
    """
    Modified DTW average with local-window alignment.

    Only the samples inside dtw_window are warped. The rest of each trial stays unchanged.
    max_warp_ms constrains the DTW path to a narrow Sakoe-Chiba-style band.
    """
    X = np.asarray(X, dtype=float)
    n_epochs, n_channels, n_times = X.shape
    warped_all = X.copy()

    if times is None:
        win_start = max(0, int(round(dtw_window[0] * sfreq)))
        win_end = min(n_times, int(round(dtw_window[1] * sfreq)) + 1)
        dtw_idx = np.arange(win_start, win_end, dtype=int)
    else:
        times = np.asarray(times, dtype=float)
        if times.shape[0] != n_times:
            raise ValueError("times length must match the time dimension of X")
        dtw_idx = np.flatnonzero((times >= dtw_window[0]) & (times <= dtw_window[1]))

    if dtw_idx.size == 0:
        raise ValueError("dtw_window contains no samples")

    win_start = int(dtw_idx[0])
    win_end = int(dtw_idx[-1]) + 1
    target_len = win_end - win_start
    max_warp_samples = max(1, int(round(max_warp_ms * sfreq / 1000.0)))

    ref_avg = np.mean(X, axis=0)
    path_lens = []
    costs = []

    for ch in range(n_channels):
        ref = ref_avg[ch, win_start:win_end]

        for t in range(n_epochs):
            sig = X[t, ch, win_start:win_end]
            path, cost = _dtw_min_cost_path_l1(
                ref,
                sig,
                max_warp_offset=max_warp_samples,
            )
            restricted_path = _restrict_path_remove_ref_stalls(path)
            warped = _warp_signal_from_restricted_path(
                sig=sig,
                restricted_path=restricted_path,
                target_len=target_len,
            )

            if do_filter:
                warped = _lowpass_fir_kaiser_1d(
                    warped,
                    sfreq=sfreq,
                    cutoff_hz=lpf_cutoff_hz,
                    transition_hz=lpf_transition_hz,
                    atten_db=lpf_atten_db,
                )

            warped_all[t, ch, win_start:win_end] = warped
            path_lens.append(len(restricted_path))
            costs.append(cost)

    avg_data = np.mean(warped_all, axis=0)

    meta = {
        "method_detail": "filtered_modified_DTW_average" if do_filter else "modified_DTW_average",
        "n_epochs": int(n_epochs),
        "n_channels": int(n_channels),
        "n_times": int(n_times),
        "sfreq": float(sfreq),
        "do_filter": bool(do_filter),
        "dtw_window": [float(dtw_window[0]), float(dtw_window[1])],
        "dtw_window_sample_start": int(win_start),
        "dtw_window_sample_end": int(win_end),
        "dtw_window_n_times": int(target_len),
        "max_warp_ms": float(max_warp_ms),
        "max_warp_samples": int(max_warp_samples),
        "lpf_cutoff_hz": float(lpf_cutoff_hz),
        "lpf_transition_hz": float(lpf_transition_hz),
        "lpf_atten_db": float(lpf_atten_db),
        "mean_restricted_path_len": float(np.mean(path_lens)),
        "std_restricted_path_len": float(np.std(path_lens)),
        "mean_dtw_cost": float(np.mean(costs)),
        "std_dtw_cost": float(np.std(costs)),
    }
    return avg_data, warped_all, meta
