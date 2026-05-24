#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_DIR = Path("result") / "sim_robust_metrics"
METRIC_COLUMNS = [
    "signal_peak_ratio_to_reference",
    "signal_peak_abs_error_to_reference",
    "baseline_rms",
    "snr_like",
    "peak_latency_error",
    "corr_to_reference",
    "rmse_to_reference",
]
TABLE_METRIC_COLUMNS = [
    "rmse_to_reference",
    "corr_to_reference",
    "peak_latency_error",
    "baseline_rms",
    "signal_peak_abs_error_to_reference",
    "signal_peak_ratio_to_reference",
]
PLOT_METRIC_COLUMNS = [
    "signal_peak_ratio_to_reference",
    "signal_peak_abs_error_to_reference",
    "corr_to_reference",
    "rmse_to_reference",
    "peak_latency_error",
]
METHOD_ORDER = [
    "ground_truth",
    "conventional",
    "median",
    "tanh_mean",
    "trimmed_mean",
    "wacfm",
    # "cor_wacfm",
    "mcd_weighted",
]
PLOT_METHOD_ORDER = [method for method in METHOD_ORDER if method != "ground_truth"]
METHOD_COLORS = {
    "conventional": "#50749e",
    "median": "#f1b584",
    "tanh_mean": "#ccadc3",
    "trimmed_mean": "#5e994d",
    "wacfm": "#4c9085",
    "mcd_weighted": "#d97a7a",
}
METRIC_PLOT_CONFIG = {
    "signal_peak_ratio_to_reference": {"scale": 1.0, "ylabel": "Signal peak / reference"},
    "signal_peak_abs_error_to_reference": {"scale": 1e15, "ylabel": "Signal peak abs error (fT)"},
    "baseline_rms": {"scale": 1e15, "ylabel": "Baseline RMS (fT)"},
    "snr_like": {"scale": 1.0, "ylabel": "SNR-like"},
    "peak_latency_error": {"scale": 1e3, "ylabel": "Peak latency error (ms)"},
    "corr_to_reference": {"scale": 1.0, "ylabel": "Correlation to reference"},
    "rmse_to_reference": {"scale": 1e15, "ylabel": "RMSE to reference (fT)"},
}


def find_metric_csv_files(metric_dir: Path) -> list[Path]:
    if not metric_dir.exists():
        raise FileNotFoundError(f"Metric directory not found: {metric_dir}")

    csv_files = sorted(metric_dir.glob("*.csv"))
    return [
        path for path in csv_files
        if "__metrics__" in path.name and "summary" not in path.name
    ]


def load_metric_tables(csv_files: list[Path]) -> pd.DataFrame:
    tables = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        tables.append(df)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def add_peak_latency_error(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    required_cols = {"source_file", "method", "peak_latency"}
    if not required_cols.issubset(df_all.columns):
        raise KeyError(f"Missing required columns for peak_latency_error: {sorted(required_cols)}")

    gt_latency = (
        df_all.loc[df_all["method"] == "ground_truth", ["source_file", "peak_latency"]]
        .drop_duplicates(subset=["source_file"])
        .rename(columns={"peak_latency": "peak_latency_gt"})
    )
    df_out = df_all.merge(gt_latency, on="source_file", how="left", validate="many_to_one")
    df_out["peak_latency_error"] = np.abs(df_out["peak_latency"] - df_out["peak_latency_gt"])
    return df_out


def compute_method_metric_means(df_all: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    available_metrics = [col for col in metric_columns if col in df_all.columns]
    if "method" not in df_all.columns:
        raise KeyError("Column 'method' not found in metric tables.")
    if not available_metrics:
        raise KeyError(f"No target metric columns found. Expected any of: {metric_columns}")

    df_mean = (
        df_all.groupby("method", as_index=False)[available_metrics]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )

    order_map = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    df_mean["_method_order"] = df_mean["method"].map(order_map).fillna(len(METHOD_ORDER))
    df_mean = (
        df_mean.sort_values(["_method_order", "method"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )
    return df_mean


# def compute_method_metric_stats(df_all: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
def compute_method_metric_stats(df_all: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    available_metrics = [col for col in metric_columns if col in df_all.columns]
    df_stats_src = df_all[df_all["method"] != "cor_wacfm"].copy()
    if df_stats_src.empty:
        return pd.DataFrame()

    rows = []
    ordered_methods = []
    if "ground_truth" in df_stats_src["method"].unique():
        ordered_methods.append("ground_truth")
    ordered_methods.extend([m for m in PLOT_METHOD_ORDER if m in df_stats_src["method"].unique()])

    extra_methods = sorted(set(df_stats_src["method"].dropna().unique()) - set(ordered_methods))
    ordered_methods.extend(extra_methods)

    for method in ordered_methods:
        row = {"method": method}
        df_method = df_stats_src[df_stats_src["method"] == method]
        for metric in available_metrics:
            vals = pd.to_numeric(df_method[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            cfg = METRIC_PLOT_CONFIG.get(metric, {"scale": 1.0})
            vals_scaled = vals * cfg["scale"]
            row[metric] = (
                f"{vals_scaled.mean():.2f} \u00B1 {vals_scaled.std(ddof=1):.2f}"
                if len(vals_scaled) > 1
                else f"{vals_scaled.mean():.2f} \u00B1 0.00"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def print_metric_summary(df_mean: pd.DataFrame, metric_columns: list[str]) -> None:
    print(f"Loaded {len(df_mean)} methods.")
    print()

    for metric in metric_columns:
        if metric not in df_mean.columns:
            continue
        print(metric)
        for _, row in df_mean.iterrows():
            print(f"  {row['method']}: {row[metric]:.6g}")
        print()


def print_metric_stats_table(df_stats: pd.DataFrame, metric_columns: list[str]) -> None:
    if df_stats.empty:
        print("No metric statistics available.")
        return

    cols = ["method"] + [metric for metric in metric_columns if metric in df_stats.columns]
    print("Metric statistics table (mean \u00B1 std):")
    print(df_stats[cols].to_string(index=False))
    print()
    print()


def plot_metric_scatter_boxplots(
    df_all: pd.DataFrame,
    metric_columns: list[str],
) -> None:
    if df_all.empty:
        print("No metric data found for plotting.")
        return

    
    df_plot = df_all[df_all["method"] != "ground_truth"].copy()
    df_plot = df_plot[df_plot["method"] != "cor_wacfm"]
    if df_plot.empty:
        print("No non-ground-truth metric data found for plotting.")
        return

    order_map = {method: idx for idx, method in enumerate(PLOT_METHOD_ORDER)}
    available_methods = [m for m in PLOT_METHOD_ORDER if m in df_plot["method"].unique()]
    extra_methods = sorted(set(df_plot["method"].dropna().unique()) - set(available_methods))
    method_order = [m for m in available_methods if m != "mcd_weighted"]
    method_order += extra_methods
    if "mcd_weighted" in available_methods:
        method_order.append("mcd_weighted")

    available_metric_columns = [metric for metric in metric_columns if metric in df_plot.columns]
    if not available_metric_columns:
        print("No target metric columns found for plotting.")
        return

    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
    axes = np.asarray(axes).reshape(-1)

    for ax, metric in zip(axes, available_metric_columns):
        if metric not in df_plot.columns:
            continue

        cfg = METRIC_PLOT_CONFIG.get(metric, {"scale": 1.0, "ylabel": metric})
        scale = cfg["scale"]
        ylabel = cfg["ylabel"]

        data_by_method = []
        labels = []
        box_colors = []

        for pos, method in enumerate(method_order, start=1):
            vals = df_plot.loc[df_plot["method"] == method, metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue

            vals_scaled = vals * scale
            data_by_method.append(vals_scaled)
            labels.append(method)
            color = METHOD_COLORS.get(method, "#808080")
            box_colors.append(color)

            rng = np.random.default_rng(20260415 + pos)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals_scaled))
            ax.scatter(
                np.full(len(vals_scaled), pos, dtype=float) + jitter,
                vals_scaled,
                s=22,
                alpha=0.65,
                color=color,
                edgecolors="none",
            )

        if not data_by_method:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data_by_method,
            labels=labels,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for median in bp["medians"]:
            median.set_color("#333333")
            median.set_linewidth(1.6)
        for whisker in bp["whiskers"]:
            whisker.set_color("#666666")
        for cap in bp["caps"]:
            cap.set_color("#666666")

        ax.set_title(metric)
        ax.set_xlabel("Method")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for tick in ax.get_xtiscklabels():
            tick.set_ha("right")

    for ax in axes[len(available_metric_columns):]:
        ax.set_visible(False)

    fig.suptitle("Metric scatter + boxplots", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



csv_files = find_metric_csv_files(METRIC_DIR)
print(f"Found {len(csv_files)} metric CSV files in: {METRIC_DIR}")

df_all = load_metric_tables(csv_files)
df_all = add_peak_latency_error(df_all)
df_mean = compute_method_metric_means(df_all, METRIC_COLUMNS)
print_metric_summary(df_mean, METRIC_COLUMNS)
df_stats = compute_method_metric_stats(df_all, TABLE_METRIC_COLUMNS)
print_metric_stats_table(df_stats, TABLE_METRIC_COLUMNS)

#%%
plot_metric_scatter_boxplots(df_all, PLOT_METRIC_COLUMNS)

#%%
