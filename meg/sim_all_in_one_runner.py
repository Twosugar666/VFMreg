"""Decoupled runner for simulation, feature extraction, methods, and metrics."""

#%%
import gc
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import sim_feature_extract as feat
import sim_com_runner as comp


DEFAULT_SIM_SCRIPT = "sim_64.py"
DEFAULT_CONFIG_TAG = "sim64"
DEFAULT_N_REPEATS = 50
DEFAULT_BASE_SEED = 20260326
DEFAULT_APPLY_REPAIR = True


def _mode_tag(apply_repair: bool) -> str:
    return "repair" if apply_repair else "norepair"


def _data_dir(apply_repair: bool) -> str:
    return "data_sim" if apply_repair else "data_sim(norepair)"


def _configure_data_dirs(apply_repair: bool):
    data_dir = _data_dir(apply_repair)
    feat.DATA_DIR = data_dir
    comp.DATA_DIR = data_dir
    return data_dir


def _name_filter(config_tag: str) -> str:
    return f"__cfg-{config_tag}"


def _merged_feature_path(config_tag: str, apply_repair: bool) -> str:
    return os.path.join(feat.FEATURE_DIR, f"ALL_sim_trial_features__cfg-{config_tag}__{_mode_tag(apply_repair)}.csv")


def _metrics_path(config_tag: str, apply_repair: bool) -> str:
    return os.path.join(comp.METRIC_DIR, f"sim_robust_metrics__cfg-{config_tag}__{_mode_tag(apply_repair)}.csv")


def _summary_path(config_tag: str, apply_repair: bool) -> str:
    return os.path.join(comp.METRIC_DIR, f"sim_robust_metrics_summary__cfg-{config_tag}__{_mode_tag(apply_repair)}.csv")


def _single_dataset_metrics_path(dataset_name: str, apply_repair: bool) -> str:
    return os.path.join(
        comp.METRIC_DIR,
        f"{dataset_name}__metrics__{_mode_tag(apply_repair)}.csv",
    )


def _list_dataset_names_for_config(
    *,
    config_tag: str = DEFAULT_CONFIG_TAG,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
):
    _configure_data_dirs(apply_repair)
    fif_files = comp.find_fif_files(comp.DATA_DIR)
    dataset_names = [Path(f).stem for f in fif_files if _name_filter(config_tag) in Path(f).stem]
    print("Matched datasets:", dataset_names)
    return dataset_names


def _append_csv_rows(df: pd.DataFrame, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def _normalize_method_names(method_names):
    if method_names is None:
        return None
    if isinstance(method_names, str):
        method_names = [method_names]
    normalized = []
    for name in method_names:
        value = str(name).strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized or None


def _merge_metric_rows(existing_df: pd.DataFrame | None, new_df: pd.DataFrame, *, replace_methods=None) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        return new_df.copy()
    if new_df is None or new_df.empty:
        return existing_df.copy()

    replace_methods = _normalize_method_names(replace_methods)
    keys = new_df[["dataset", "method"]].drop_duplicates()
    if replace_methods is not None:
        keys = keys[keys["method"].isin(replace_methods)]

    merged = existing_df.copy()
    if not keys.empty:
        merged = merged.merge(keys.assign(_drop=True), on=["dataset", "method"], how="left")
        merged = merged[merged["_drop"].isna()].drop(columns="_drop")

    merged = pd.concat([merged, new_df], ignore_index=True)
    if {"dataset", "method"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["dataset", "method"], keep="last")
        merged = merged.sort_values(["dataset", "method"], kind="stable").reset_index(drop=True)
    return merged


def _load_existing_metrics_if_any(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _rebuild_aggregate_metric_tables(
    *,
    dataset_names,
    config_tag: str = DEFAULT_CONFIG_TAG,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
):
    per_dataset_tables = []
    for dataset_name in dataset_names:
        metric_path = _single_dataset_metrics_path(dataset_name, apply_repair)
        if os.path.exists(metric_path):
            per_dataset_tables.append(pd.read_csv(metric_path))

    metrics_path = _metrics_path(config_tag, apply_repair)
    summary_path = _summary_path(config_tag, apply_repair)

    if per_dataset_tables:
        df_metrics = pd.concat(per_dataset_tables, ignore_index=True)
        df_metrics = comp.annotate_metrics_with_dataset_tags(df_metrics)
        df_summary = comp.summarize_metrics_by_config(df_metrics)
        if df_summary.empty:
            df_summary = df_metrics.groupby("method", as_index=False).mean(numeric_only=True)
    else:
        df_metrics = pd.DataFrame()
        df_summary = pd.DataFrame()

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    df_metrics.to_csv(metrics_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    return df_metrics, df_summary, metrics_path, summary_path


@contextmanager
def temporary_module_overrides(module, overrides: dict | None):
    overrides = dict(overrides or {})
    old_values = {}
    for key, value in overrides.items():
        if not hasattr(module, key):
            raise AttributeError(f"{module.__name__} has no attribute {key}")
        old_values[key] = getattr(module, key)
        setattr(module, key, value)
    try:
        yield
    finally:
        for key, value in old_values.items():
            setattr(module, key, value)


def _mcd_summary_row(df_metrics: pd.DataFrame, label: str) -> dict:
    row = {"label": label}
    df_mcd = df_metrics[df_metrics["method"] == "mcd_weighted"].copy()
    if df_mcd.empty:
        return row
    for col in [
        "snr_like",
        "signal_peak_ratio_to_reference",
        "signal_peak_abs_error_to_reference",
        "baseline_rms",
        "corr_to_reference",
        "rmse_to_reference",
        "peak_latency_error",
    ]:
        if col in df_mcd.columns:
            row[col] = float(df_mcd[col].mean())
    return row


def run_single_simulation(
    *,
    sim_script: str,
    seed: int,
    config_tag: str,
    repeat_index: int,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
):
    env = os.environ.copy()
    env["SIM_SEED"] = str(seed)
    env["SIM_CONFIG_TAG"] = config_tag
    env["SIM_RUN_TAG"] = f"{repeat_index:03d}"
    env["SIM_DISABLE_PLOTS"] = "1"
    env["SIM_APPLY_REPAIR"] = "1" if apply_repair else "0"

    cmd = [sys.executable, str(Path(sim_script))]
    print(f"[Sim] config={config_tag} repeat={repeat_index:03d} seed={seed} apply_repair={apply_repair}")
    subprocess.run(cmd, check=True, env=env)


def run_repeated_simulations(
    *,
    sim_script: str = DEFAULT_SIM_SCRIPT,
    config_tag: str = DEFAULT_CONFIG_TAG,
    n_repeats: int = DEFAULT_N_REPEATS,
    base_seed: int = DEFAULT_BASE_SEED,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
):
    for repeat_index in range(1, n_repeats + 1):
        run_single_simulation(
            sim_script=sim_script,
            seed=base_seed + repeat_index - 1,
            config_tag=config_tag,
            repeat_index=repeat_index,
            apply_repair=apply_repair,
        )


def load_datasets_for_config(
    *,
    config_tag: str = DEFAULT_CONFIG_TAG,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
):
    _configure_data_dirs(apply_repair)
    datasets = comp.prepare_sim_datasets(name_contains=_name_filter(config_tag))
    print("Loaded datasets:", [ds["filename"] for ds in datasets])
    return datasets


def load_single_dataset(
    *,
    dataset_name: str,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
):
    _configure_data_dirs(apply_repair)
    datasets = comp.prepare_sim_datasets(name_exact=dataset_name)
    matched = [ds for ds in datasets if ds["filename"] == dataset_name]
    if not matched:
        available = [ds["filename"] for ds in datasets]
        raise FileNotFoundError(f"Dataset not found: {dataset_name}. Candidates: {available}")
    if len(matched) > 1:
        raise RuntimeError(f"Multiple datasets matched the same name: {dataset_name}")
    print("Loaded single dataset:", matched[0]["filename"])
    return matched[0]


def compute_features_for_datasets(
    datasets,
    *,
    config_tag: str = DEFAULT_CONFIG_TAG,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    reuse_existing: bool = True,
):
    _configure_data_dirs(apply_repair)
    feature_tables = []
    for ds in datasets:
        csv_path = os.path.join(feat.FEATURE_DIR, f'{ds["filename"]}_trial_features.csv')
        if reuse_existing and os.path.exists(csv_path):
            df_features = pd.read_csv(csv_path)
            print(f"[Feature] reuse {csv_path}")
        else:
            fif_path = os.path.join(feat.DATA_DIR, f'{ds["filename"]}.fif')
            print(f"[Feature] compute {fif_path}")
            df_features = feat.process_one_dataset(fif_path)
        feature_tables.append(df_features)

    if feature_tables:
        df_all_features = pd.concat(feature_tables, ignore_index=True)
    else:
        df_all_features = pd.DataFrame()

    merged_feature_path = _merged_feature_path(config_tag, apply_repair)
    df_all_features.to_csv(merged_feature_path, index=False)
    print(f"Saved merged feature CSV: {merged_feature_path}")
    return df_all_features, merged_feature_path


def compute_features_sequentially(
    dataset_names,
    *,
    config_tag: str = DEFAULT_CONFIG_TAG,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    reuse_existing: bool = True,
):
    _configure_data_dirs(apply_repair)
    merged_feature_path = _merged_feature_path(config_tag, apply_repair)
    if os.path.exists(merged_feature_path):
        os.remove(merged_feature_path)

    for dataset_name in dataset_names:
        csv_path = os.path.join(feat.FEATURE_DIR, f"{dataset_name}_trial_features.csv")
        if reuse_existing and os.path.exists(csv_path):
            df_features = pd.read_csv(csv_path)
            print(f"[Feature] reuse {csv_path}")
        else:
            fif_path = os.path.join(feat.DATA_DIR, f"{dataset_name}.fif")
            print(f"[Feature] compute {fif_path}")
            df_features = feat.process_one_dataset(fif_path)

        _append_csv_rows(df_features, merged_feature_path)
        del df_features
        gc.collect()

    df_all_features = pd.read_csv(merged_feature_path) if os.path.exists(merged_feature_path) else pd.DataFrame()
    print(f"Saved merged feature CSV: {merged_feature_path}")
    return df_all_features, merged_feature_path


def compute_features_for_single_dataset(
    dataset,
    *,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    reuse_existing: bool = True,
):
    df_all_features, _ = compute_features_for_datasets(
        [dataset],
        config_tag=dataset["filename"],
        apply_repair=apply_repair,
        reuse_existing=reuse_existing,
    )
    return df_all_features


def ensure_single_dataset_features(
    dataset,
    *,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    reuse_existing: bool = True,
):
    _configure_data_dirs(apply_repair)
    csv_path = os.path.join(feat.FEATURE_DIR, f'{dataset["filename"]}_trial_features.csv')
    if os.path.exists(csv_path):
        print(f"[Feature] found {csv_path}")
        return None, csv_path

    print(f"[Feature] missing {csv_path}, compute features for single dataset")
    df_all_features = compute_features_for_single_dataset(
        dataset,
        apply_repair=apply_repair,
        reuse_existing=reuse_existing,
    )
    return df_all_features, csv_path


def run_methods_for_datasets(
    datasets,
    *,
    selected_methods=None,
    plot_subjects: bool = True,
):
    robust_results_all = comp.run_all_robust_methods(
        datasets,
        baseline=comp.ROBUST_BASELINE,
        selected_methods=_normalize_method_names(selected_methods),
    )
    if plot_subjects:
        comp.plot_all_subject_comparisons(
            datasets,
            robust_results_all,
            show_figures=False,
        )
    return robust_results_all


def analyze_metrics(
    datasets,
    robust_results_all,
    *,
    config_tag: str = DEFAULT_CONFIG_TAG,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    plot_metrics: bool = False,
    replace_methods=None,
):
    df_metrics = comp.build_all_metrics(datasets, robust_results_all)
    df_metrics_annotated = comp.annotate_metrics_with_dataset_tags(df_metrics)
    replace_methods = _normalize_method_names(replace_methods)

    if replace_methods is not None:
        df_metrics_annotated = df_metrics_annotated[df_metrics_annotated["method"].isin(replace_methods)].copy()
        metrics_path = _metrics_path(config_tag, apply_repair)
        existing_df = _load_existing_metrics_if_any(metrics_path)
        df_metrics_annotated = _merge_metric_rows(
            existing_df,
            df_metrics_annotated,
            replace_methods=replace_methods,
        )

    df_summary = comp.summarize_metrics_by_config(df_metrics_annotated)
    if df_summary.empty:
        df_summary = df_metrics_annotated.groupby("method", as_index=False).mean(numeric_only=True)

    metrics_path = _metrics_path(config_tag, apply_repair)
    summary_path = _summary_path(config_tag, apply_repair)
    df_metrics_annotated.to_csv(metrics_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    boxplot_paths = comp.plot_metric_boxplots(
        df_metrics_annotated,
        config_tag=config_tag,
        mode_tag=_mode_tag(apply_repair),
    )

    if plot_metrics:
        comp.plot_metric_summary(df_metrics_annotated)

    return df_metrics_annotated, df_summary, metrics_path, summary_path, boxplot_paths


def save_single_dataset_metrics(
    dataset,
    robust_results_all,
    *,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    replace_methods=None,
    include_reference_rows: bool = True,
):
    df_metrics = comp.build_all_metrics([dataset], robust_results_all)
    if not include_reference_rows:
        replace_methods = _normalize_method_names(replace_methods)
        if replace_methods is None:
            raise ValueError("replace_methods is required when include_reference_rows=False")
        df_metrics = df_metrics[df_metrics["method"].isin(replace_methods)].copy()

    df_metrics = comp.annotate_metrics_with_dataset_tags(df_metrics)
    metrics_path = _single_dataset_metrics_path(dataset["filename"], apply_repair)
    existing_df = _load_existing_metrics_if_any(metrics_path)
    df_metrics = _merge_metric_rows(
        existing_df,
        df_metrics,
        replace_methods=replace_methods,
    )
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Saved single-dataset metric CSV: {metrics_path}")
    return df_metrics, metrics_path


def print_metrics_summary(df_metrics: pd.DataFrame, *, title: str = "Metrics summary"):
    if df_metrics is None or df_metrics.empty:
        print(f"{title}: no metrics available.")
        return

    summary_columns = [
        "rmse_to_reference",
        "corr_to_reference",
        "peak_latency_error",
        "baseline_rms",
        "signal_peak_abs_error_to_reference",
        "signal_peak_ratio_to_reference",
    ]
    available_columns = [col for col in summary_columns if col in df_metrics.columns]
    if not available_columns:
        print(f"{title}: no printable metric columns found.")
        return

    grouped = df_metrics.groupby("method", dropna=False)[available_columns]
    df_mean = grouped.mean(numeric_only=True)
    df_std = grouped.std(numeric_only=True)

    df_table = pd.DataFrame(index=df_mean.index)
    for col in available_columns:
        df_table[col] = [
            f"{mean_val:.6g} \u00B1 {std_val:.6g}" if pd.notna(std_val) else f"{mean_val:.6g} \u00B1 nan"
            for mean_val, std_val in zip(df_mean[col], df_std[col])
        ]

    df_table = df_table.reset_index().rename(columns={"method": "Method"})

    print(title)
    print("Metric statistics table (mean \u00B1 std):")
    print(df_table.to_string(index=False))


def run_single_dataset(
    *,
    dataset_name: str,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    do_features: bool = True,
    do_methods: bool = True,
    do_metrics: bool = True,
    reuse_existing_features: bool = True,
    plot_subject: bool = True,
    plot_metrics: bool = False,
    selected_methods=None,
    overwrite_existing_metrics: bool = False,
):
    selected_methods = _normalize_method_names(selected_methods)
    if overwrite_existing_metrics and selected_methods is None:
        raise ValueError("overwrite_existing_metrics=True requires selected_methods.")

    dataset = load_single_dataset(
        dataset_name=dataset_name,
        apply_repair=apply_repair,
    )
    datasets = [dataset]

    df_all_features = None
    if do_features:
        df_all_features = compute_features_for_single_dataset(
            dataset,
            apply_repair=apply_repair,
            reuse_existing=reuse_existing_features,
        )
    elif do_methods or do_metrics:
        auto_features, _ = ensure_single_dataset_features(
            dataset,
            apply_repair=apply_repair,
            reuse_existing=reuse_existing_features,
        )
        if auto_features is not None:
            df_all_features = auto_features

    robust_results_all = None
    if do_methods or do_metrics:
        robust_results_all = run_methods_for_datasets(
            datasets,
            selected_methods=selected_methods,
            plot_subjects=plot_subject,
        )

    df_metrics = None
    df_summary = None
    metrics_path = None
    summary_path = None
    boxplot_paths = []
    if do_metrics:
        if robust_results_all is None:
            raise RuntimeError("do_metrics=True requires robust results. Set do_methods=True.")
        replace_methods = _normalize_method_names(selected_methods) if overwrite_existing_metrics else None
        df_metrics, metrics_path = save_single_dataset_metrics(
            dataset,
            robust_results_all,
            apply_repair=apply_repair,
            replace_methods=replace_methods,
            include_reference_rows=not overwrite_existing_metrics,
        )

        print_metrics_summary(
            df_metrics,
            title=f"Single-dataset metrics: {dataset_name}",
        )

        if plot_metrics:
            comp.plot_metric_summary(df_metrics)

    return {
        "dataset": dataset,
        "df_all_features": df_all_features,
        "robust_results_all": robust_results_all,
        "df_metrics": df_metrics,
        "df_summary": df_summary,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "boxplot_paths": boxplot_paths,
    }


def run_single_dataset_with_mcd_overrides(
    *,
    dataset_name: str,
    feature_overrides: dict | None = None,
    method_overrides: dict | None = None,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    plot_subject: bool = False,
    plot_metrics: bool = False,
):
    with temporary_module_overrides(feat, feature_overrides), temporary_module_overrides(comp, method_overrides):
        result = run_single_dataset(
            dataset_name=dataset_name,
            apply_repair=apply_repair,
            do_features=True,
            do_methods=True,
            do_metrics=True,
            reuse_existing_features=False,
            plot_subject=plot_subject,
            plot_metrics=plot_metrics,
        )
    return result


def sweep_mcd_weighted_single_dataset(
    *,
    dataset_name: str,
    presets: list[dict],
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    plot_subject: bool = False,
    plot_metrics: bool = False,
):
    rows = []
    detailed_results = []
    for idx, preset in enumerate(presets, start=1):
        label = preset.get("label", f"preset_{idx:02d}")
        feature_overrides = preset.get("feature_overrides")
        method_overrides = preset.get("method_overrides")
        print(f"[Sweep] {label}")
        result = run_single_dataset_with_mcd_overrides(
            dataset_name=dataset_name,
            feature_overrides=feature_overrides,
            method_overrides=method_overrides,
            apply_repair=apply_repair,
            plot_subject=plot_subject,
            plot_metrics=plot_metrics,
        )
        row = _mcd_summary_row(result["df_metrics"], label)
        row["feature_overrides"] = feature_overrides
        row["method_overrides"] = method_overrides
        rows.append(row)
        detailed_results.append({"label": label, "result": result})

    df_sweep = pd.DataFrame(rows)
    if not df_sweep.empty and "corr_to_reference" in df_sweep.columns:
        df_sweep = df_sweep.sort_values(
            ["corr_to_reference", "rmse_to_reference", "snr_like"],
            ascending=[False, True, False],
        ).reset_index(drop=True)

    sweep_path = os.path.join(
        comp.METRIC_DIR,
        f"{dataset_name}__mcd_sweep__{_mode_tag(apply_repair)}.csv",
    )
    df_sweep.to_csv(sweep_path, index=False)
    print(f"Saved MCD sweep CSV: {sweep_path}")
    return df_sweep, detailed_results, sweep_path


def run_experiment(
    *,
    sim_script: str = DEFAULT_SIM_SCRIPT,
    config_tag: str = DEFAULT_CONFIG_TAG,
    n_repeats: int = DEFAULT_N_REPEATS,
    base_seed: int = DEFAULT_BASE_SEED,
    apply_repair: bool = DEFAULT_APPLY_REPAIR,
    do_generate: bool = True,
    do_features: bool = True,
    do_methods: bool = True,
    do_metrics: bool = True,
    reuse_existing_features: bool = True,
    plot_subjects: bool = True,
    plot_metrics: bool = False,
    sequential: bool = True,
    selected_methods=None,
    overwrite_existing_metrics: bool = False,
):
    selected_methods = _normalize_method_names(selected_methods)
    if overwrite_existing_metrics and selected_methods is None:
        raise ValueError("overwrite_existing_metrics=True requires selected_methods.")
    if do_generate:
        run_repeated_simulations(
            sim_script=sim_script,
            config_tag=config_tag,
            n_repeats=n_repeats,
            base_seed=base_seed,
            apply_repair=apply_repair,
        )

    dataset_names = _list_dataset_names_for_config(
        config_tag=config_tag,
        apply_repair=apply_repair,
    )

    if not sequential:
        datasets = load_datasets_for_config(
            config_tag=config_tag,
            apply_repair=apply_repair,
        )
    else:
        datasets = dataset_names

    df_all_features = None
    merged_feature_path = None
    if do_features:
        if sequential:
            df_all_features, merged_feature_path = compute_features_sequentially(
                dataset_names,
                config_tag=config_tag,
                apply_repair=apply_repair,
                reuse_existing=reuse_existing_features,
            )
        else:
            df_all_features, merged_feature_path = compute_features_for_datasets(
                datasets,
                config_tag=config_tag,
                apply_repair=apply_repair,
                reuse_existing=reuse_existing_features,
            )

    robust_results_all = None
    df_metrics = None
    df_summary = None
    metrics_path = None
    summary_path = None
    boxplot_paths = []
    metric_paths = []
    if do_methods or do_metrics:
        if sequential:
            for dataset_name in dataset_names:
                dataset = load_single_dataset(
                    dataset_name=dataset_name,
                    apply_repair=apply_repair,
                )
                rr = run_methods_for_datasets(
                    [dataset],
                    selected_methods=selected_methods,
                    plot_subjects=plot_subjects,
                )
                if do_metrics:
                    _, metrics_path_one = save_single_dataset_metrics(
                        dataset,
                        rr,
                        apply_repair=apply_repair,
                        replace_methods=selected_methods if overwrite_existing_metrics else None,
                        include_reference_rows=not overwrite_existing_metrics,
                    )
                    metric_paths.append(metrics_path_one)

                del rr
                del dataset
                gc.collect()

            if do_metrics:
                print(f"Sequential mode saved {len(metric_paths)} per-dataset metric CSV files.")
                df_metrics, df_summary, metrics_path, summary_path = _rebuild_aggregate_metric_tables(
                    dataset_names=dataset_names,
                    config_tag=config_tag,
                    apply_repair=apply_repair,
                )
            robust_results_all = None
        else:
            robust_results_all = run_methods_for_datasets(
                datasets,
                selected_methods=selected_methods,
                plot_subjects=plot_subjects,
            )

            if do_metrics:
                if robust_results_all is None:
                    raise RuntimeError("do_metrics=True requires robust results. Set do_methods=True.")
                df_metrics, df_summary, metrics_path, summary_path, boxplot_paths = analyze_metrics(
                    datasets,
                    robust_results_all,
                    config_tag=config_tag,
                    apply_repair=apply_repair,
                    plot_metrics=plot_metrics,
                    replace_methods=selected_methods if overwrite_existing_metrics else None,
                )

    return {
        "datasets": datasets,
        "df_all_features": df_all_features,
        "merged_feature_path": merged_feature_path,
        "robust_results_all": robust_results_all,
        "df_metrics": df_metrics,
        "df_summary": df_summary,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "metric_paths": metric_paths,
        "boxplot_paths": boxplot_paths,
    }


# Common usage:
# 1) 全流程从头跑
result = run_experiment(config_tag="sim64", n_repeats=5, apply_repair=True)
#
# 2) 已经有数据和特征，只重跑方法和指标
# result = run_experiment(
#     config_tag="sim64",
#     apply_repair=True,
#     do_generate=False,
#     do_features=False,
#     do_methods=True,
#     do_metrics=True,
# )

# Example for partial rerun:
# result = run_experiment(
#     config_tag="sim64",
#     apply_repair=True,
#     do_generate=False,
#     do_features=False,
#     do_methods=True,
#     do_metrics=True,
#     selected_methods=["mcd_weighted"],
#     overwrite_existing_metrics=True,
# )

# result = run_experiment(
#     config_tag="sim64",
#     apply_repair=True,
#     do_generate=False,
#     do_features=False,
#     reuse_existing_features=True,
#     do_methods=True,
#     do_metrics=True,
# )
#
# 3) 从现有数据重新计算特征，然后重新运行方法和指标
# result = run_experiment(
#     config_tag="sim64",
#     apply_repair=True,
#     do_generate=False,
#     do_features=True,
#     reuse_existing_features=False,
#     do_methods=True,
#     do_metrics=True,
# )
#
# # 4) 只跑某一条已有数据
# one_result = run_single_dataset(
#     dataset_name="raw_sim_repaired_badratio_15pct",
#     apply_repair=True,
#     do_features=True,
#     do_methods=True,
#     do_metrics=True,
#     plot_subject=True,
# )
#%%


# 已经有数据和特征，只重跑方法和指标
# result = run_experiment(
#     config_tag="sim64",
#     apply_repair=True,
#     do_generate=False,
#     do_features=False,
#     do_methods=True,
#     do_metrics=True,
#     plot_subjects=True,
# )


#%%
# df_metrics_mean = (
#     result["df_metrics"]
#     .groupby("method", as_index=False)[
#         ["snr_like", "signal_peak", "baseline_rms", "corr_to_reference", "rmse_to_reference", "peak_latency"]
#     ]
#     .mean(numeric_only=True)
# )

# print("30组数据各方法指标平均值如下：")
# for _, row in df_metrics_mean.iterrows():
#     print(f"\n方法：{row['method']}")
#     print(f"  类SNR平均值：{row['snr_like']:.6g}")
#     print(f"  信号峰值平均值：{row['signal_peak']:.6g}")
#     print(f"  基线RMS平均值：{row['baseline_rms']:.6g}")
#     print(f"  与参考相关系数平均值：{row['corr_to_reference']:.6g}")
#     print(f"  与参考RMSE平均值：{row['rmse_to_reference']:.6g}")
#     print(f"  峰值潜伏期平均值：{row['peak_latency']:.6g}")
# #%%
