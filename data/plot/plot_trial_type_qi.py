#%%
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

# 路径与绘图配置
FEATURE_DIR = Path("result/features_sim")
DATA_DIR = Path("data_sim")
OUTPUT_DIR = Path("result/feature_figs_sim")
TRIAL_TYPE_ORDER = ["Good", "Borderline", "BadGross"]
MAX_SCATTER_POINTS_PER_GROUP = 400
TRIAL_TYPE_COLORS = {
    "Good": "#48D215",
    "Borderline": "#0475C6",
    "BadGross": "#ef3d3d",
}

# ---------------------- 核心工具函数 ----------------------
def find_feature_csv_files(feature_dir: Path) -> list[Path]:
    """查找特征CSV文件（排除ALL_开头的汇总文件）"""
    csv_files = sorted(feature_dir.glob("*_trial_features.csv"))
    return [path for path in csv_files if not path.name.startswith("ALL_")]

def infer_epochs_path(feature_csv: Path, data_dir: Path) -> Path:
    """根据特征文件推导epochs文件路径"""
    dataset_name = feature_csv.stem.removesuffix("_trial_features")
    epochs_name = dataset_name.replace("raw_sim_", "epochs_sim_", 1) + "-epo.fif"
    return data_dir / epochs_name

def infer_trial_metadata_path(feature_csv: Path, data_dir: Path) -> Path:
    """根据特征文件推导trial元数据文件路径"""
    dataset_name = feature_csv.stem.removesuffix("_trial_features")
    meta_name = f"{dataset_name}__trial_metadata.csv"
    return data_dir / meta_name

def load_trial_info(feature_csv: Path, data_dir: Path) -> pd.DataFrame:
    """
    加载完整的trial元数据（包含Trial_Type、DetectedByRepairbads等关键列）
    优先加载trial_metadata.csv → 其次加载epochs.metadata → 无则报错
    """
    # 1. 优先读取独立的trial元数据文件（包含DetectedByRepairbads）
    try:
        meta_path = infer_trial_metadata_path(feature_csv, data_dir)
        trial_info = pd.read_csv(meta_path)
        # 必须包含的核心列
        required_cols = ["Trial_Type", "DetectedByRepairbads"]
        if not all(col in trial_info.columns for col in required_cols):
            raise KeyError(f"元数据缺少必要列！需要：{required_cols}，文件：{meta_path}")
        trial_info = trial_info.reset_index(drop=True)
        trial_info["Trial"] = np.arange(len(trial_info))
        return trial_info[["Trial", "Trial_Type", "DetectedByRepairbads"]]
    except FileNotFoundError:
        pass

    # 2. 读取epochs文件中的元数据
    epochs_path = infer_epochs_path(feature_csv, data_dir)
    if not epochs_path.exists():
        raise FileNotFoundError(f"未找到元数据文件和epochs文件！")
    
    epochs = mne.read_epochs(epochs_path, preload=False, verbose=False)
    if epochs.metadata is None or "DetectedByRepairbads" not in epochs.metadata.columns:
        raise KeyError(f"epochs元数据缺少Trial_Type/DetectedByRepairbads列：{epochs_path}")
    
    trial_info = epochs.metadata.reset_index(drop=True)
    trial_info["Trial"] = np.arange(len(trial_info))
    return trial_info[["Trial", "Trial_Type", "DetectedByRepairbads"]]

def load_features_with_trial_type(feature_csv: Path, data_dir: Path) -> pd.DataFrame:
    """加载特征文件，并匹配完整的标签+检测结果"""
    df = pd.read_csv(feature_csv)
    required_cols = ["Trial", "w_i", "q_i"]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"特征文件必须包含Trial/w_i/q_i：{feature_csv}")

    trial_info = load_trial_info(feature_csv, data_dir)
    merged = df.merge(trial_info, on="Trial", how="left", validate="one_to_one")
    if merged["Trial_Type"].isna().any() or merged["DetectedByRepairbads"].isna().any():
        raise ValueError(f"部分trial未匹配到标签/检测结果：{feature_csv}")

    merged["dataset_name"] = feature_csv.stem.removesuffix("_trial_features")
    return merged

def load_all_trials(feature_dir: Path, data_dir: Path) -> pd.DataFrame:
    """加载所有特征文件并合并"""
    csv_files = find_feature_csv_files(feature_dir)
    if not csv_files:
        raise FileNotFoundError(f"未找到特征文件：{feature_dir}")

    tables = [load_features_with_trial_type(csv, data_dir) for csv in csv_files]
    return pd.concat(tables, ignore_index=True)

def filter_valid_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    筛选有效试次：标签与坏段检测结果完全一致
    规则：
    - Good + 未检测到坏段 (0)
    - Borderline + 未检测到坏段 (0)
    - BadGross + 检测到坏段 (1)
    """
    # 构建筛选条件
    condition_good = (df["Trial_Type"] == "Good") & (df["DetectedByRepairbads"] == 0)
    condition_borderline = (df["Trial_Type"] == "Borderline") & (df["DetectedByRepairbads"] == 0)
    condition_badgross = (df["Trial_Type"] == "BadGross") & (df["DetectedByRepairbads"] == 1)
    
    # 合并所有有效条件
    valid_mask = condition_good | condition_borderline | condition_badgross
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    
    return df_valid

# ---------------------- 绘图函数 ----------------------
def plot_trial_type_distribution(
    df: pd.DataFrame,
    value_col: str,  # 直接传入w_i / q_i
    ylabel: str,
    title: str,
    save_path: Path | None = None,
) -> None:
    """绘制三类标签的数值分布（小提琴图+散点+中位数）"""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    violin_data = []
    violin_positions = []
    violin_colors = []
    scatter_specs = []

    # 按指定顺序提取每类数据
    for pos, trial_type in enumerate(TRIAL_TYPE_ORDER, start=1):
        vals = df.loc[df["Trial_Type"] == trial_type, value_col].dropna().to_numpy(float)
        if len(vals) == 0:
            continue

        color = TRIAL_TYPE_COLORS[trial_type]
        violin_data.append(vals)
        violin_positions.append(pos)
        violin_colors.append(color)
        scatter_specs.append((pos, vals, color))

    # 绘制小提琴图
    if violin_data:
        vp = ax.violinplot(
            violin_data, positions=violin_positions, widths=0.72,
            showmeans=False, showmedians=False, showextrema=False
        )
        for body, color in zip(vp["bodies"], violin_colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)

    # 绘制散点图（限制最大点数）
    rng = np.random.default_rng(20260416)
    for pos, vals, color in scatter_specs:
        if len(vals) > MAX_SCATTER_POINTS_PER_GROUP:
            vals = rng.choice(vals, MAX_SCATTER_POINTS_PER_GROUP, replace=False)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        x = np.full(len(vals), pos) + jitter
        ax.scatter(x, vals, s=20, alpha=0.46, color=color, edgecolors="none", zorder=3)

    # 绘制中位数
    for vals, pos in zip(violin_data, violin_positions):
        ax.hlines(np.median(vals), pos-0.2, pos+0.2, color="#111", linewidth=2.4, zorder=5)

    # 图表样式
    ax.set_xticks(range(1, len(TRIAL_TYPE_ORDER)+1))
    ax.set_xticklabels(TRIAL_TYPE_ORDER)
    ax.set_xlabel("Trial Type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    # 保存图片
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存：{save_path}")

    plt.show()

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 1. 加载所有原始数据
    df_raw = load_all_trials(FEATURE_DIR, DATA_DIR)
    print("="*60)
    print(f"原始数据 | 数据集数量：{df_raw['dataset_name'].nunique()}")
    print(f"原始数据 | 总试次数量：{len(df_raw)}")
    
    # 2. 筛选有效试次（核心新增逻辑）
    df_valid = filter_valid_trials(df_raw)
    print(f"筛选后 | 有效试次数量：{len(df_valid)}")
    print(f"剔除无效试次数量：{len(df_raw) - len(df_valid)}")
    print("="*60)

    # 3. 统计有效试次的标签分布
    print("\n✅ 有效试次标签统计：")
    counts = df_valid["Trial_Type"].value_counts()
    for label in TRIAL_TYPE_ORDER:
        print(f"  {label}: {counts.get(label, 0)}")

    # 4. 保存路径
    save_path_w = OUTPUT_DIR / "valid_trials__trial_type_vs_w_i.png"
    save_path_q = OUTPUT_DIR / "valid_trials__trial_type_vs_q_i.png"

    # 5. 绘制筛选后的 w_i / q_i 分布图
    plot_trial_type_distribution(
        df_valid,
        value_col="w_i",
        ylabel="w_i ",
        title="Trial Type vs  w_i",
        save_path=save_path_w
    )

    plot_trial_type_distribution(
        df_valid,
        value_col="q_i",
        ylabel="q_i ",
        title="Trial Type vs q_i",
        save_path=save_path_q
    )
#%%