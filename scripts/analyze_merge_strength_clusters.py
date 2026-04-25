import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGE_DIR = PROJECT_ROOT / "data_clean" / "merge_analysis"
DATA_DICT = PROJECT_ROOT / "data_dict"

INPUT_PATH = MERGE_DIR / "point_onset_merge_rates.csv"
COORD_PATH = DATA_DICT / "point_coords_master.csv"
POINT_OUTPUT = MERGE_DIR / "point_merge_strength_summary.csv"
CLUSTER_OUTPUT = MERGE_DIR / "point_merge_strength_clusters.csv"
REPORT_OUTPUT = MERGE_DIR / "merge_strength_cluster_report.txt"

MERGE_COLS = ["merge_S0_S1", "merge_S1_S2", "merge_S2_S3"]
FEATURE_COLS = ["avg_S0_S1", "avg_S1_S2", "avg_S2_S3"]


def classify_dominant_stage(row: pd.Series) -> str:
    values = row[FEATURE_COLS].astype(float)
    if values.max() < 0.12:
        return "overall_low"
    if values["avg_S2_S3"] >= values["avg_S1_S2"] and values["avg_S2_S3"] >= values["avg_S0_S1"]:
        if values["avg_S0_S1"] >= 0.20:
            return "S2_S3_plus_S0_S1"
        return "S2_S3_dominant"
    if values["avg_S1_S2"] >= values["avg_S0_S1"]:
        return "S1_S2_dominant"
    return "S0_S1_dominant"


def initialize_centers(x: np.ndarray, k: int) -> np.ndarray:
    """Deterministic farthest-point initialization."""
    first_idx = int(np.argmin(x.sum(axis=1)))
    centers = [x[first_idx]]
    while len(centers) < k:
        existing = np.vstack(centers)
        distances = ((x[:, None, :] - existing[None, :, :]) ** 2).sum(axis=2)
        min_distances = distances.min(axis=1)
        centers.append(x[int(np.argmax(min_distances))])
    return np.vstack(centers)


def kmeans(x: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    centers = initialize_centers(x, k)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = x[mask].mean(axis=0)
    return labels


def label_clusters(df: pd.DataFrame, cluster_col: str) -> dict[int, str]:
    labels = {}
    centers = df.groupby(cluster_col)[FEATURE_COLS].mean()
    for cluster_id, row in centers.iterrows():
        labels[int(cluster_id)] = classify_dominant_stage(row)
    return labels


def run_analysis() -> None:
    rates = pd.read_csv(INPUT_PATH)
    rates.columns = [col.strip().lstrip("\ufeff") for col in rates.columns]
    for col in ["point_id", "point_name", "onset_class"]:
        rates[col] = rates[col].astype("string").str.strip()
    for col in MERGE_COLS:
        rates[col] = pd.to_numeric(rates[col], errors="coerce")

    summary = (
        rates.groupby(["point_id", "point_name"], dropna=False)
        .agg(
            avg_S0_S1=("merge_S0_S1", "mean"),
            avg_S1_S2=("merge_S1_S2", "mean"),
            avg_S2_S3=("merge_S2_S3", "mean"),
            valid_S0_S1_cells=("merge_S0_S1", "count"),
            valid_S1_S2_cells=("merge_S1_S2", "count"),
            valid_S2_S3_cells=("merge_S2_S3", "count"),
        )
        .reset_index()
    )

    coords = pd.read_csv(COORD_PATH)
    coords.columns = [col.strip().lstrip("\ufeff") for col in coords.columns]
    coords["point_name"] = coords["point_name"].astype("string").str.strip()
    coords = coords[["point_name", "subbranch", "lat", "lon"]]
    summary = summary.merge(coords, on="point_name", how="left")

    summary["dominant_stage_type"] = summary.apply(classify_dominant_stage, axis=1)
    summary = summary[
        [
            "point_id",
            "point_name",
            "subbranch",
            "lat",
            "lon",
            *FEATURE_COLS,
            "valid_S0_S1_cells",
            "valid_S1_S2_cells",
            "valid_S2_S3_cells",
            "dominant_stage_type",
        ]
    ].sort_values(["subbranch", "point_id", "point_name"])
    summary.to_csv(POINT_OUTPUT, index=False, encoding="utf-8-sig")

    cluster_df = summary.copy()
    x = cluster_df[FEATURE_COLS].to_numpy(dtype=float)
    for k in [3, 4, 5]:
        col = f"kmeans_k{k}"
        cluster_df[col] = kmeans(x, k) + 1
        label_map = label_clusters(cluster_df, col)
        cluster_df[f"{col}_label"] = cluster_df[col].map(label_map)
    cluster_df.to_csv(CLUSTER_OUTPUT, index=False, encoding="utf-8-sig")

    report_lines = [
        "合并强度与聚类结果说明",
        "=" * 40,
        "",
        "指标口径：",
        "avg_S0_S1 = 每个方言点内所有有效声母条件的 S0-S1 merge_rate 平均值",
        "avg_S1_S2 = 每个方言点内所有有效声母条件的 S1-S2 merge_rate 平均值",
        "avg_S2_S3 = 每个方言点内所有有效声母条件的 S2-S3 merge_rate 平均值",
        "N/T 在低位链段为无效格位，平均时自动按 NaN 跳过。",
        "",
        "按最大链段强度的规则分类：",
    ]
    for label, group in summary.groupby("dominant_stage_type"):
        points = "、".join(group["point_name"].astype(str))
        report_lines.append(f"- {label}: {len(group)} 点；{points}")

    report_lines.extend(["", "k-means 聚类摘要："])
    for k in [3, 4, 5]:
        col = f"kmeans_k{k}"
        report_lines.append("")
        report_lines.append(f"k = {k}")
        for cluster_id, group in cluster_df.groupby(col):
            center = group[FEATURE_COLS].mean()
            label = group[f"{col}_label"].iloc[0]
            subbranch_counts = " | ".join(
                f"{idx}({val})" for idx, val in group["subbranch"].value_counts().items()
            )
            report_lines.append(
                f"  C{cluster_id} {label}: {len(group)} 点；"
                f"中心=({center['avg_S0_S1']:.3f}, {center['avg_S1_S2']:.3f}, {center['avg_S2_S3']:.3f})；"
                f"{subbranch_counts}"
            )

    REPORT_OUTPUT.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"已生成平均合并强度表：{POINT_OUTPUT}")
    print(f"已生成聚类结果表：{CLUSTER_OUTPUT}")
    print(f"已生成聚类说明：{REPORT_OUTPUT}")


if __name__ == "__main__":
    run_analysis()
