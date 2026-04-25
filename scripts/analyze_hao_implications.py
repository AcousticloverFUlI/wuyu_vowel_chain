from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALUE_DIR = PROJECT_ROOT / "data_clean" / "value_type"

INPUT_PATH = VALUE_DIR / "point_hao_monophthong_relation.csv"
FEATURE_OUTPUT = VALUE_DIR / "hao_implication_point_features.csv"
RULE_OUTPUT = VALUE_DIR / "hao_typological_implication_rules.csv"


def split_set(value) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["core_nucleus_set"] = work["monophthong_nuclei"].apply(split_set)
    work["back_nucleus_set"] = work["back_monophthong_nuclei"].apply(split_set)

    low_back = {"ɑ", "ɒ"}
    open_mid_back = {"ɔ", "ɒ"}

    features = pd.DataFrame(
        {
            "point_id": work["point_id"],
            "point_name": work["point_name"],
            "subbranch": work["subbranch"],
            "hao_value_counts": work["hao_value_counts"],
            "hao_status": work["hao_status"],
            "hao_has_monophthong": work["hao_monophthong_token_count"] > 0,
            "hao_unmonophthongized": work["hao_monophthong_token_count"] == 0,
            "hao_dominant_value": work["hao_dominant_value"],
            "core_nuclei": work["monophthong_nuclei"],
            "back_nuclei": work["back_monophthong_nuclei"],
            "core_nucleus_size": work["monophthong_nucleus_size"],
            "back_nucleus_size": work["back_monophthong_nucleus_size"],
            "subbranch_is_piling": work["subbranch"] == "太湖片-毗陵小片",
            "subbranch_not_piling": work["subbranch"] != "太湖片-毗陵小片",
            "core_has_low_back_ɑ_or_ɒ": work["core_nucleus_set"].apply(
                lambda values: bool(values & low_back)
            ),
            "core_lacks_low_back_ɑ_or_ɒ": work["core_nucleus_set"].apply(
                lambda values: not bool(values & low_back)
            ),
            "core_has_open_mid_back_ɔ_or_ɒ": work["core_nucleus_set"].apply(
                lambda values: bool(values & open_mid_back)
            ),
            "core_lacks_open_mid_back_ɔ_or_ɒ": work["core_nucleus_set"].apply(
                lambda values: not bool(values & open_mid_back)
            ),
            "core_has_ɑ": work["core_nucleus_set"].apply(lambda values: "ɑ" in values),
            "core_has_ɒ": work["core_nucleus_set"].apply(lambda values: "ɒ" in values),
            "core_has_ɔ": work["core_nucleus_set"].apply(lambda values: "ɔ" in values),
            "core_has_o_and_u": work["core_nucleus_set"].apply(
                lambda values: {"o", "u"} <= values
            ),
            "core_nucleus_size_ge_6": work["monophthong_nucleus_size"] >= 6,
            "core_nucleus_size_le_5": work["monophthong_nucleus_size"] <= 5,
            "back_nucleus_size_ge_4": work["back_monophthong_nucleus_size"] >= 4,
            "back_nucleus_size_le_3": work["back_monophthong_nucleus_size"] <= 3,
        }
    )
    features["piling_and_lacks_low_back"] = (
        features["subbranch_is_piling"] & features["core_lacks_low_back_ɑ_or_ɒ"]
    )
    features["piling_and_has_o_u_and_lacks_low_back"] = (
        features["subbranch_is_piling"]
        & features["core_has_o_and_u"]
        & features["core_lacks_low_back_ɑ_or_ɒ"]
    )
    return features


def format_points(df: pd.DataFrame) -> str:
    return " | ".join(
        f"{row.point_id}:{row.point_name}({row.hao_value_counts})"
        for row in df.itertuples(index=False)
    )


def evaluate_rule(features: pd.DataFrame, condition_col: str, target_col: str) -> dict:
    mask = features[condition_col].astype(bool)
    target = features[target_col].astype(bool)
    support = int(mask.sum())
    hit = int((mask & target).sum())
    confidence = hit / support if support else 0
    exceptions = features[mask & ~target]
    examples = features[mask & target]
    return {
        "condition": condition_col,
        "target": target_col,
        "support": support,
        "hit": hit,
        "exception_count": int(len(exceptions)),
        "confidence": confidence,
        "examples": format_points(examples),
        "exceptions": format_points(exceptions),
    }


def build_rules(features: pd.DataFrame) -> pd.DataFrame:
    rule_specs = [
        (
            "core_has_low_back_ɑ_or_ɒ",
            "hao_has_monophthong",
            "S0-S3 有低后元音 ɑ/ɒ",
            "豪韵有单音化成分",
            "强蕴涵：有低后元音位置则豪韵必有单音化",
        ),
        (
            "hao_unmonophthongized",
            "core_lacks_low_back_ɑ_or_ɒ",
            "豪韵完全未单音化",
            "S0-S3 缺低后元音 ɑ/ɒ",
            "必要条件：未单音化点都缺低后元音位置",
        ),
        (
            "core_nucleus_size_ge_6",
            "hao_has_monophthong",
            "S0-S3 单元音核数 >= 6",
            "豪韵有单音化成分",
            "强蕴涵：单元音库藏较大则豪韵必有单音化",
        ),
        (
            "hao_unmonophthongized",
            "core_nucleus_size_le_5",
            "豪韵完全未单音化",
            "S0-S3 单元音核数 <= 5",
            "必要条件：未单音化点的 S0-S3 单元音库藏不大",
        ),
        (
            "back_nucleus_size_ge_4",
            "hao_has_monophthong",
            "S0-S3 后元音核数 >= 4",
            "豪韵有单音化成分",
            "强蕴涵：后元音库藏较大则豪韵必有单音化",
        ),
        (
            "hao_unmonophthongized",
            "back_nucleus_size_le_3",
            "豪韵完全未单音化",
            "S0-S3 后元音核数 <= 3",
            "必要条件：未单音化点的后元音库藏不大",
        ),
        (
            "piling_and_lacks_low_back",
            "hao_unmonophthongized",
            "毗陵小片且 S0-S3 缺低后元音 ɑ/ɒ",
            "豪韵完全未单音化",
            "局部强蕴涵：毗陵内部的未单音化条件",
        ),
        (
            "subbranch_not_piling",
            "hao_has_monophthong",
            "非毗陵小片",
            "豪韵有单音化成分",
            "地理趋势：非毗陵点几乎都有单音化",
        ),
        (
            "core_has_open_mid_back_ɔ_or_ɒ",
            "hao_has_monophthong",
            "S0-S3 有 ɔ/ɒ",
            "豪韵有单音化成分",
            "强趋势但非绝对：有 ɔ/ɒ 通常对应豪韵单音化",
        ),
    ]

    rows = []
    for condition, target, condition_label, target_label, interpretation in rule_specs:
        row = evaluate_rule(features, condition, target)
        row["condition_label"] = condition_label
        row["target_label"] = target_label
        row["interpretation"] = interpretation
        rows.append(row)

    ordered_columns = [
        "condition_label",
        "target_label",
        "support",
        "hit",
        "exception_count",
        "confidence",
        "interpretation",
        "examples",
        "exceptions",
        "condition",
        "target",
    ]
    return pd.DataFrame(rows)[ordered_columns]


def run_analysis() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件：{INPUT_PATH}")

    relation = pd.read_csv(INPUT_PATH)
    features = build_features(relation)
    rules = build_rules(features)

    features.to_csv(FEATURE_OUTPUT, index=False, encoding="utf-8-sig")
    rules.to_csv(RULE_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"已生成蕴涵特征表：{FEATURE_OUTPUT}")
    print(f"已生成类型学蕴涵规则表：{RULE_OUTPUT}")


if __name__ == "__main__":
    run_analysis()
