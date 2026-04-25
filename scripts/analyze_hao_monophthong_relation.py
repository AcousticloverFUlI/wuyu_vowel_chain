import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT_DIR = DATA_CLEAN / "value_type"

INPUT_PATH = DATA_CLEAN / "wuyu_lexeme.csv"
CORE_INVENTORY_OUTPUT = OUTPUT_DIR / "point_monophthong_inventory.csv"
RELATION_OUTPUT = OUTPUT_DIR / "point_hao_monophthong_relation.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "hao_monophthong_relation_summary.csv"
VALUE_SUMMARY_OUTPUT = OUTPUT_DIR / "hao_value_summary.csv"

CORE_SLOTS = ["S0", "S1", "S2", "S3"]
TARGET_RHYME = "豪"
EXCLUDED_CHARS = {"靴", "茄"}

# Working IPA vowel inventory for deciding whether an item contains one vowel nucleus.
# Glides such as w/ɥ are treated as complex offglides rather than monophthongs.
VOWEL_CHARS = set("aeiouyæøœɐɑɒɔəɛɜɞɤɯɪʊʌʏɵʉɷᴀᴇ")
GLIDE_CHARS = set("wɥj")
BACK_NUCLEI = {"u", "o", "ɔ", "ɒ", "ɑ", "ɤ", "ɯ", "ʊ", "ʌ", "ɷ"}


def split_phonetic(value) -> list[str]:
    if pd.isna(value):
        return []
    value = str(value).replace("\u00a0", " ").strip()
    if not value or value.lower() == "nan":
        return []
    if value == "o（uo）":
        return ["u", "uo"]
    return [part.strip() for part in value.split("/") if part.strip()]


def decomposed_chars(value: str) -> list[str]:
    return [
        char
        for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    ]


def vowel_sequence(value: str) -> list[str]:
    return [char for char in decomposed_chars(value) if char in VOWEL_CHARS]


def has_glide(value: str) -> bool:
    return any(char in GLIDE_CHARS for char in decomposed_chars(value))


def classify_value(value: str) -> str:
    if "ʔ" in value:
        return "checked_or_coda"
    vowels = vowel_sequence(value)
    if len(vowels) == 1 and not has_glide(value):
        return "monophthong"
    if len(vowels) > 1 or (len(vowels) == 1 and has_glide(value)):
        return "diphthong_or_glide"
    return "other"


def monophthong_nucleus(value: str) -> str:
    if classify_value(value) != "monophthong":
        return ""
    vowels = vowel_sequence(value)
    return vowels[0] if vowels else ""


def format_values(values) -> str:
    return ", ".join(sorted(value for value in values if value))


def format_counts(counts: pd.Series) -> str:
    return ", ".join(f"{value}({int(count)})" for value, count in counts.items())


def expand_phonetics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["phonetic"] = work["phonetic"].apply(split_phonetic)
    work = work.explode("phonetic")
    work = work[work["phonetic"].notna() & (work["phonetic"] != "")].copy()
    work["value_class"] = work["phonetic"].apply(classify_value)
    work["nucleus"] = work["phonetic"].apply(monophthong_nucleus)
    work["is_back_nucleus"] = work["nucleus"].isin(BACK_NUCLEI)
    return work


def summarize_core_inventory(core: pd.DataFrame) -> pd.DataFrame:
    mono = core[core["value_class"] == "monophthong"].copy()

    rows = []
    for keys, group in mono.groupby(["point_id", "point_name", "subbranch"], dropna=False):
        point_id, point_name, subbranch = keys
        values = set(group["phonetic"])
        nuclei = set(group["nucleus"])
        back_values = set(group.loc[group["is_back_nucleus"], "phonetic"])
        back_nuclei = set(group.loc[group["is_back_nucleus"], "nucleus"])
        nucleus_counts = group["nucleus"].value_counts().sort_values(ascending=False)
        back_nucleus_counts = (
            group.loc[group["is_back_nucleus"], "nucleus"]
            .value_counts()
            .sort_values(ascending=False)
        )
        rows.append(
            {
                "point_id": point_id,
                "point_name": point_name,
                "subbranch": subbranch,
                "core_slots": "+".join(CORE_SLOTS),
                "monophthong_values": format_values(values),
                "monophthong_value_size": len(values),
                "monophthong_token_count": len(group),
                "monophthong_nuclei": format_values(nuclei),
                "monophthong_nucleus_size": len(nuclei),
                "monophthong_nucleus_counts": format_counts(nucleus_counts),
                "back_monophthong_values": format_values(back_values),
                "back_monophthong_value_size": len(back_values),
                "back_monophthong_token_count": int(group["is_back_nucleus"].sum()),
                "back_monophthong_nuclei": format_values(back_nuclei),
                "back_monophthong_nucleus_size": len(back_nuclei),
                "back_monophthong_nucleus_counts": format_counts(
                    back_nucleus_counts
                ),
            }
        )
    return pd.DataFrame(rows)


def hao_status(group: pd.DataFrame) -> str:
    classes = set(group["value_class"])
    has_mono = "monophthong" in classes
    has_complex = "diphthong_or_glide" in classes
    has_other = bool(classes - {"monophthong", "diphthong_or_glide"})
    if not classes:
        return "missing"
    if has_mono and not has_complex and not has_other:
        return "monophthong_only"
    if has_complex and not has_mono and not has_other:
        return "diphthong_or_glide_only"
    if has_mono and has_complex and not has_other:
        return "mixed_mono_and_diphthong"
    if has_other and not has_mono and not has_complex:
        return "other_only"
    return "mixed_with_other"


def summarize_hao(hao: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in hao.groupby(["point_id", "point_name", "subbranch"], dropna=False):
        point_id, point_name, subbranch = keys
        counts = group["phonetic"].value_counts().sort_values(ascending=False)
        mono = group[group["value_class"] == "monophthong"]
        complex_values = group[group["value_class"] == "diphthong_or_glide"]
        other = group[
            ~group["value_class"].isin(["monophthong", "diphthong_or_glide"])
        ]
        dominant_value = counts.index[0] if len(counts) else ""
        dominant_count = int(counts.iloc[0]) if len(counts) else 0
        total_count = len(group)
        mono_count = len(mono)
        rows.append(
            {
                "point_id": point_id,
                "point_name": point_name,
                "subbranch": subbranch,
                "hao_slot_in_data": format_values(set(group["chain_slot"])),
                "hao_value_counts": format_counts(counts),
                "hao_total_token_count": total_count,
                "hao_monophthong_token_count": mono_count,
                "hao_monophthong_token_share": mono_count / total_count
                if total_count
                else 0,
                "hao_dominant_value": dominant_value,
                "hao_dominant_value_class": classify_value(dominant_value)
                if dominant_value
                else "",
                "hao_dominant_token_count": dominant_count,
                "hao_dominant_token_share": dominant_count / total_count
                if total_count
                else 0,
                "hao_status": hao_status(group),
                "hao_monophthong_values": format_values(set(mono["phonetic"])),
                "hao_monophthong_nuclei": format_values(set(mono["nucleus"])),
                "hao_back_monophthong_nuclei": format_values(
                    set(mono.loc[mono["is_back_nucleus"], "nucleus"])
                ),
                "hao_diphthong_or_glide_values": format_values(
                    set(complex_values["phonetic"])
                ),
                "hao_other_values": format_values(set(other["phonetic"])),
            }
        )
    return pd.DataFrame(rows)


def relation_label(hao_nuclei: set[str], core_nuclei: set[str], core_back: set[str]) -> str:
    if not hao_nuclei:
        return "no_hao_monophthong"
    if hao_nuclei <= core_back:
        return "all_hao_mono_in_core_back_inventory"
    if hao_nuclei <= core_nuclei:
        return "all_hao_mono_in_core_inventory"
    if hao_nuclei & core_nuclei:
        return "partial_hao_mono_in_core_inventory"
    return "hao_mono_outside_core_inventory"


def split_set(value: str) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def build_relation(core_summary: pd.DataFrame, hao_summary: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    relation = points.merge(
        core_summary,
        on=["point_id", "point_name", "subbranch"],
        how="left",
    ).merge(
        hao_summary,
        on=["point_id", "point_name", "subbranch"],
        how="left",
    )

    fill_empty = [
        "monophthong_values",
        "monophthong_nuclei",
        "monophthong_nucleus_counts",
        "back_monophthong_values",
        "back_monophthong_nuclei",
        "back_monophthong_nucleus_counts",
        "hao_slot_in_data",
        "hao_value_counts",
        "hao_status",
        "hao_dominant_value",
        "hao_dominant_value_class",
        "hao_monophthong_values",
        "hao_monophthong_nuclei",
        "hao_back_monophthong_nuclei",
        "hao_diphthong_or_glide_values",
        "hao_other_values",
    ]
    for column in fill_empty:
        if column in relation.columns:
            relation[column] = relation[column].fillna("")

    for column in [
        "monophthong_value_size",
        "monophthong_token_count",
        "monophthong_nucleus_size",
        "back_monophthong_value_size",
        "back_monophthong_token_count",
        "back_monophthong_nucleus_size",
        "hao_total_token_count",
        "hao_monophthong_token_count",
        "hao_dominant_token_count",
    ]:
        if column in relation.columns:
            relation[column] = relation[column].fillna(0).astype(int)

    for column in ["hao_monophthong_token_share", "hao_dominant_token_share"]:
        if column in relation.columns:
            relation[column] = relation[column].fillna(0.0)

    relation["core_slots"] = relation["core_slots"].fillna("+".join(CORE_SLOTS))

    relation["hao_monophthong_nuclei_in_core"] = ""
    relation["hao_monophthong_nuclei_in_core_back"] = ""
    relation["hao_monophthong_nuclei_not_in_core"] = ""
    relation["hao_core_relation"] = ""

    for idx, row in relation.iterrows():
        hao_nuclei = split_set(row["hao_monophthong_nuclei"])
        core_nuclei = split_set(row["monophthong_nuclei"])
        core_back = split_set(row["back_monophthong_nuclei"])
        relation.at[idx, "hao_monophthong_nuclei_in_core"] = format_values(
            hao_nuclei & core_nuclei
        )
        relation.at[idx, "hao_monophthong_nuclei_in_core_back"] = format_values(
            hao_nuclei & core_back
        )
        relation.at[idx, "hao_monophthong_nuclei_not_in_core"] = format_values(
            hao_nuclei - core_nuclei
        )
        relation.at[idx, "hao_core_relation"] = relation_label(
            hao_nuclei, core_nuclei, core_back
        )

    return relation.sort_values(["point_id", "point_name"]).reset_index(drop=True)


def build_value_summary(hao: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for value, group in hao.groupby("phonetic"):
        point_pairs = group[["point_id", "point_name"]].drop_duplicates()
        examples = [
            f"{row.point_id}:{row.point_name}" for row in point_pairs.head(12).itertuples(index=False)
        ]
        rows.append(
            {
                "hao_value": value,
                "value_class": classify_value(value),
                "nucleus": monophthong_nucleus(value),
                "point_count": len(point_pairs),
                "token_count": len(group),
                "example_points": " | ".join(examples),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["value_class", "point_count", "hao_value"], ascending=[True, False, True]
    )


def run_analysis() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件：{INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required = {
        "point_id",
        "point_name",
        "subbranch",
        "rhyme_modern",
        "chain_slot",
        "char",
        "phonetic",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列：{sorted(missing)}")

    work = df[list(required)].copy()
    for column in required:
        work[column] = work[column].astype("string").str.strip()
    work = work[
        work["point_id"].notna()
        & (work["point_id"] != "")
        & work["point_name"].notna()
        & (work["point_name"] != "")
    ].copy()

    points = (
        work[["point_id", "point_name", "subbranch"]]
        .drop_duplicates()
    )

    expanded = expand_phonetics(work)
    core = expanded[
        expanded["chain_slot"].isin(CORE_SLOTS)
        & ~expanded["char"].isin(EXCLUDED_CHARS)
    ].copy()
    hao = expanded[expanded["rhyme_modern"] == TARGET_RHYME].copy()

    core_summary = summarize_core_inventory(core)
    hao_summary = summarize_hao(hao)
    relation = build_relation(core_summary, hao_summary, points)
    status_summary = (
        relation.groupby(["hao_status", "hao_core_relation"], dropna=False)
        .size()
        .reset_index(name="point_count")
        .sort_values(["hao_status", "hao_core_relation"])
    )
    value_summary = build_value_summary(hao)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    core_summary.to_csv(CORE_INVENTORY_OUTPUT, index=False, encoding="utf-8-sig")
    relation.to_csv(RELATION_OUTPUT, index=False, encoding="utf-8-sig")
    status_summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
    value_summary.to_csv(VALUE_SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"已生成单元音库藏表：{CORE_INVENTORY_OUTPUT}")
    print(f"已生成豪韵单音化关系表：{RELATION_OUTPUT}")
    print(f"已生成关系汇总表：{SUMMARY_OUTPUT}")
    print(f"已生成豪韵音值汇总表：{VALUE_SUMMARY_OUTPUT}")
    print(f"方言点记录数：{len(relation)}")


if __name__ == "__main__":
    run_analysis()
