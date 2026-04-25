# 按照声母排列的声母库藏
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT_DIR = DATA_CLEAN / "value_type"

INPUT_PATH = DATA_CLEAN / "wuyu_lexeme.csv"
OUTPUT_PATH = OUTPUT_DIR / "point_slot_onset_phonetic_counts.csv"

SLOT_ORDER = ["S0", "S1", "S2", "S3"]
REQUIRED_COLUMNS = {"point_id", "point_name", "onset_class", "chain_slot", "phonetic"}


def split_phonetic(value) -> list[str]:
    if pd.isna(value):
        return []
    value = str(value).replace("\u00a0", " ").strip()
    if not value or value.lower() == "nan":
        return []
    if value == "o（uo）":
        return ["u", "uo"]
    return [part.strip() for part in value.split("/") if part.strip()]


def format_counts(group: pd.DataFrame) -> str:
    group = group.sort_values(["count", "phonetic"], ascending=[False, True])
    return ", ".join(
        f"{row.phonetic}({int(row.count)})" for row in group.itertuples(index=False)
    )


def run_analysis() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件：{INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列：{sorted(missing)}")

    work = df[list(REQUIRED_COLUMNS)].copy()
    for col in REQUIRED_COLUMNS:
        work[col] = work[col].astype("string").str.strip()
    work["onset_class"] = work["onset_class"].replace({"TS*": "TS"})
    work["phonetic"] = work["phonetic"].apply(split_phonetic)
    work = work.explode("phonetic")

    work = work[
        work["phonetic"].notna()
        & (work["phonetic"] != "")
        & (work["phonetic"].str.lower() != "nan")
        & work["chain_slot"].isin(SLOT_ORDER)
    ].copy()
    work = work[
        ~(
            work["onset_class"].isin(["T", "N"])
            & work["chain_slot"].isin(["S0", "S1"])
        )
    ].copy()

    counts = (
        work.groupby(
            ["point_id", "point_name", "onset_class", "chain_slot", "phonetic"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    formatted = (
        counts.groupby(
            ["point_id", "point_name", "onset_class", "chain_slot"],
            dropna=False,
        )
        .apply(format_counts, include_groups=False)
        .reset_index(name="phonetic_counts")
    )

    wide = formatted.pivot_table(
        index=["point_id", "point_name", "onset_class"],
        columns="chain_slot",
        values="phonetic_counts",
        aggfunc="first",
        fill_value="",
    ).reset_index()

    for slot in SLOT_ORDER:
        if slot not in wide.columns:
            wide[slot] = ""

    wide = wide[["point_id", "point_name", "onset_class", *SLOT_ORDER]]
    wide = wide.sort_values(["point_id", "onset_class", "point_name"]).reset_index(
        drop=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"已生成统计表：{OUTPUT_PATH}")
    print(f"统计记录数：{len(wide)}")


if __name__ == "__main__":
    run_analysis()
