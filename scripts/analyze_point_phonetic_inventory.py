import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT_DIR = DATA_CLEAN / "value_type"

INPUT_PATH = DATA_CLEAN / "wuyu_lexeme.csv"
OUTPUT_PATH = OUTPUT_DIR / "point_phonetic_inventory.csv"

TARGET_RHYMES = ["歌", "戈", "麻", "模", "佳", "皆"]
EXCLUDED_CHARS = {"靴", "茄"}


def split_phonetic(value) -> list[str]:
    if pd.isna(value):
        return []
    value = str(value).replace("\u00a0", " ").strip()
    if not value or value.lower() == "nan":
        return []
    if value == "o（uo）":
        return ["u", "uo"]
    return [part.strip() for part in value.split("/") if part.strip()]


def format_inventory(values: pd.Series) -> str:
    phonetics = set()
    for value in values.dropna():
        phonetics.update(split_phonetic(value))
    return ", ".join(sorted(phonetics))


def run_analysis() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件：{INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required = {"point_id", "point_name", "rhyme_modern", "char", "phonetic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列：{sorted(missing)}")

    work = df[list(required)].copy()
    for col in required:
        work[col] = work[col].astype("string").str.strip()

    work = work[
        work["rhyme_modern"].isin(TARGET_RHYMES)
        & ~work["char"].isin(EXCLUDED_CHARS)
        & work["point_id"].notna()
        & (work["point_id"] != "")
        & work["phonetic"].notna()
        & (work["phonetic"] != "")
        & (work["phonetic"].str.lower() != "nan")
    ].copy()

    inventory = (
        work.groupby(["point_id", "point_name"], dropna=True)["phonetic"]
        .apply(format_inventory)
        .reset_index(name="phonetic_set")
    )
    inventory["vowel_inventory_size"] = inventory["phonetic_set"].apply(
        lambda value: len([part for part in value.split(", ") if part])
    )
    inventory = inventory.sort_values(["point_id", "point_name"]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"已生成音值总库藏表：{OUTPUT_PATH}")
    print(f"统计记录数：{len(inventory)}")


if __name__ == "__main__":
    run_analysis()
