import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = PROJECT_ROOT / "data_clean"
DATA_DICT = PROJECT_ROOT / "data_dict"
OUTPUT_DIR = DATA_CLEAN / "value_type"

LEXEME_PATH = DATA_CLEAN / "wuyu_lexeme.csv"
RHYME_INVENTORY_PATH = OUTPUT_DIR / "point_rhyme_inventory.csv"
RHYME_SLOT_PATH = DATA_DICT / "rhyme_slot_mapping.csv"

DETAIL_OUTPUT = OUTPUT_DIR / "point_onset_inventory_ratios.csv"
POINT_INEQUALITY_OUTPUT = OUTPUT_DIR / "point_onset_inventory_inequalities.csv"
GLOBAL_INEQUALITY_OUTPUT = OUTPUT_DIR / "global_onset_inventory_inequality.csv"

SLOT_ORDER = ["S0", "S1", "S2", "S3"]
NO_LOW_SLOT_ONSETS = {"T", "N"}


def split_phonetic(value) -> list[str]:
    if pd.isna(value):
        return []
    value = str(value).replace("\u00a0", " ").strip()
    if not value or value.lower() == "nan":
        return []
    if value == "o（uo）":
        return ["u", "uo"]
    return [part.strip() for part in value.split("/") if part.strip()]


def parse_set(value) -> set[str]:
    values = set()
    if pd.isna(value):
        return values
    for item in str(value).split(","):
        values.update(split_phonetic(item))
    return values


def valid_slots_for_onset(onset_class: str) -> list[str]:
    if onset_class in NO_LOW_SLOT_ONSETS:
        return ["S2", "S3"]
    return SLOT_ORDER


def format_ratio(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_inequality(rows: pd.DataFrame) -> str:
    rows = rows.sort_values(["ratio", "onset_class"], ascending=[False, True])
    return " > ".join(
        f"{row.onset_class}({format_ratio(row.ratio)})"
        for row in rows.itertuples(index=False)
        if pd.notna(row.ratio)
    )


def format_inventory(values: set[str]) -> str:
    return ", ".join(sorted(values))


def build_point_slot_sets() -> dict[tuple[str, str], dict[str, set[str]]]:
    rhyme_df = pd.read_csv(RHYME_INVENTORY_PATH)
    slot_df = pd.read_csv(RHYME_SLOT_PATH)
    slot_df = slot_df[slot_df["chain_slot"].isin(SLOT_ORDER)].copy()

    rhyme_df = rhyme_df.merge(
        slot_df[["rhyme", "chain_slot"]],
        left_on="rhyme_modern",
        right_on="rhyme",
        how="inner",
    )

    point_slot_sets: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in rhyme_df.itertuples(index=False):
        key = (str(row.point_id).strip(), str(row.point_name).strip())
        point_slot_sets.setdefault(key, {slot: set() for slot in SLOT_ORDER})
        point_slot_sets[key][row.chain_slot].update(parse_set(row.rhyme_phonetic_set))

    return point_slot_sets


def build_onset_slot_sets() -> dict[tuple[str, str, str], dict[str, set[str]]]:
    lexeme_df = pd.read_csv(LEXEME_PATH)
    required = {"point_id", "point_name", "onset_class", "chain_slot", "phonetic"}
    missing = required - set(lexeme_df.columns)
    if missing:
        raise ValueError(f"wuyu_lexeme.csv 缺少必要列：{sorted(missing)}")

    work = lexeme_df[list(required)].copy()
    for col in required:
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
            work["onset_class"].isin(NO_LOW_SLOT_ONSETS)
            & work["chain_slot"].isin(["S0", "S1"])
        )
    ].copy()

    onset_slot_sets: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for row in work.itertuples(index=False):
        key = (row.point_id, row.point_name, row.onset_class)
        onset_slot_sets.setdefault(key, {slot: set() for slot in SLOT_ORDER})
        onset_slot_sets[key][row.chain_slot].add(row.phonetic)

    return onset_slot_sets


def calculate_point_ratios(
    point_slot_sets: dict[tuple[str, str], dict[str, set[str]]],
    onset_slot_sets: dict[tuple[str, str, str], dict[str, set[str]]],
) -> pd.DataFrame:
    rows = []
    for (point_id, point_name, onset_class), onset_sets in onset_slot_sets.items():
        point_sets = point_slot_sets.get((point_id, point_name))
        if not point_sets:
            continue

        valid_slots = valid_slots_for_onset(onset_class)
        numerator = sum(len(onset_sets[slot]) for slot in valid_slots)
        denominator = sum(len(point_sets[slot]) for slot in valid_slots)
        ratio = numerator / denominator if denominator else pd.NA

        row = {
            "point_id": point_id,
            "point_name": point_name,
            "onset_class": onset_class,
            "valid_slots": "+".join(valid_slots),
            "onset_inventory_size": numerator,
            "slot_inventory_size": denominator,
            "ratio": ratio,
        }
        for slot in SLOT_ORDER:
            if slot not in valid_slots:
                row[f"{slot}_onset_size"] = pd.NA
                row[f"{slot}_slot_size"] = pd.NA
                row[f"{slot}_ratio"] = pd.NA
                continue

            slot_denominator = len(point_sets[slot])
            slot_numerator = len(onset_sets[slot])
            row[f"{slot}_onset_size"] = slot_numerator
            row[f"{slot}_slot_size"] = slot_denominator
            row[f"{slot}_ratio"] = (
                slot_numerator / slot_denominator if slot_denominator else pd.NA
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["point_id", "onset_class", "point_name"])


def calculate_global_ratios(
    point_slot_sets: dict[tuple[str, str], dict[str, set[str]]],
    onset_slot_sets: dict[tuple[str, str, str], dict[str, set[str]]],
) -> pd.DataFrame:
    global_slot_sets = {slot: set() for slot in SLOT_ORDER}
    for slot_sets in point_slot_sets.values():
        for slot in SLOT_ORDER:
            global_slot_sets[slot].update(slot_sets[slot])

    global_onset_sets: dict[str, dict[str, set[str]]] = {}
    for (_, _, onset_class), slot_sets in onset_slot_sets.items():
        global_onset_sets.setdefault(onset_class, {slot: set() for slot in SLOT_ORDER})
        for slot in SLOT_ORDER:
            global_onset_sets[onset_class][slot].update(slot_sets[slot])

    rows = []
    for onset_class, onset_sets in global_onset_sets.items():
        valid_slots = valid_slots_for_onset(onset_class)
        numerator_values_by_slot = {
            slot: onset_sets[slot] for slot in valid_slots
        }
        denominator_values_by_slot = {
            slot: global_slot_sets[slot] for slot in valid_slots
        }
        numerator = sum(len(values) for values in numerator_values_by_slot.values())
        denominator = sum(len(values) for values in denominator_values_by_slot.values())
        rows.append(
            {
                "scope": "all_points",
                "onset_class": onset_class,
                "valid_slots": "+".join(valid_slots),
                "onset_inventory_size": numerator,
                "onset_inventory_values": " | ".join(
                    f"{slot}: {format_inventory(values)}"
                    for slot, values in numerator_values_by_slot.items()
                ),
                "slot_inventory_size": denominator,
                "slot_inventory_values": " | ".join(
                    f"{slot}: {format_inventory(values)}"
                    for slot, values in denominator_values_by_slot.items()
                ),
                "ratio": numerator / denominator if denominator else pd.NA,
            }
        )

    return pd.DataFrame(rows).sort_values(["ratio", "onset_class"], ascending=[False, True])


def run_analysis() -> None:
    point_slot_sets = build_point_slot_sets()
    onset_slot_sets = build_onset_slot_sets()

    detail_df = calculate_point_ratios(point_slot_sets, onset_slot_sets)
    point_ineq_df = (
        detail_df.groupby(["point_id", "point_name"], dropna=False)
        .apply(format_inequality, include_groups=False)
        .reset_index(name="onset_inequality")
        .sort_values(["point_id", "point_name"])
    )

    global_df = calculate_global_ratios(point_slot_sets, onset_slot_sets)
    global_inequality = format_inequality(global_df)
    global_df["global_onset_inequality"] = global_inequality

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(DETAIL_OUTPUT, index=False, encoding="utf-8-sig")
    point_ineq_df.to_csv(POINT_INEQUALITY_OUTPUT, index=False, encoding="utf-8-sig")
    global_df.to_csv(GLOBAL_INEQUALITY_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"已生成详细比值表：{DETAIL_OUTPUT}")
    print(f"已生成每点不等式表：{POINT_INEQUALITY_OUTPUT}")
    print(f"已生成全局不等式表：{GLOBAL_INEQUALITY_OUTPUT}")
    print(f"全局不等式：{global_inequality}")


if __name__ == "__main__":
    run_analysis()
