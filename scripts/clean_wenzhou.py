import pandas as pd
import re
from pathlib import Path

# === 路径设置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_CLEAN = PROJECT_ROOT / "data_clean"
DATA_DICT = PROJECT_ROOT / "data_dict"

# === 读入原始数据（每个方言点一个 CSV） ===
# 约定原始文件路径为 data_raw/points/*.csv
raw_dir = DATA_RAW / "points"
raw_files = sorted(raw_dir.glob("*.csv"))
if raw_files:
    df_list = [pd.read_csv(p) for p in raw_files]
    df = pd.concat(df_list, ignore_index=True)
else:
    # 兼容旧的单文件输入
    raw_path = DATA_RAW / "wuyu_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"未找到原始数据：请放在 {raw_dir}/*.csv 或 {raw_path}"
        )
    df = pd.read_csv(raw_path)

# 过滤掉不是数据的行：只保留“韵”和“声组”都有值的
required_base_cols = ["韵", "声组", "汉字", "读音"]
missing_base_cols = [c for c in required_base_cols if c not in df.columns]
if missing_base_cols:
    raise ValueError(f"缺少基础列：{missing_base_cols}，请确认原始数据列名")
df = df[df["韵"].notna() & df["声组"].notna()]
df = df[df["韵"] != "韵"]   # 防止表头之类被重复读进来

# === 方言点基础信息（来自原始数据列） ===
required_point_cols = ["point_id", "point_name", "subbranch", "lat", "lon"]
missing_point_cols = [c for c in required_point_cols if c not in df.columns]
if missing_point_cols:
    raise ValueError(f"缺少方言点列：{missing_point_cols}，请在原始数据中提供")

# === 读入韵部→slot mapping ===
rhyme_map = pd.read_csv(DATA_DICT / "rhyme_slot_mapping.csv")

df = df.merge(
    rhyme_map,
    left_on="韵",
    right_on="rhyme",
    how="left"
)

# === onset_class 映射（现在你的“声组”本身就是标准值，可以直接用） ===
# 如果以后遇到更复杂的 raw_onset，可以再用 onset_mapping.csv 做 merge
df["onset_class"] = df["声组"]

# === 重命名例字 & 读音列 ===
# 单文件多方言点模式下，统一使用“读音”列
if "读音" in df.columns:
    df = df.rename(columns={"汉字": "char", "读音": "phonetic"})
else:
    raise ValueError("没找到读音列，请确认列名为 '读音'")

# === 提取元音符号（粗略版） ===
VOWEL_PATTERN = r"[aeiouAEIOUɤɔøœəɯɐʌyɨ]+"

def extract_vowel_symbol(ipa):
    if pd.isna(ipa):
        return None
    m = re.search(VOWEL_PATTERN, str(ipa))
    if m:
        return m.group(0)
    return None

df["vowel_symbol"] = df["phonetic"].apply(extract_vowel_symbol)
df["vowel_class"] = df["vowel_symbol"]   # 先简单等同，以后细分

# === 主体层 & 缺项标记 ===
df["reading_layer"] = "main"
df["mainlayer_flag"] = 1
df["combo_validity"] = 1
df["zero_type"] = "none"
df["feature_change"] = None   # 之后我们写规则来填

# === 整理列顺序 ===
cols_order = [
    "point_id", "point_name", "subbranch", "lat", "lon",
    "rhyme_modern", "chain_slot", "slot_type_initial", "is_free",
    "onset_class",
    "char", "phonetic", "vowel_symbol", "vowel_class",
    "reading_layer", "mainlayer_flag",
    "combo_validity", "zero_type",
    "feature_change"
]

# 你 merge 之后有这些列：rhyme, chain_slot, slot_type_initial, is_free
df["rhyme_modern"] = df["韵"]

# 确保列存在
cols_order = [c for c in cols_order if c in df.columns]
df_clean = df[cols_order].copy()

# === 导出 ===
DATA_CLEAN.mkdir(exist_ok=True)
out_path = DATA_CLEAN / "wuyu_lexeme.csv"
df_clean.to_csv(out_path, index=False, encoding="utf-8")

print("✅ 温州清洗完成：", out_path)
print(df_clean.head())
