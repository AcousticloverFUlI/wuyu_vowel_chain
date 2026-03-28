import pandas as pd
from pathlib import Path

# === 路径设置 ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CLEAN = PROJECT_ROOT / "data_clean"
# 按照你的要求，放在 data_clean/value_type 文件夹下
OUTPUT_DIR = DATA_CLEAN / "value_type" 

# === 读取清洗后的数据 ===
input_path = DATA_CLEAN / "wuyu_lexeme.csv"
if not input_path.exists():
    raise FileNotFoundError(f"未找到清洗后的数据文件：{input_path}，请先运行清洗脚本。")

df = pd.read_csv(input_path)

# === 核心参数设置 ===
target_rhymes = ["歌", "戈", "麻", "模", "佳", "皆"]
CHAR_THRESHOLD = 4

# === 数据过滤 ===
df_filtered = df[df["rhyme_modern"].isin(target_rhymes)].copy()

# === 1. 分韵统计读音分布 (原有逻辑) ===
summary = df_filtered.groupby(['point_id', 'point_name', 'rhyme_modern', 'phonetic']).agg(
    char_count=('char', 'nunique'),
    char_list=('char', lambda x: list(set(x.dropna())))
).reset_index()

def format_char_info(row):
    if row['char_count'] <= CHAR_THRESHOLD:
        return " / ".join(row['char_list'])
    return f"共 {row['char_count']} 个例字"

summary['example_chars'] = summary.apply(format_char_info, axis=1)
dist_report = summary.drop(columns=['char_list'])

# === 2. 新增部分：统计每个方言点的读音集合 (Point Inventory) ===
# 提取每个点所有不重复的读音，并排序
inventory = df_filtered.groupby(['point_id', 'point_name'])['phonetic'].apply(
    lambda x: sorted(list(set(x.dropna())))
).reset_index()

# 格式化集合显示
inventory['phonetic_set'] = inventory['phonetic'].apply(lambda x: ", ".join(x))
# 统计音值总数（用于衡量系统复杂度）
inventory['vowel_inventory_size'] = inventory['phonetic'].apply(len)

# 移除原始列表列，保留格式化后的列
inventory_report = inventory.drop(columns=['phonetic'])

# === 导出结果 ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 导出分韵分布表
dist_path = OUTPUT_DIR / "rhyme_phonetic_distribution.csv"
dist_report.to_csv(dist_path, index=False, encoding="utf-8-sig")

# 导出方言点库藏集合表
inv_path = OUTPUT_DIR / "point_phonetic_inventory.csv"
inventory_report.to_csv(inv_path, index=False, encoding="utf-8-sig")

print(f"✅ 统计分析完成！")
print(f"1. 分韵读音分布已生成：{dist_path}")
print(f"2. 方言点读音库藏集合已生成：{inv_path}")

# === 控制台预览：展示库藏集合 ===
print("\n--- 每个方言点的读音集合预览 ---")
print(inventory_report[['point_name', 'vowel_inventory_size', 'phonetic_set']].head(10))