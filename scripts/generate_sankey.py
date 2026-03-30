import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# === 1. 路径与参数设置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
FIGS_DIR = PROJECT_ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# 颜色配置 (RGBA)
COLOR_SPLIT = "rgba(189, 195, 199, 0.4)"   # 分立型：中性灰
COLOR_MERGE = "rgba(52, 152, 219, 0.5)"    # 合流型：明亮蓝
COLOR_LEAP = "rgba(230, 126, 34, 0.8)"     # 越级：高饱和橙色
COLOR_NT = "rgba(155, 89, 182, 0.5)"       # N/T组：深紫色

# === 2. 加载与数据对齐 ===
input_path = DATA_RAW / "dialect_evolution_profiles_full.csv"
if not input_path.exists():
    raise FileNotFoundError(f"未找到数据文件：{input_path}")

df = pd.read_csv(input_path)

# --- 步骤 A: 方言点群组化 (基于 Combination 列) ---
# 统计每个组合包含哪些方言点名
group_info = df.groupby('Combination')['point_name'].apply(list).reset_index()

def get_group_label(points):
    points = sorted(list(set(points)))
    if len(points) > 1:
        return f"{points[0]}等{len(points)}点"
    return points[0]

group_info['group_node'] = group_info['point_name'].apply(get_group_label)
# 将群组名称映射回主表
df = df.merge(group_info[['Combination', 'group_node']], on='Combination', how='left')

# --- 步骤 B: 宽表转长表 (处理 5 个核心声组) ---
onsets = ['K', 'M', 'P', 'Ø', 'TS']
melted_rows = []

for _, row in df.iterrows():
    g_node = row['group_node']
    for ons in onsets:
        # 提取该声组的 L1, L2, L3 状态
        l1 = str(row[f'{ons}_L1']).strip()
        l2 = str(row[f'{ons}_L2']).strip()
        l3 = str(row[f'{ons}_L3']).strip()
        
        melted_rows.append({
            'group_node': g_node,
            'onset': ons,
            'L1': l1,
            'L2': l2,
            'L3': l3,
            'N_Status': str(row['N_Status']).strip(),
            'T_Status': str(row['T_Status']).strip()
        })

df_long = pd.DataFrame(melted_rows)

# === 3. 构建桑基图链路 (Links) ===
links = []

def add_link(source, target, value, color):
    links.append({'source': source, 'target': target, 'value': value, 'color': color})

# --- 流 A: 方言群组 -> L1 (一级分类) ---
flow_a = df_long.groupby(['group_node', 'L1']).size().reset_index(name='count')
for _, row in flow_a.iterrows():
    color = COLOR_SPLIT if '分立' in row['L1'] else COLOR_MERGE
    add_link(row['group_node'], "L1:" + row['L1'], row['count'], color)

# --- 流 B: L1 -> L2 (二级分类) ---
flow_b = df_long.groupby(['L1', 'L2']).size().reset_index(name='count')
for _, row in flow_b.iterrows():
    color = COLOR_SPLIT if '分立' in row['L1'] else COLOR_MERGE
    add_link("L1:" + row['L1'], "L2:" + row['L2'], row['count'], color)

# --- 流 C: L2 -> L3 (三级具体模式) ---
flow_c = df_long.groupby(['L2', 'L3']).size().reset_index(name='count')
for _, row in flow_c.iterrows():
    # 逻辑：检测 L3 中是否包含越级信息，或者根据之前的分类结果高亮
    color = COLOR_LEAP if '越级' in row['L3'] or 'S1=S3' in row['L3'] or 'S0=S2' in row['L3'] else COLOR_MERGE
    if '全对立' in row['L2'] or '全不等' in row['L2']:
        color = COLOR_SPLIT
    add_link("L2:" + row['L2'], "L3:" + row['L3'], row['count'], color)

# --- N/T 联动独立路径 ---
# 直接从群组连向 N_Status 和 T_Status
flow_n = df_long.drop_duplicates(['group_node', 'N_Status']).groupby(['group_node', 'N_Status']).size().reset_index(name='count')
for _, row in flow_n.iterrows():
    add_link(row['group_node'], "N组:" + row['N_Status'], row['count'], COLOR_NT)

flow_t = df_long.drop_duplicates(['group_node', 'T_Status']).groupby(['group_node', 'T_Status']).size().reset_index(name='count')
for _, row in flow_t.iterrows():
    add_link(row['group_node'], "T组:" + row['T_Status'], row['count'], COLOR_NT)

# === 4. 节点编码与渲染 ===
all_nodes = sorted(list(set([l['source'] for l in links] + [l['target'] for l in links])))
node_map = {node: i for i, node in enumerate(all_nodes)}

sankey_data = {
    'source': [node_map[l['source']] for l in links],
    'target': [node_map[l['target']] for l in links],
    'value': [l['value'] for l in links],
    'color': [l['color'] for l in links]
}

# 清理节点标签（去除内部使用的前缀）
clean_labels = []
for n in all_nodes:
    label = n
    for p in ["L1:", "L2:", "L3:", "N组:", "T组:"]:
        label = label.replace(p, "")
    clean_labels.append(label)

fig = go.Figure(data=[go.Sankey(
    node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5), label = clean_labels),
    link = sankey_data
)])

fig.update_layout(
    title_text="吴语太湖片各方言点歌、模、麻、佳皆韵合并模式桑基图",
    font_size=12, width=1600, height=900
)

# 保存
html_path = FIGS_DIR / "sankey_evolution_profiles.html"
fig.write_html(str(html_path))
print(f"✅ 桑基图已保存：{html_path}")

# === 5. 描述性统计展示 (优化可读性版) ===
print("\n" + "="*60)
print("吴语元音演变模式描述性统计总结")
print("="*60)

# 1. 预先提取每个 Combination 对应的各声组特征 (取该组第一个方言点作为代表)
# 这一步是为了让输出能读到 K_L3, M_L3 等具体内容
features_rep = df.drop_duplicates('Combination').set_index('Combination')

# 2. 统计 Combination 出现的频次
pattern_counts = df.groupby('Combination')['point_name'].apply(list).reset_index()
pattern_counts['count'] = pattern_counts['point_name'].apply(len)
pattern_counts = pattern_counts.sort_values(by='count', ascending=False)

top_n = 3
print(f"\n【演变模式 Top {top_n} 画像分析】")

stats_content = [] # 用于保存到文件

for i, (_, row) in enumerate(pattern_counts.head(top_n).iterrows()):
    comb_id = row['Combination']
    dialects = "、".join(row['point_name'])
    
    # 提取该模式下的具体声组表现 (L3 数据)
    k_v = features_rep.loc[comb_id, 'K_L3']
    m_v = features_rep.loc[comb_id, 'M_L3']
    p_v = features_rep.loc[comb_id, 'P_L3']
    o_v = features_rep.loc[comb_id, 'Ø_L3']
    ts_v = features_rep.loc[comb_id, 'TS_L3']
    n_v = features_rep.loc[comb_id, 'N_Status']
    t_v = features_rep.loc[comb_id, 'T_Status']
    
    # 格式化特征描述字符串
    feature_desc = (f"K: {k_v}, M: {m_v}, P: {p_v} "
                    f"Ø: {o_v}, TS: {ts_v}"
                    f"N: {n_v}, T: {t_v}")
    
    # 打印到控制台
    print(f"NO.{i+1} 模式: {comb_id}")
    print(f"   ➤ 核心特征: {feature_desc}")
    print(f"   ➤ 出现频次: {row['count']} 个方言点")
    print(f"   ➤ 代表方言: {dialects}\n")
    
    # 存入列表准备写入文件
    stats_content.append(f"NO.{i+1} 模式: {comb_id}\n"
                         f"   核心特征: {feature_desc}\n"
                         f"   出现频次: {row['count']} 个方言点\n"
                         f"   详细名单: {dialects}\n\n")

# 3. 将美化后的结果保存到文件
stats_path = FIGS_DIR / "evolution_descriptive_stats.txt"
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("吴语元音演变模式描述性统计总结\n")
    f.write("="*60 + "\n")
    f.writelines(stats_content)

print(f"✅ 易读版统计报告已保存至: {stats_path}")