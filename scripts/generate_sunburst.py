import pandas as pd
import plotly.express as px
from pathlib import Path

# === 1. 路径设置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
FIGS_DIR = PROJECT_ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === 2. 加载数据 ===
file_name = "mainlayer_merge.csv"
input_path = DATA_RAW / file_name

if not input_path.exists():
    raise FileNotFoundError(f"未找到输入文件：{input_path}")

df = pd.read_csv(input_path, encoding='utf-8')

# === 3. 数据清洗与描述性统计计算 ===
# 清洗分类列的空格
for col in ['一级分类', '二级分类', '三级分类(详细模式)']:
    df[col] = df[col].astype(str).str.strip()

# 计算一级分类占比
l1_counts = df['一级分类'].value_counts(normalize=True) * 100
# 计算二级分类占比（Top 3）
l2_counts = df['二级分类'].value_counts(normalize=True) * 100
# 计算三级分类占比（Top 3）
l3_counts = df['三级分类(详细模式)'].value_counts(normalize=True) * 100

# B. 【核心新增】越级合并（异常值）专项统计
# 统一处理布尔值
df['is_leap'] = df['是否越级'].apply(lambda x: str(x).strip().lower() == 'true')
leap_count = df['is_leap'].sum()
leap_percent = (leap_count / len(df)) * 100

# 构建统计文本
stats_text = (
    "<b>描述性统计</b><br>"
    "--------------------------------<br>"
    "<b>[第一层]</b><br>"
    f"• {l1_counts.index[0]}: {l1_counts.values[0]:.1f}%<br>"
    f"• {l1_counts.index[1]}: {l1_counts.values[1]:.1f}%<br><br>"
    "<b>[第二层]</b><br>"
    f"• {l2_counts.index[0]}: {l2_counts.values[0]:.1f}%<br>"
    f"• {l2_counts.index[1]}: {l2_counts.values[1]:.1f}%<br><br>"
    "<b>[第三层]</b><br>"
    f"• {l3_counts.index[0]}: {l3_counts.values[0]:.1f}%<br>"
    f"• {l3_counts.index[1]}: {l3_counts.values[1]:.1f}%<br>"
    f"• {l3_counts.index[2]}: {l3_counts.values[2]:.1f}<br>"
    "<b>[越级合并]</b><br>"
    f"• <span style='color:red;'><b>S1=S3: {leap_percent:.1f}%</b></span><br>"
    f"  (共 {leap_count} 个样本)"
)

# === 4. 视觉编码 (颜色映射) ===
def assign_visual_group(row):
    if row['一级分类'] == '分立型': return '稳定分立 (Cold)'
    if str(row['是否越级']).lower() == 'true': return '越级合并 (Highlight)'
    return '常规合流 (Warm)'

df['VisualGroup'] = df.apply(assign_visual_group, axis=1)
df['label'] = df['point_name'] + "(" + df['onset_class'] + ")"

color_map = {
    '稳定分立 (Cold)': '#20b2aa',    # 青色
    '常规合流 (Warm)': '#f4a460',    # 橙色
    '越级合并 (Highlight)': '#ff4500' # 红色
}

# === 5. 绘制旭日图 ===
fig = px.sunburst(
    df,
    path=['一级分类', '二级分类', '三级分类(详细模式)', 'label'],
    color='VisualGroup',
    color_discrete_map=color_map,
    branchvalues='total'
)

# 样式精修：增加百分比显示
fig.update_traces(
    # label: 名字, percent entry: 占总体的百分比
    textinfo='label+percent entry', 
    insidetextorientation='radial',
    marker=dict(line=dict(color='white', width=1)),
    hovertemplate='<b>%{label}</b><br>占比: %{percentEntry:.1%}'
)

# === 6. 添加侧边描述性统计看板 (Annotation) ===
fig.add_annotation(
    text=stats_text,
    align='left',
    showarrow=False,
    xref='paper', yref='paper',
    x=1.18, y=0.5,  # 将看板置于图表右侧
    bordercolor='black',
    borderwidth=1,
    borderpad=10,
    bgcolor='rgba(255,255,255,0.8)', # 半透明背景
    font=dict(size=11, family="Arial")
)

# 调整整体布局，给右侧看板留出空间
fig.update_layout(
    title_text="太湖片吴语歌、模、麻、佳皆韵合并模式图",
    title_x=0.4,
    margin=dict(t=80, l=50, r=250, b=50), # 增加右边距 (r=250)
    width=1100, height=800
)

# === 7. 导出文件 ===
html_out = FIGS_DIR / "vowel_chain_sunburst_stats.html"
fig.write_html(str(html_out))

# 如果需要图片导出，请确保安装了 kaleido (pip install kaleido)
img_out = FIGS_DIR / "vowel_chain_sunburst_stats.png"
try:
    fig.write_image(str(img_out), scale=3)
    print(f"✅ 带有统计看板的图片已保存：{img_out}")
except:
    print("⚠️ 仅生成交互式 HTML 报告。")

print(f"✅ 描述性统计旭日图已完成！")