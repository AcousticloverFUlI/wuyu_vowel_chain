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

# 读取数据，强制指定编码防止中文乱码
df = pd.read_csv(input_path, encoding='utf-8')

# === 3. 数据清洗与整理 ===

# 为了方便区分，我们创建一个“展示标签”，格式为：方言名(声组)
# 这样在旭日图的最外圈能看到具体是哪个点的哪个声组
df['label'] = df['point_name'].astype(str) + "(" + df['onset_class'].astype(str) + ")"

# 清洗分类列的空格
for col in ['一级分类', '二级分类', '三级分类(详细模式)']:
    df[col] = df[col].astype(str).str.strip()

# 排序逻辑：确保分立型在圆图的起始位置，且 S0->S1->S2->S3 按演化逻辑排列
def get_sort_score(row):
    # 分立型排最前
    if row['一级分类'] == '分立型': return 0
    # 合流型内部排序
    p = row['三级分类(详细模式)']
    if 'S0=S1' in p: return 10
    if 'S1=S2' in p: return 20
    if 'S2=S3' in p: return 30
    if row['是否越级'] in [True, 'True', 'true']: return 40 # 越级排后面
    return 50

df['sort_score'] = df.apply(get_sort_score, axis=1)
df = df.sort_values(by=['sort_score', '一级分类', '二级分类'])

# === 4. 视觉编码 (颜色映射) ===

def assign_visual_group(row):
    # 逻辑：分立型冷色，合流型暖色，越级强高亮
    if row['一级分类'] == '分立型':
        return '稳定分立 (Cold)'
    if str(row['是否越级']).lower() == 'true':
        return '越级合并 (Highlight)'
    return '常规合流 (Warm)'

df['VisualGroup'] = df.apply(assign_visual_group, axis=1)

# 定义配色方案
# 稳定分立：青蓝色 | 常规合流：暖橙色 | 越级合并：鲜红色（为了在图中跳脱出来）
color_map = {
    '稳定分立 (Cold)': '#20b2aa',    # LightSeaGreen
    '常规合流 (Warm)': '#f4a460',    # SandyBrown
    '越级合并 (Highlight)': '#ff4500' # OrangeRed
}

# === 5. 绘制旭日图 ===

fig = px.sunburst(
    df,
    path=[
        '一级分类',        # 核心圈
        '二级分类',        # 内环
        '三级分类(详细模式)', # 中环
        'label'           # 外环：方言点(声组)
    ],
    color='VisualGroup',
    color_discrete_map=color_map,
    branchvalues='total',
    hover_data=['S0', 'S1', 'S2', 'S3'] # 悬停时显示具体音值
)

# 样式精修
fig.update_traces(
    textinfo='label',
    insidetextorientation='radial', # 文本沿半径方向排列
    marker=dict(line=dict(color='white', width=0.8)) # 增加白色分割线提高质感
)

# 标题与布局
fig.update_layout(
    title_text="太湖片吴语歌、模、麻、佳皆韵合并模式图",
    title_x=0.5,
    margin=dict(t=60, l=10, r=10, b=10),
    font=dict(size=12)
)

# === 6. 导出文件 ===
# 1. 交互式网页 (推荐)
html_out = FIGS_DIR / "vowel_chain_sunburst.html"
fig.write_html(str(html_out))

# 2. 静态图片 (需要 kaleido)
img_out = FIGS_DIR / "vowel_chain_sunburst.png"
try:
    fig.write_image(str(img_out), scale=3, width=1000, height=1000)
    print(f"✅ 静态图已保存：{img_out}")
except:
    print("⚠️ 环境中未找到 kaleido，仅生成 HTML。")

print(f"✅ 交互式报告已保存：{html_out}")