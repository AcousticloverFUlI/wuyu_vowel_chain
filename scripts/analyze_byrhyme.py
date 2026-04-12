import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from pathlib import Path
import ssl

# === 1. 环境修复 (SSL & Font) ===
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError: pass

def set_chinese_font():
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    preferred_fonts = ["PingFang SC", "Heiti SC", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            break
set_chinese_font()

# === 2. 路径配置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_CSV_PATH = PROJECT_ROOT / "data_clean" / "value_type" / "point_rhyme_inventory.csv"
DATA_DICT = PROJECT_ROOT / "data_dict"
FIGS_DIR = PROJECT_ROOT / "figs" / "rhyme_detail_maps"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === 3. 核心清洗逻辑 (去 i 存 u) ===
def clean_vowel_logic(v_str):
    v = v_str.strip()
    if not v: return None
    # 仅剔除 i/j 介音，保留 u/w 等
    if len(v) > 1 and v[0] in ['i', 'j']:
        return v[1:]
    return v

# === 4. 绘图引擎 ===
def draw_map(gdf, column, title, filename, show_pattern=True):
    fig, ax = plt.subplots(figsize=(14, 12))
    plot = gdf.plot(
        ax=ax, column=column, cmap='YlOrRd', legend=True,
        legend_kwds={'label': f"音值丰富度 ({column})", 'orientation': "horizontal", 'pad': 0.05},
        markersize=220, edgecolor='black', linewidth=1, alpha=0.9, zorder=3
    )
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldPhysical, alpha=0.7, zorder=1)
    except: pass

    for x, y, name, pat, val in zip(gdf.geometry.x, gdf.geometry.y, gdf['point_name'], gdf['pattern'], gdf[column]):
        label = f"{name}\n{pat}\nΣ={val}" if show_pattern else f"{name}\n{val}"
        ax.text(x + 1200, y + 1200, label, fontsize=8, fontweight='bold', ha='left',
                path_effects=[path_effects.withStroke(linewidth=3, foreground="white", alpha=0.8)])

    ax.set_title(title, fontsize=20, pad=30)
    ax.set_axis_off()
    plt.savefig(FIGS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

# === 5. 分析流程 ===
def run_expert_analysis():
    if not TARGET_CSV_PATH.exists(): return
    df = pd.read_csv(TARGET_CSV_PATH)
    
    # A. 韵部合并
    rhyme_map = {'佳': '佳皆', '皆': '佳皆', '麻': '麻', '歌': '歌戈', '戈': '歌戈', '模': '模'}
    df = df[df['rhyme_modern'].isin(rhyme_map.keys())].copy()
    df['category'] = df['rhyme_modern'].map(rhyme_map)

    # B. 统计 (执行“去i留u”清洗)
    point_results = []
    for (name, cat), group in df.groupby(['point_name', 'category']):
        all_raw = []
        for p_set in group['rhyme_phonetic_set']:
            all_raw.extend([s.strip() for s in str(p_set).split(',') if s.strip()])
        # 执行清洗逻辑
        cleaned_set = {clean_vowel_logic(v) for v in all_raw}
        point_results.append({
            'point_name': name, 'category': cat, 'inv_size': len(set(filter(None, cleaned_set)))
        })

    # C. 数据透视与指纹生成
    res_df = pd.DataFrame(point_results)
    target_cats = ['佳皆', '麻', '歌戈', '模']
    pivot_df = res_df.pivot_table(index='point_name', columns='category', values='inv_size').fillna(0).astype(int)
    pivot_df = pivot_df.reindex(columns=target_cats, fill_value=0)
    pivot_df['total_complexity'] = pivot_df.sum(axis=1)
    pivot_df['pattern'] = pivot_df[target_cats].apply(lambda x: "-".join(x.astype(str)), axis=1)

    # D. 坐标合并与地理转换
    coords_df = pd.read_csv(DATA_DICT / "point_coords_master.csv")
    coords_df.columns = coords_df.columns.str.strip()
    final_df = pivot_df.reset_index().merge(coords_df[['point_name', 'lat', 'lon']], on='point_name', how='left').dropna()
    gdf = gpd.GeoDataFrame(final_df, geometry=gpd.points_from_xy(final_df.lon, final_df.lat), crs="EPSG:4326").to_crs(epsg=3857)

    # E. 生成全套地图
    print("🎨 正在绘制分韵地图...")
    for cat in target_cats:
        draw_map(gdf, cat, f" {cat} 韵", f"detail_{cat}.png", show_pattern=False)
    draw_map(gdf, 'total_complexity', "吴语元音复杂度分布总览", "total_summary.png")

    # F. 专家文本报告
    print("\n" + "📜" + " 吴语元音演化复杂度专家报告 " + "📜")
    print("-" * 50)
    
    top_3 = pivot_df.nlargest(3, 'total_complexity')
    bot_3 = pivot_df.nsmallest(3, 'total_complexity')
    print(f"🔥 保守核心 (音值最多): {', '.join(top_3.index.tolist())}")
    print(f"🌊 演化先锋 (归并最快): {', '.join(bot_3.index.tolist())}")

    stability = pivot_df[target_cats].std().sort_values(ascending=False)
    print(f"\n⚡️ 韵部演化活跃度排行 (离散度):")
    for r, s in stability.items():
        tag = "演变风暴眼" if s > 1.2 else "演变平稳区"
        print(f" - {r:<5} 韵: {s:.2f} ({tag})")

    trend_rhyme = pivot_df[target_cats].mean().idxmax()
    print(f"\n💡 核心趋势：当前太湖片吴语的复杂度主要由【{trend_rhyme}】韵维系。")
    print(f"📂 地图已保存至: {FIGS_DIR}")
    print("-" * 50)

if __name__ == "__main__":
    run_expert_analysis()