import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects  # 显式导入，防止路径报错
from pathlib import Path

# === 1. 路径与中文字体设置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DICT = PROJECT_ROOT / "data_dict"
FIGS_DIR = PROJECT_ROOT / "figs"
FIGS_DIR.mkdir(exist_ok=True)

def set_chinese_font():
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    preferred_fonts = ["PingFang SC", "Heiti SC", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# === 2. 加载与转换数据 ===
def plot_map():
    coord_file = DATA_DICT / "point_coords_master.csv"
    if not coord_file.exists():
        print("❌ 错误：找不到坐标映射文件。请先填写坐标。")
        return

    df = pd.read_csv(coord_file)
    df = df.dropna(subset=['lat', 'lon'])
    
    if df.empty:
        print("⚠️ 警告：坐标映射表中没有有效的经纬度数据。")
        return

    # 转换为 GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    # 投影转换
    gdf_web = gdf.to_crs(epsg=3857)

    # === 3. 绘图配置 ===
    fig, ax = plt.subplots(figsize=(12, 12))

    subbranches = gdf_web["subbranch"].unique()
    cmap = plt.get_cmap('tab10')
    color_map = {sub: cmap(i) for i, sub in enumerate(subbranches)}

    for sub in subbranches:
        subset = gdf_web[gdf_web["subbranch"] == sub]
        subset.plot(
            ax=ax, 
            color=color_map[sub], 
            label=sub, 
            markersize=80, 
            edgecolor='white', 
            linewidth=1,
            alpha=0.9, 
            zorder=3
        )

    # === 4. 地貌底图设置 ===
    try:
        ctx.add_basemap(
            ax, 
            source=ctx.providers.Esri.WorldPhysical, 
            alpha=0.6, 
            zorder=1
        )
    except Exception as e:
        print(f"底图加载失败，尝试备用源...")
        ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap, alpha=0.5, zorder=1)

    # === 5. 标注与修饰 (已修正拼写错误) ===
    for x, y, label in zip(gdf_web.geometry.x, gdf_web.geometry.y, gdf_web["point_name"]):
        ax.text(
            x + 1500, y + 1500, 
            label, 
            fontsize=9, 
            ha='left', 
            va='bottom', 
            fontweight='bold', 
            zorder=4,
            # 修正处：patheffects 拼写已更正，并使用了导入的 path_effects 模块
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white", alpha=0.7)]
        )

    ax.legend(title="吴语小片分类", loc='upper right', frameon=True)
    ax.set_title("吴语太湖片方言采样点地理分布及地形地貌图", fontsize=18, pad=20)
    ax.set_axis_off()

    # 保存图片
    output_png = FIGS_DIR / "wuyu_topography_map.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"✅ 地图已成功生成并保存至: {output_png}")

if __name__ == "__main__":
    plot_map()