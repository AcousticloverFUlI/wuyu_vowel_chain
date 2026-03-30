import pandas as pd
from pathlib import Path

# === 路径设置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = PROJECT_ROOT / "data_clean"
DATA_DICT = PROJECT_ROOT / "data_dict"
DATA_DICT.mkdir(exist_ok=True)

def generate_mapping_template():
    input_file = DATA_CLEAN / "wuyu_lexeme.csv"
    output_file = DATA_DICT / "point_coords_master.csv"
    
    if not input_file.exists():
        print(f"❌ 错误：找不到文件 {input_file}")
        return

    # 读取数据
    df = pd.read_csv(input_file)
    
    # 提取唯一的方言点及其小片映射
    # 假设列名为 point_name 和 subbranch
    mapping = df[['point_name', 'subbranch']].drop_duplicates().sort_values('subbranch')
    
    # 增加经纬度列（初始为空）
    mapping['lat'] = ""
    mapping['lon'] = ""
    
    # 导出模板
    mapping.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 模板已生成：{output_file}")
    print(f"👉 请在 CSV 中填写 'lat' 和 'lon' 列后再运行绘图脚本。")

    # === 新增：打印苏沪嘉小片所有方言点 ===
    target_subbranch = "苏沪嘉小片"
    shj_points = mapping[mapping['subbranch'] == target_subbranch]['point_name'].unique().tolist()
    
    print(f"\n🔍 检索到【{target_subbranch}】共 {len(shj_points)} 个方言点：")
    if shj_points:
        # 每行打印 5 个点，方便阅读
        for i in range(0, len(shj_points), 5):
            print("  " + "、".join(shj_points[i:i+5]))
    else:
        print(f"  ⚠️ 未在数据中找到属于 {target_subbranch} 的点。")

if __name__ == "__main__":
    generate_mapping_template()