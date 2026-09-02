#!/usr/bin/env python3
"""
快速检查 2023-2025 年 Stage 1 的 unique location 数量
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.database_service import DatabaseManager
from app.config.weevilltrak_config import WeevillTrakConfig
import pandas as pd

# 初始化配置和数据库连接
config = WeevillTrakConfig("app/config/weeviltrak_v2.3.yml")
db = DatabaseManager(config)

# 直接查询数据库
query = """
SELECT 
    YEAR(stage_date) as year,
    location_id,
    latitude,
    longitude
FROM europe_dna.europe_dna_sps_weeviltrak
WHERE stage_name = 'Stage 1'
    AND YEAR(stage_date) IN (2023, 2024, 2025)
    AND stage_date <= '2025-12-31'
    AND location_id != '5'
GROUP BY YEAR(stage_date), location_id, latitude, longitude
ORDER BY year, location_id
"""

print("正在查询数据库...")
with db as db_conn:
    conn = db_conn.get_connection()
    df = pd.read_sql(query, conn)

print(f"\n总共获取 {len(df)} 条 Stage 1 记录")
print(f"\n年份分布:")
print(df['year'].value_counts().sort_index())

print("\n" + "=" * 60)
print("每年 Stage 1 Unique Location 数量:")
print("=" * 60)

for year in [2023, 2024, 2025]:
    year_df = df[df['year'] == year]
    unique_locs = year_df['location_id'].unique()
    print(f"\n{year}年: {len(unique_locs)} 个 locations")
    print(f"Location IDs: {sorted(unique_locs)}")

print("\n" + "=" * 60)
