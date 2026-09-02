"""
查询 weevil_data 的时间范围。

使用方法:
    poetry run python scripts/weather/check_weevil_data_range.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from app.services.database_service import DatabaseManager
from app.services.data_preparation_service import DataPreparationService
from app.config.weevilltrak_config import WeevillTrakConfig
from griddedweather.s3_store import S3Manager


def main():
    # 加载配置
    config_path = project_root / "app" / "config" / "weeviltrak_v2.3.yml"
    config = WeevillTrakConfig(str(config_path))
    
    # 初始化服务
    db_manager = DatabaseManager(config)
    bucket_name = config.config.get('data_source', {}).get('s3_bucket', 'sps-ds-bucket')
    s3_manager = S3Manager(bucket_name)
    data_service = DataPreparationService(db_manager, s3_manager)
    
    # 拉取数据（使用 2026-04-14 作为截止日期）
    today_str = "2026-04-14"
    print(f"正在拉取 weevil_data (today={today_str})...")
    
    try:
        weevil_data = data_service.pull_weevil_data(today=today_str)
        
        print(f"\n=== 数据时间范围 ===")
        print(f"总行数: {len(weevil_data)}")
        
        if 'stage_date' in weevil_data.columns:
            weevil_data['stage_date'] = pd.to_datetime(weevil_data['stage_date'])
            min_date = weevil_data['stage_date'].min()
            max_date = weevil_data['stage_date'].max()
            print(f"最早日期: {min_date.strftime('%Y-%m-%d')}")
            print(f"最晚日期: {max_date.strftime('%Y-%m-%d')}")
            print(f"时间跨度: {(max_date - min_date).days} 天")
        
        print(f"\n=== 年份分布 ===")
        if 'year' in weevil_data.columns:
            print(weevil_data['year'].value_counts().sort_index())
        
        print(f"\n=== 数据列 ===")
        print(weevil_data.columns.tolist())
        
        print(f"\n=== 前5行 ===")
        print(weevil_data.head())
        
    except Exception as e:
        print(f"查询失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
