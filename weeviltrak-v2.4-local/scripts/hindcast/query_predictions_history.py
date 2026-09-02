"""
查询 weeviltrak_models_predictions_history 表并保存到本地文件。

使用方法:
    poetry run python scripts/hindcast/query_predictions_history.py

输出:
    outputs/hindcast/weeviltrak_models_predictions_history_YYYYMMDD.csv
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from app.services.database_service import DatabaseManager
from app.config.weevilltrak_config import WeevillTrakConfig


def main():
    # 加载配置
    config_path = project_root / "app" / "config" / "weeviltrak_v2.3.yml"
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    config = WeevillTrakConfig(str(config_path))

    # 初始化数据库管理器
    db_manager = DatabaseManager(config)

    # 查询数据
    query = """
    SELECT *
    FROM weeviltrak_models_predictions_history
    WHERE prediction_date BETWEEN '2026-03-01' AND '2026-03-24'
    ORDER BY prediction_date, location_id, stage_id
    """

    print("正在查询 weeviltrak_models_predictions_history 表...")
    print(f"查询范围: 2026-03-01 至 2026-03-24")

    try:
        with db_manager as db:
            conn = db.get_connection()
            df = pd.read_sql(query, conn)

        print(f"查询完成，共 {len(df)} 行数据")

        # 保存到 docs 文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = project_root / "outputs" / "hindcast" / f"weeviltrak_models_predictions_history_{timestamp}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"数据已保存到: {output_file}")

        # 显示数据概览
        print("\n数据概览:")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print(f"  列名: {list(df.columns)}")
        print(f"\n前5行:")
        print(df.head())

    except Exception as e:
        print(f"查询失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
