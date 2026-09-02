"""
分析 2023-2025 年每年 Stage 1 的 unique location 数量并绘制 Choropleth 地图
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pipeline.base import PipelineBase
from app.settings import setup_logger

logger = setup_logger()


def analyze_stage1_locations():
    """分析 2023-2025 年 Stage 1 的 unique location 数量"""
    
    # 初始化 pipeline（用于调用 data_service）
    pipeline = PipelineBase(config_path="app/config/weeviltrak_v2.3.yml")
    
    # 获取 2023-2025 年的数据（用 2025-12-31 作为 cutoff，确保包含全年）
    logger.info("拉取 2023-2025 年的 weevil 数据...")
    weevil_data = pipeline.data_service.pull_weevil_data(today="2025-12-31")
    
    logger.info(f"总共获取 {len(weevil_data)} 条记录")
    logger.info(f"数据列: {weevil_data.columns.tolist()}")
    logger.info(f"年份分布:\n{weevil_data['year'].value_counts().sort_index()}")
    
    # 筛选 Stage 1 的数据
    stage1_data = weevil_data[weevil_data['stage_id'] == 1].copy()
    logger.info(f"\nStage 1 记录数: {len(stage1_data)}")
    
    # 统计每年 Stage 1 的 unique location_id 数量
    results = {}
    for year in [2023, 2024, 2025]:
        year_data = stage1_data[stage1_data['year'] == year]
        unique_locs = year_data['location_id'].unique()
        results[year] = {
            'count': len(unique_locs),
            'locations': unique_locs,
            'data': year_data
        }
        logger.info(f"\n{year}年 Stage 1 unique location 数量: {len(unique_locs)}")
        logger.info(f"Location IDs: {sorted(unique_locs)}")
    
    return results


def create_choropleth_maps(results: dict):
    """为每年创建 Choropleth 地图"""
    
    output_dir = Path("outputs/validation/implementation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for year, info in results.items():
        data = info['data']
        
        # 按 location_id 去重，保留坐标信息
        loc_data = data[['location_id', 'latitude', 'longitude']].drop_duplicates()
        
        logger.info(f"\n创建 {year} 年的地图，共 {len(loc_data)} 个地点")
        
        # 创建地图
        fig = go.Figure(data=go.Scattergeo(
            lon=loc_data['longitude'],
            lat=loc_data['latitude'],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                opacity=0.7,
                line=dict(width=1, color='darkred')
            ),
            text=loc_data['location_id'],
            hovertemplate='<b>Location ID:</b> %{text}<br>' +
                         '<b>Latitude:</b> %{lat:.4f}<br>' +
                         '<b>Longitude:</b> %{lon:.4f}<br>' +
                         '<extra></extra>'
        ))
        
        # 设置地图布局
        fig.update_layout(
            title=dict(
                text=f'{year}年 Stage 1 Unique Locations (n={len(loc_data)})',
                x=0.5,
                font=dict(size=20)
            ),
            geo=dict(
                scope='north america',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                showcountries=True,
                countrycolor='rgb(204, 204, 204)',
                showsubunits=True,
                subunitcolor='rgb(217, 217, 217)',
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
                showrivers=True,
                rivercolor='rgb(255, 255, 255)',
                showocean=True,
                oceancolor='rgb(230, 245, 255)',
                projection=dict(type='albers usa'),
                center=dict(lat=39.8, lon=-98.5),
                lonaxis=dict(range=[-130, -65]),
                lataxis=dict(range=[25, 50]),
            ),
            height=700,
            width=1000,
        )
        
        # 保存为 HTML 和 PNG
        html_path = output_dir / f'stage1_locations_{year}_map.html'
        png_path = output_dir / f'stage1_locations_{year}_map.png'
        
        fig.write_html(str(html_path))
        fig.write_image(str(png_path), scale=2)
        
        logger.info(f"地图已保存: {html_path}")
        logger.info(f"图片已保存: {png_path}")
    
    return output_dir


def create_summary_table(results: dict, output_dir: Path):
    """创建汇总表格"""
    
    summary_data = []
    for year in [2023, 2024, 2025]:
        info = results[year]
        loc_list = sorted(info['locations'])
        summary_data.append({
            '年份': year,
            'Stage 1 Unique Location 数量': info['count'],
            'Location IDs': ', '.join(loc_list)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存为 CSV
    csv_path = output_dir / 'stage1_locations_summary.csv'
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n汇总表已保存: {csv_path}")
    
    # 保存为 Markdown
    md_path = output_dir / 'stage1_locations_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Stage 1 Unique Locations 统计 (2023-2025)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## 详细说明\n\n")
        for year in [2023, 2024, 2025]:
            info = results[year]
            f.write(f"\n### {year}年\n")
            f.write(f"- **Unique Location 数量**: {info['count']}\n")
            f.write(f"- **Location IDs**: {', '.join(sorted(info['locations']))}\n")
    
    logger.info(f"汇总报告已保存: {md_path}")
    
    return summary_df


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始分析 2023-2025 年 Stage 1 Unique Locations")
    logger.info("=" * 60)
    
    # 分析数据
    results = analyze_stage1_locations()
    
    # 创建地图
    output_dir = create_choropleth_maps(results)
    
    # 创建汇总表
    summary_df = create_summary_table(results, output_dir)
    
    # 打印最终结果
    logger.info("\n" + "=" * 60)
    logger.info("分析完成！结果汇总:")
    logger.info("=" * 60)
    for year in [2023, 2024, 2025]:
        count = results[year]['count']
        logger.info(f"{year}年 Stage 1 Unique Location 数量: {count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
