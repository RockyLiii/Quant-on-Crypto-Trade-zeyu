import os
import yaml
import argparse
import logging
import pandas as pd
from src.dataloader.df_loader import load_symbol_data, merge_dataframes, get_symbol
from src.utils.calculate import calculate_correlations, calculate_residuals, filter_residuals, calculate_returns
from src.utils.logger import get_logger
from src.utils.parser import get_args
from src.visual.visualize import (
    plot_prices, plot_residuals, plot_betas, plot_std, plot_correlations, 
    plot_residual_histograms, plot_cumulative_returns, plot_return_series, 
    plot_daily_transaction_count
)

import warnings
warnings.filterwarnings("ignore")

def main():
    """
    主函数：加载数据、计算指标、可视化结果
    """
    args = get_args()
    
    # 读取配置文件
    with open(args.conf_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger = get_logger(config)

    # 批量读取K线数据
    logger.info(f"正在从 {config['data_dir']} 读取K线数据")
    files = [f for f in os.listdir(config['data_dir']) if f.endswith(config['file_suffix'])]
    dataframes = [load_symbol_data(f, config['data_dir'], config['min_csv_length']) for f in files]
    dataframes = [df for df in dataframes if df is not None]

    # 合并所有币种的数据（按 timestamp）
    if dataframes:
        logger.info(f"成功加载 {len(dataframes)} 个币种数据，正在合并")
        df_all = merge_dataframes(dataframes, files)
        symbols = [get_symbol(f) for f in files if get_symbol(f) in df_all.columns]
        logger.info(f"分析的币种: {symbols}")
        if config['visualize_input']:
            logger.info("正在可视化输入数据")
            plot_prices(df_all, symbols, config)
    else:
        logger.error("🚫 没有成功加载任何币种数据")
        return
    
    # Use first half of data for analysis
    logger.info("正在使用前一半数据进行分析")

    df_all = df_all.iloc[:int(len(df_all) * 1/5)].reset_index(drop=True)

    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    
    # 从分析对象中移除BTC
    if 'BTC' in symbols:
        symbols.remove('BTC')

    # 计算残差、beta系数、标准差
    logger.info("正在计算残差、beta系数和标准差")
    df_resid, residuals_dict, betas_dict, std_dict, timestamps = calculate_residuals(df_all, symbols, config)
    
    # 计算相关性
    logger.info("正在计算相关性")
    corr_aves_df, corr_aves_dict = calculate_correlations(df_resid, df_all, symbols, config, timestamps)
    
    # === 保留结构不变，同时加入平均相关性计算 ===
    df_resid = pd.DataFrame(residuals_dict)
    df_resid.index.name = 'timestamp'

    df_std = pd.DataFrame(std_dict)
    df_std.index.name = 'timestamp'

    df_beta = pd.DataFrame(betas_dict)
    df_beta.index.name = 'timestamp'
    
    # 调试信息：打印相关性DataFrame的结构
    logger.info(f"相关性DataFrame的类型: {type(corr_aves_df)}")
    logger.info(f"相关性DataFrame的形状: {corr_aves_df.shape if hasattr(corr_aves_df, 'shape') else 'Series'}")
    logger.info(f"相关性DataFrame的前几行:\n{corr_aves_df.head() if not corr_aves_df.empty else '空Series'}")
    
    # 可视化中间数据
    if config['visualize_statics']:
        logger.info("正在可视化中间数据")
        plot_residuals(df_resid, symbols, config)
        #plot_residual_histograms(residuals_dict, config)
        plot_betas(df_beta, symbols, config)
        plot_std(df_std, symbols, config)
        plot_correlations(corr_aves_df, config)

    # 过滤残差
    logger.info("正在过滤残差")

    residuals_positive_dict, residuals_positive_count = filter_residuals(residuals_dict, std_dict, corr_aves_dict, config)
    logger.info(f"过滤后的交易信号数量: {residuals_positive_count}")

    # 计算收益
    logger.info("正在计算收益")
    result_df, result_list, hold_df, hold_list, metric = calculate_returns(df_all, residuals_positive_dict, betas_dict, config)
    print(f"################# metric: {metric} #################")
    
    # 调试信息：打印结果DataFrame的结构
    logger.info(f"结果DataFrame的列: {result_df.columns.tolist() if not result_df.empty else '空DataFrame'}")
    logger.info(f"结果DataFrame的形状: {result_df.shape if not result_df.empty else '空DataFrame'}")
    if not result_df.empty:
        logger.info(f"结果DataFrame的前几行:\n{result_df.head()}")

    # 可视化输出结果
    if config['visualize_output']:
        logger.info("正在可视化输出结果")
        plot_cumulative_returns(result_df, config)
        plot_return_series(result_list, config)
        plot_daily_transaction_count(residuals_positive_dict, config)
    
    logger.info("分析完成")


if __name__ == "__main__":
    main()