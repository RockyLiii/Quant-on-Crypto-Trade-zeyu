import os
import matplotlib.pyplot as plt
import pandas as pd
from src.dataloader.df_loader import get_symbol
import numpy as np

def plot_prices(df_all, symbols, config):
    """
    可视化归一化后的币种价格走势
    Args:
        df_all (pd.DataFrame): 包含所有币种价格数据的 DataFrame
        files (list): 文件列表
        config (dict): 配置字典
    """
    
    df_all.set_index("timestamp")[symbols].plot(figsize=(12, 6), title="Normalized Price Trends")
    plt.ylabel("Normalized Price")
    plt.grid(True)
    plt.savefig(os.path.join(config['output_path'], "normalized_price_trends.png"))
    

def plot_residuals(df_resid, symbols, config):
    """
    绘制残差图表
    
    Args:
        df_resid (pd.DataFrame): 包含所有币种残差的DataFrame
        symbols (list): 币种符号列表
        config (dict): 配置参数字典
    """
    plt.figure(figsize=(14, 6))
    for sym in symbols:
        plt.plot(df_resid.index, df_resid[sym], label=f"Residual(BTC - {sym})")
    plt.legend()
    plt.grid(True)
    plt.title(f"Rolling Residuals using t-past-{config['window_size']} to t-past Regression (past={2 * config['window_size']})")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], "residual_plot.png"))
    plt.close()


def plot_betas(df_beta, symbols, config):
    """
    绘制beta系数图表
    
    Args:
        df_beta (pd.DataFrame): 包含所有币种beta系数的DataFrame
        symbols (list): 币种符号列表
        config (dict): 配置参数字典
    """
    plt.figure(figsize=(14, 6))
    for sym in symbols:
        plt.plot(df_beta.index, df_beta[sym], label=f"Beta(BTC ~ {sym})")
    plt.legend()
    plt.grid(True)
    plt.title(f"Rolling Betas using t-past-{config['window_size']} to t-past Regression (past={2 * config['window_size']})")
    plt.xlabel("Time")
    plt.ylabel("Beta")
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], "residual_beta_plot.png"))
    plt.close()


def plot_std(df_std, symbols, config):
    """
    绘制标准差图表
    
    Args:
        df_std (pd.DataFrame): 包含所有币种残差标准差的DataFrame
        symbols (list): 币种符号列表
    """
    plt.figure(figsize=(14, 6))
    for sym in symbols:
        plt.plot(df_std.index, df_std[sym], label=f"STD(BTC - {sym})")
    plt.legend()
    plt.grid(True)
    plt.title(f"Rolling Residual STD")
    plt.xlabel("Time")
    plt.ylabel("Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], "residual_std_plot.png"))
    plt.close()


def plot_correlations(corr_aves_df, config):
    """
    绘制残差之间的滚动平均相关性图表
    
    Args:
        corr_aves_df (pd.Series): 平均相关性Series
    """
    plt.figure(figsize=(14, 6))
    plt.plot(corr_aves_df.index, corr_aves_df.values, label="Rolling Avg Pairwise Correlation")
    plt.legend()
    plt.grid(True)
    plt.title("Rolling Average Pairwise Correlation (288*3 window)")
    plt.xlabel("Time")
    plt.ylabel("Average Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], "rolling_avg_correlation_plot.png"))
    plt.close()
    

def plot_residual_histograms(residuals_dict, config):
    """
    绘制每个币种的残差分布直方图

    Args:
        residuals_dict (dict): 包含每个币种残差的字典
        config (dict): 配置参数字典
    """
    for symbol, series in residuals_dict.items():
        # 提取残差值并转换为数值类型
        values = pd.to_numeric(series.apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x), errors='coerce')
        
        # 移除任何 NaN 值
        values = values.dropna()
        
        if len(values) == 0:
            continue
            
        # 计算统计指标
        mean_val = values.mean()
        std_val = values.std()

        # 创建直方图
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        
        # 添加均值线
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, 
                    label=f"Mean: {mean_val:.4f}")
        
        # 设置图表标题和标签
        plt.title(f"{symbol} Residuals Distribution")
        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        plt.text(0.95, 0.95, f"μ = {mean_val:.4f}\nσ = {std_val:.4f}", 
                ha='right', va='top', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_path'], f"{symbol}_residuals_histogram.png"))
        plt.close()
            
def plot_cumulative_returns(result_df, config):
    """
    绘制总累计收益图
    
    Args:
        result_df (pd.DataFrame): 包含交易结果的DataFrame
    """
    # 检查DataFrame是否为空
    if result_df.empty:
        print("警告: 结果DataFrame为空，无法绘制累计收益图")
        return
        
    # 检查timestamp列是否存在，如果不存在，尝试使用索引
    if 'timestamp' not in result_df.columns:
        print("警告: 结果DataFrame中没有'timestamp'列，尝试使用索引作为时间轴")
        # 如果索引是时间戳类型，直接使用索引
        if isinstance(result_df.index, pd.DatetimeIndex):
            result_df = result_df.copy()
            result_df['timestamp'] = result_df.index
        else:
            # 否则创建一个简单的时间序列索引
            result_df = result_df.copy()
            result_df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(result_df), freq='D')
    
    # 确保 timestamp 为 datetime 类型
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])

    # 将列表类型的 beta 和 return_5steps 转换为 float
    if 'beta' in result_df.columns:
        # 保持beta为列表格式，不转换为float
        pass
    if 'return_5steps' in result_df.columns:
        # 保持return_5steps为列表格式，不转换为float
        pass
    else:
        print("警告: 结果DataFrame中没有'return_5steps'列，无法计算累计收益")
        return

    # 按时间排序
    result_df = result_df.sort_values('timestamp')

    # 计算累计收益 - 从列表中提取值
    result_df['cumulative_return'] = result_df['return_5steps'].apply(lambda x: x[0] if isinstance(x, list) else x).cumsum()

    # ---------- 📈 图一：总累计收益 ----------
    plt.figure(figsize=(12, 6))
    plt.plot(result_df['timestamp'], result_df['cumulative_return'], label='Cumulative Return', color='blue')
    plt.xlabel('Timestamp')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], "cumulative_return_plot.png"))
    plt.close()

def plot_return_series(result_list, config):
    """
    绘制所有收益序列和均值标准差图
    
    Args:
        result_list (list): 包含交易结果的列表
    """
    # 检查结果列表是否为空
    if not result_list:
        print("警告: 结果列表为空，无法绘制收益序列图")
        return
        
    # 假设 result_list 中每个 item 的 "return_series" 长度相同
    # 从列表中提取值
    all_series = np.array([[item[0] for item in series] for series in [item["return_series"] for item in result_list]])
    mean_series = np.mean(all_series, axis=0).flatten()  # 每个时间点的均值
    std_series = np.std(all_series, axis=0).flatten()    # 每个时间点的标准差

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- 第一张图：所有的 return_series 曲线 ---
    for series in all_series:
        axs[0].plot(series, color='lightblue', alpha=0.4)

    axs[0].set_title("All Return Series")
    axs[0].set_ylabel("Total Return")
    axs[0].grid(True)

    # --- 第二张图：均值和 ±1 标准差 ---
    axs[1].plot(mean_series, color='blue', linewidth=2, label='Mean')
    axs[1].fill_between(range(len(mean_series)),
                        mean_series - std_series,
                        mean_series + std_series,
                        color='blue',
                        alpha=0.2,
                        label='±1 Std Dev')

    axs[1].set_title("Mean ± Std Dev of Return Series")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Total Return")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], "all_return_series_with_mean_std.png"))
    plt.close()

def plot_daily_transaction_count(residuals_positive_dict, config):
    """
    绘制每日交易量统计图
    
    Args:
        residuals_positive_dict (dict): 包含满足条件的残差的字典
    """
    # 创建一个图形
    plt.figure(figsize=(14, 6))

    # 设置颜色池，这里为示例使用了 matplotlib 自带的颜色
    colors = plt.cm.tab20c.colors  # 可以选择不同的颜色地图，这里用的是 tab20c

    # 获取所有 symbol 的每日交易量统计数据
    daily_counts_dict = {}

    for symbol in residuals_positive_dict.keys():
        # 提取每个 symbol 的时间戳数据
        data = residuals_positive_dict[symbol]
        
        # 提取时间戳并转成 DataFrame
        timestamps = [item[0] for item in data]
        df = pd.DataFrame({'timestamp': timestamps})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 按天统计频次
        df['date'] = df['timestamp'].dt.date
        
        # 创建完整的日期范围，从最小日期到最大日期
        date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        
        # 计算每天的交易量（频次），并确保所有日期都有显示
        daily_counts = df['date'].value_counts().reindex(date_range.date, fill_value=0)
        
        # 将结果保存到字典中
        daily_counts_dict[symbol] = daily_counts

    # 获取所有日期（时间戳）
    all_dates = sorted(set(date for counts in daily_counts_dict.values() for date in counts.index))

    # 设置柱状图的底部
    bottoms = {symbol: [0] * len(all_dates) for symbol in daily_counts_dict.keys()}

    # 叠加每个 symbol 的交易量
    for idx, (symbol, daily_counts) in enumerate(daily_counts_dict.items()):
        # 获取每个日期的交易量
        counts = daily_counts.reindex(all_dates, fill_value=0).values
        
        # 绘制叠加的柱状图
        plt.bar(all_dates, counts, label=symbol, color=colors[idx % len(colors)], bottom=bottoms[symbol])
        
        # 更新底部位置
        for i in range(len(all_dates)):
            bottoms[symbol][i] += counts[i]

    # 图形设置
    plt.title('Stacked Daily Transaction Count for All Symbols')
    plt.xlabel('Date')
    plt.ylabel('Transaction Count')
    plt.xticks(rotation=45)
    plt.legend(title='Symbols')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_path'], "daily_transaction_count.png"))
    plt.close()