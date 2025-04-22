import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import warnings
import copy  # 引入 copy 模块

def calculate_residuals(df_all, symbols, config):
        """
        计算每个币种相对于BTC的残差、beta系数和标准差
        
        Args:
            df_all (pd.DataFrame): 包含所有币种价格数据的DataFrame
            symbols (list): 币种符号列表
            config (dict): 配置参数字典
            
        Returns:
            tuple: 包含残差DataFrame、beta系数DataFrame和标准差DataFrame
        """
        # 初始化存储各种计算结果的字典
        residuals_dict = {}  # 存储每个币种的残差
        betas_dict = {}      # 存储每个币种的beta系数
        std_dict = {}        # 存储每个币种残差的标准差

        # 对每个币种进行回归分析
        for sym in tqdm(symbols, desc="计算残差和beta系数"):
            std_temp = 0.02  # 初始标准差值
            residuals = []   # 存储该币种的残差
            betas = []       # 存储该币种的beta系数
            stds = []        # 存储该币种残差的标准差
            timestamps = []  # 存储对应的时间戳

            # 滚动窗口分析：从(4*window_size)开始，确保有足够的历史数据进行回归
            for t in tqdm(range( 4*config['window_size'], len(df_all)), desc=f"处理 {sym}", leave=False):
                # 提取回归窗口的数据：使用t-4*window_size到t-window_size-1的数据进行回归
                x_window = df_all.loc[t - 4*config['window_size']: t - 1*config['window_size'] - 1, sym].values.reshape(-1, 1)
                y_window = df_all.loc[t - 4*config['window_size']: t - 1*config['window_size'] - 1, 'BTC'].values.reshape(-1, 1)

                # 检查数据是否包含NaN值，如果有则跳过此次计算
                if np.any(np.isnan(x_window)) or np.any(np.isnan(y_window)):
                    residuals.append(np.nan)
                    timestamps.append(df_all.loc[t, 'timestamp'])
                    continue

                # 使用线性回归模型拟合数据，不包含截距项
                reg = LinearRegression(fit_intercept=False).fit(x_window, y_window)
                beta = reg.coef_[0]  # 获取回归系数
                # 获取当前时间点的实际值
                x_t = df_all.loc[t, sym]
                y_t = df_all.loc[t, 'BTC']
                
                # 计算预测值和残差
                pred = beta * x_t
                residual = y_t - pred

                # 使用指数加权移动平均更新标准差
                std_temp = np.sqrt(std_temp**2 * 0.99 + residual**2 * 0.01)

                # 存储计算结果
                residuals.append(residual)
                betas.append(beta)
                stds.append(std_temp)
                timestamps.append(df_all.loc[t, 'timestamp'])

            # 将计算结果存入对应的字典
            residuals_dict[sym] = pd.Series(residuals, index=timestamps)
            betas_dict[sym] = pd.Series(betas, index=timestamps)
            std_dict[sym] = pd.Series(stds, index=timestamps)
        
        df_resid = pd.DataFrame(residuals_dict)
        return df_resid, residuals_dict, betas_dict, std_dict, timestamps


def calculate_correlations(df_resid, df_all, symbols, config, timestamps):
    """
    计算残差之间的滚动相关性，使用NumPy优化和并行处理
    
    Args:
        df_resid (pd.DataFrame): 包含所有币种残差的DataFrame
        df_all (pd.DataFrame): 包含所有币种价格数据的DataFrame
        config (dict): 配置参数字典
        
    Returns:
        pd.Series: 平均相关性Series
    """
    window_corr = 12*config['window_size']
    corr_aves = []
    corr_timestamps = []
    corr_aves_dict = {}

    for t in range(window_corr, len(df_resid)):
        df_resid_temp = df_resid.iloc[t - window_corr: t]

        # 防止 list 元素的情况
        df_float = df_resid_temp.apply(lambda col: col.map(lambda x: x[0] if isinstance(x, list) else x))

        corr_matrix = df_float.corr()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_ave = upper_triangle.stack().mean()

        corr_aves.append(corr_ave)
        corr_timestamps.append(df_all.index[t])

    # 修复 deepcopy 调用
    temp = copy.deepcopy(corr_aves)
    corr_period = config['window_size'] * 24

    for t in range(len(corr_aves)):
        if t < corr_period:
            corr_aves[t] = 1
        else:
            corr_aves[t] = np.mean(temp[t - corr_period: t])

    for sym in symbols:
        corr_aves_dict[sym] = pd.Series(corr_aves, index=timestamps[-len(corr_aves):])
        
    corr_aves_df = pd.Series(corr_aves, index=corr_timestamps, name="avg_correlation")
    return corr_aves_df, corr_aves_dict

def filter_residuals(residuals_dict, std_dict, corr_aves_dict, config):
    """
    筛选满足条件的残差
    
    Args:
        residuals_dict (dict): 包含每个币种残差的字典
        std_dict (dict): 包含每个币种标准差的字典
        corr_aves_df (pd.Series): 平均相关性Series
        config (dict): 配置参数字典
        
    Returns:
        tuple: 包含满足条件的残差字典和计数字典
    """
    residuals_positive_dict = {}
    residuals_positive_count = {}
    
    
    for coin, series in residuals_dict.items():

        start_time = series.index[0]
        

        # 遍历每个时间戳，替换为实时的std
        filtered = []
        for timestamp, row in series.items():
            residual = row[0]
            # 从 std_dict 中取出该 coin 对应 timestamp 的 std 值
            std_at_time = std_dict[coin].get(timestamp, None)
            corr_ave_at_time = corr_aves_dict[coin].get(timestamp, None)
            
            if std_at_time is None or corr_ave_at_time is None:
                continue  # 如果该时间戳没有 std，就跳过
                  
            # 判断是否满足阈值条件
            if abs(residual) > config['thres'] * std_at_time and std_at_time < config['std_max_threshold']:
                if (timestamp - start_time) >= timedelta(days=1): 
                    if corr_ave_at_time < config['corr_ave_threshold']:
                        filtered.append((timestamp, residual))

        # 保存结果
        if filtered:
            residuals_positive_dict[coin] = filtered
            residuals_positive_count[coin] = len(filtered)
    
    return residuals_positive_dict, residuals_positive_count

def calculate_returns(df_all, residuals_positive_dict, betas_dict, config):
    """
    计算每个交易信号的收益序列
    
    Args:
        df_all (pd.DataFrame): 包含所有币种价格数据的DataFrame
        residuals_positive_dict (dict): 包含满足条件的残差的字典
        betas_dict (dict): 包含每个币种beta系数的字典
        config (dict): 配置参数字典
        
    Returns:
        tuple: 包含结果DataFrame、结果列表、持仓DataFrame和持仓列表
    """
    result_list = []
    hold_list = []
    metric = 0
    return_list = []
    
    # 检查是否有交易信号
    if not residuals_positive_dict:
        print("警告: 没有找到任何交易信号，无法计算收益")
        # 返回空的DataFrame，但确保它们有正确的列
        empty_result_df = pd.DataFrame(columns=['timestamp', 'coin', 'beta', 'BTC_price', 'coin_price', 'position', 'return_series', 'return_5steps'])
        empty_hold_df = pd.DataFrame(columns=['timestamp', 'coin', 'residuals'])
        return empty_result_df, result_list, empty_hold_df, hold_list
    
    total_signals = sum(len(signals) for signals in residuals_positive_dict.values())
    print(f"开始计算收益，共有 {total_signals} 个交易信号")
    
    pbar = tqdm(total=total_signals, desc="计算收益")
    
    for coin, timestamps in residuals_positive_dict.items():
        for timestamp_, residual in timestamps:  # timestamp_ 是时间戳，residual 是残差
            
            timestamp = timestamp_

            try:
                btc_price = df_all.loc[df_all['timestamp'] == timestamp, 'BTC'].values[0]
                coin_price = df_all.loc[df_all['timestamp'] == timestamp, coin].values[0]
            except IndexError:
                print(f"警告: 在 df_all 中找不到时间戳 {timestamp} 的数据")
                pbar.update(1)
                continue

            # 获取该时间戳的 beta 值
            beta = betas_dict[coin].get(timestamp, None)
            if beta is None:
                print(f"警告: 找不到币种 {coin} 在时间戳 {timestamp} 的 beta 值")
                pbar.update(1)
                continue  # 如果没有找到 beta 值，则跳过

            # 根据 residual 的符号决定对冲组合
            position = -1 if residual > 0 else 1
            
            # 找出当前时间戳之后的所有时间戳
            future_timestamps = df_all.loc[df_all['timestamp'] > timestamp, 'timestamp']

            # 确保有足够未来数据
            if len(future_timestamps) < config['hold_time']:
                print(f"警告: 币种 {coin} 在时间戳 {timestamp} 之后没有足够的数据来计算收益")
                pbar.update(1)
                continue
            
            res_series = []
            
            stop_loss = False
            
            for i in range(config['hold_time']):
                if stop_loss:
                    total_return = temp
                else:
                    future_timestamp = future_timestamps.iloc[i]
                    future_row = df_all.loc[df_all['timestamp'] == future_timestamp]
                
                    future_btc_price = future_row['BTC'].values[0]
                    future_coin_price = future_row[coin].values[0]
        
                    ammount = 1/(coin_price*beta+btc_price) # amount个BTC，amount*beta个coin
                
                    return_btc = future_btc_price - btc_price
                    return_coin = future_coin_price - coin_price
                
                    total_return = ammount * (return_btc - beta * return_coin) * position - config['fee'] * 1
                
                if total_return < config['zs']:
                    stop_loss = True
                    temp = total_return

                # 将每个收益值包装在列表中，以匹配Pair24.py的结构
                res_series.append([total_return])
            
            # 添加到 result_list
            result_list.append({
                "timestamp": timestamp,
                "coin": coin,
                "beta": [beta],  # 将beta包装在列表中
                "BTC_price": btc_price,
                "coin_price": coin_price,
                "position": position,
                "return_series": res_series,        # 加入收益序列
                "return_5steps": [res_series[-1][0]]     # 保留最后一步的total_return作为简化结果，并包装在列表中
            })
            return_list.append(res_series[-1][0])  # 将最后一步的收益添加到返回列表

            hold_list.append({
                "timestamp": timestamp,
                "coin": coin,
                "residuals": res_series
            })
            
            pbar.update(1)
    
    pbar.close()
    
    # 检查是否有计算结果
    if not result_list:
        print("警告: 没有计算出任何收益，可能是数据问题")
        # 返回空的DataFrame，但确保它们有正确的列
        empty_result_df = pd.DataFrame(columns=['timestamp', 'coin', 'beta', 'BTC_price', 'coin_price', 'position', 'return_series', 'return_5steps'])
        empty_hold_df = pd.DataFrame(columns=['timestamp', 'coin', 'residuals'])
        return empty_result_df, result_list, empty_hold_df, hold_list
    
    # 创建DataFrame并确保timestamp列是datetime类型
    result_df = pd.DataFrame(result_list)
    if not result_df.empty and 'timestamp' in result_df.columns:
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    hold_df = pd.DataFrame(hold_list)
    if not hold_df.empty and 'timestamp' in hold_df.columns:
        hold_df['timestamp'] = pd.to_datetime(hold_df['timestamp'])
    
    print(f"收益计算完成，共处理 {len(result_list)} 个交易信号")

        # 转换为 NumPy 数组
    return_array = np.array(return_list)
    return_std = np.std(return_array)

    cumulative_returns = np.cumsum(return_array)
    historical_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - historical_max
    max_drawdown = abs(drawdown.min())

    metric_1 = np.mean(return_array)/return_std 
    metric_2 = np.mean(return_array)/max_drawdown
    metric = metric_1 * metric_2


    return result_df, result_list, hold_df, hold_list, metric