#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings("ignore")


import os
import pandas as pd
import matplotlib.pyplot as plt

# 文件夹路径
DATA_DIR = "data/5m_klines_example"

# 提取币种名的函数
def get_symbol(filename):
    return filename.replace("_klines_5m.csv", "")

# 读取单个币种数据
def load_symbol_data(file):
    symbol = get_symbol(file)
    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path, header=None)

    if df.shape[0]>10000:
        if df.shape[1] != 12 or df.empty:
            print(f"⚠️ 跳过 {file}（列数 ≠ 12 或为空）")
            return None
        df.columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        return df[["timestamp", "close"]].rename(columns={"close": symbol})
    

# 批量读取并合并
files = [f for f in os.listdir(DATA_DIR) if f.endswith("_klines_5m.csv")]
dataframes = [load_symbol_data(f) for f in files]
dataframes = [df for df in dataframes if df is not None]

if dataframes:
    df_all = dataframes[0]
    for df in dataframes[1:]:
        df_all = df_all.merge(df, on="timestamp", how="inner")
    
    # 时间戳转为时间（Binance 用的是毫秒）
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], unit="ms")
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    # 归一化每列
    symbols = [get_symbol(f) for f in files if get_symbol(f) in df_all.columns]
    for sym in symbols:
        df_all[sym] = df_all[sym] / df_all[sym].iloc[0]

    # 输出前几行
    print(df_all.head())

    # 可选：画图可视化
    df_all.set_index("timestamp")[symbols].plot(figsize=(12, 6), title="归一化后的币种价格走势")
    plt.ylabel("Normalized Price")
    plt.grid(True)
    plt.show()
else:
    print("🚫 没有成功加载任何币种数据")



df_all



n = len(df_all)
p1 = int(n * 1/2)
p2 = int(n * 3/4)

df_head = df_all.iloc[:p1].reset_index(drop=True)     # 前
df_mid  = df_all.iloc[p1:p2].reset_index(drop=True)   # 中
df_tail = df_all.iloc[p2:].reset_index(drop=True)     # 后


df_all = df_head


# 参数设置

window = 50    # 回归窗口长度
past = 288    # 回溯多少步来做回归（即用 t-past-window 到 t-past 的数据）

thres = 2    # 偏离sigma倍数
hold_time = 30   # 持有时间
window_corr = past * 3
zs = -0.03


fee = 0    # 手续费


if 'BTC' in symbols:
    symbols.remove('BTC')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df_all = df_all.sort_values('timestamp').reset_index(drop=True)

residuals_dict = {}
betas_dict = {}
std_dict = {}
corr_aves_dict = {}


# 新增部分：残差之间的滚动相关性
corr_aves = []
corr_timestamps = []

corr_aves_df = pd.Series(corr_aves, index=corr_timestamps, name="avg_correlation")

for sym in symbols:
    std_temp = 0.02
    residuals = []
    betas = []
    stds = []
    timestamps = []

    for t in range(past + window, len(df_all)):
        x_window = df_all.loc[t - past - window: t - past - 1, sym].values.reshape(-1, 1)
        y_window = df_all.loc[t - past - window: t - past - 1, 'BTC'].values.reshape(-1, 1)

        if np.any(np.isnan(x_window)) or np.any(np.isnan(y_window)):
            residuals.append(np.nan)
            timestamps.append(df_all.loc[t, 'timestamp'])
            continue

        reg = LinearRegression(fit_intercept=False).fit(x_window, y_window)
        beta = reg.coef_[0]

        x_t = df_all.loc[t, sym]
        y_t = df_all.loc[t, 'BTC']
        pred = beta * x_t
        residual = y_t - pred

        std_temp = np.sqrt(std_temp**2 * 0.99 + residual**2 * 0.01)

        residuals.append(residual)
        betas.append(beta)
        stds.append(std_temp)
        timestamps.append(df_all.loc[t, 'timestamp'])

    residuals_dict[sym] = pd.Series(residuals, index=timestamps)
    betas_dict[sym] = pd.Series(betas, index=timestamps)
    std_dict[sym] = pd.Series(stds, index=timestamps)
df_resid = pd.DataFrame(residuals_dict)

for t in range(window_corr, len(df_resid)):
    df_resid_temp = df_resid.iloc[t - window_corr: t]

    # 防止 list 元素的情况
    df_float = df_resid_temp.apply(lambda col: col.map(lambda x: x[0] if isinstance(x, list) else x))

    corr_matrix = df_float.corr()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_ave = upper_triangle.stack().mean()

    corr_aves.append(corr_ave)
    corr_timestamps.append(df_all.index[t])

for sym in symbols:
    corr_aves_dict[sym] = pd.Series(corr_aves, index=timestamps[-len(corr_aves):])
    
corr_aves_df = pd.Series(corr_aves, index=corr_timestamps, name="avg_correlation")

# valid_len = len(corr_aves)  # 统一以 corr_aves 的长度为准
# valid_index = timestamps[-valid_len:]  # 统一用 timestamps 后段
# residuals_dict[sym] = pd.Series(residuals[-valid_len:], index=valid_index)
# betas_dict[sym] = pd.Series(betas[-valid_len:], index=valid_index)
# std_dict[sym] = pd.Series(stds[-valid_len:], index=valid_index)
# corr_aves_dict[sym] = pd.Series(corr_aves, index=valid_index)


# === 保留结构不变，同时加入平均相关性计算 ===
df_resid = pd.DataFrame(residuals_dict)
df_resid.index.name = 'timestamp'

df_std = pd.DataFrame(std_dict)
df_std.index.name = 'timestamp'

df_beta = pd.DataFrame(betas_dict)
df_beta.index.name = 'timestamp'



# === 保留原始结构下的所有画图逻辑 ===
plt.figure(figsize=(14, 6))
for sym in symbols:
    plt.plot(df_resid.index, df_resid[sym], label=f"Residual(BTC - {sym})")
plt.legend()
plt.grid(True)
plt.title(f"Rolling Residuals using t-past-{window} to t-past Regression (past={past})")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
for sym in symbols:
    plt.plot(df_beta.index, df_beta[sym], label=f"Beta(BTC ~ {sym})")
plt.legend()
plt.grid(True)
plt.title(f"Rolling Betas using t-past-{window} to t-past Regression (past={past})")
plt.xlabel("Time")
plt.ylabel("Beta")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
for sym in symbols:
    plt.plot(df_std.index, df_std[sym], label=f"STD(BTC - {sym})")
plt.legend()
plt.grid(True)
plt.title(f"Rolling Residual STD")
plt.xlabel("Time")
plt.ylabel("Standard Deviation")
plt.tight_layout()
plt.show()

# 新增画图：残差之间的滚动平均相关性
plt.figure(figsize=(14, 6))
plt.plot(corr_aves_df.index, corr_aves_df.values, label="Rolling Avg Pairwise Correlation")
plt.legend()
plt.grid(True)
plt.title("Rolling Average Pairwise Correlation (288*3 window)")
plt.xlabel("Time")
plt.ylabel("Average Correlation")
plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import pandas as pd

# 每个币种残差分布
for symbol, series in residuals_dict.items():
    # 提取 float 值
    values = series.apply(lambda x: x[0])
    
    # 计算均值与标准差
    mean_val = values.mean()
    std_val = values.std()

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 均值线
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f"Mean: {mean_val:.4f}")
    
    # 标题和标签
    plt.title(f"{symbol} Residuals Histogram")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    
    # 显示均值和标准差
    plt.text(0.98, 0.95, f"μ = {mean_val:.4f}\nσ = {std_val:.4f}", 
             ha='right', va='top', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    plt.tight_layout()
    plt.show()

residuals_positive_dict = {}
residuals_positive_count = {}
from datetime import timedelta


for coin, series in residuals_dict.items():

    values = series.apply(lambda x: x[0])

    start_time = series.index[0]
    
    freeze = False
    freeze_time = 0

    # 遍历每个时间戳，替换为实时的std
    filtered = []
    for timestamp, row in series.items():
        residual = row[0]
        # 从 std_dict 中取出该 coin 对应 timestamp 的 std 值
        std_at_time = std_dict[coin].get(timestamp, None)
        corr_ave_at_time = corr_aves_dict[coin].get(timestamp, None)
        
        if std_at_time is None or corr_ave_at_time is None:
            continue  # 如果该时间戳没有 std，就跳过
            
        if corr_ave_at_time > 0.5:
            freeze = True
            freeze_time = 0
        if freeze:
            freeze_time += 1
        if freeze_time > 288:
            freeze = False
            
        # 判断是否满足阈值条件
        if abs(residual) > thres * std_at_time and std_at_time < 0.1:
            if (timestamp - start_time) >= timedelta(days=1): 
                if freeze is False:
                    filtered.append((timestamp, residual))

    # 保存结果
    if filtered:
        residuals_positive_dict[coin] = filtered
        residuals_positive_count[coin] = len(filtered)

result_list = []
hold_list = []

# 遍历 residuals_positive_dict 中的每个币种
for coin, timestamps in residuals_positive_dict.items():
    for timestamp_, residual in timestamps:  # timestamp_ 是时间戳，residual 是残差
        
        timestamp = timestamp_

        try:
            btc_price = df_all.loc[df_all['timestamp'] == timestamp, 'BTC'].values[0]
            coin_price = df_all.loc[df_all['timestamp'] == timestamp, coin].values[0]
        except IndexError:
            continue

        # 获取该时间戳的 beta 值
        beta = betas_dict[coin].get(timestamp, None)
        if beta is None:
            continue  # 如果没有找到 beta 值，则跳过

        # 根据 residual 的符号决定对冲组合
        position = -1 if residual > 0 else 1
        
        # 找出当前时间戳之后的所有时间戳
        future_timestamps = df_all.loc[df_all['timestamp'] > timestamp, 'timestamp']

        # 确保有足够未来数据
        if len(future_timestamps) < hold_time:
            continue
        
        res_series = []
        
        stop_loss = False
        
        for i in range(hold_time):
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
            
                total_return = ammount * (return_btc - beta * return_coin) * position - fee * 1
            
            if total_return < zs:
                stop_loss = True
                temp = total_return

            res_series.append(total_return)
        
        res_series = np.array(res_series)
        
        # 添加到 result_list
        result_list.append({
            "timestamp": timestamp,
            "coin": coin,
            "beta": beta,
            "BTC_price": btc_price,
            "coin_price": coin_price,
            "position": position,
            "return_series": res_series,        # 加入收益序列
            "return_5steps": res_series[-1]     # 保留你原本最后一步的total_return作为简化结果
        })

        hold_list.append({
            "timestamp": timestamp,
            "coin": coin,
            "residuals": res_series
        })

# 转换为 DataFrame
result_df = pd.DataFrame(result_list)
hold_df = pd.DataFrame(hold_list)

# 输出查看
print(result_df.head())
print(hold_df.head())



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 确保 timestamp 为 datetime 类型
result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])

# 将列表类型的 beta 和 return_5steps 转换为 float
result_df['beta'] = result_df['beta'].apply(lambda x: x[0] if isinstance(x, list) else x)
result_df['return_5steps'] = result_df['return_5steps'].apply(lambda x: x[0] if isinstance(x, list) else x)

# 按时间排序
result_df = result_df.sort_values('timestamp')

# 计算累计收益
result_df['cumulative_return'] = result_df['return_5steps'].cumsum()

# ---------- 📈 图一：总累计收益 ----------
plt.figure(figsize=(12, 6))
plt.plot(result_df['timestamp'], result_df['cumulative_return'], label='Cumulative Return', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Return Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 假设 result_list 中每个 item 的 "return_series" 长度相同
all_series = np.array([item["return_series"] for item in result_list])  # shape: (num_samples, hold_time)
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
plt.show()


all_series.shape



import pandas as pd
import matplotlib.pyplot as plt

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
plt.show()

