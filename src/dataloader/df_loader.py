import os
import pandas as pd

def get_symbol(filename, suffix="_klines_5m.csv"):
    """
    提取币种名
    Args:
        filename (str): 文件名
        suffix (str): 文件后缀
    Returns:
        str: 币种名
    """
    return filename.replace(suffix, "")


def load_symbol_data(file, data_dir, min_csv_length=0):
    """
    读取单个币种数据
    Args:
        file (str): 文件名
        data_dir (str): 数据目录
        min_csv_length (int): 最小 CSV 行数
    Returns:
        pd.DataFrame: 单个币种数据
    """
    symbol = get_symbol(file)
    path = os.path.join(data_dir, file)
    df = pd.read_csv(path, header=None)

    if df.shape[0]>min_csv_length:
        if df.shape[1] != 12 or df.empty:
            print(f"⚠️ 跳过 {file}（列数 ≠ 12 或为空）")
            return None
        df.columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        return df[["timestamp", "close"]].rename(columns={"close": symbol})


def merge_dataframes(dataframes, files):
    """
    合并多个数据帧
    Args:
        dataframes (list): 数据帧列表
        files (list): 文件列表
    Returns:
        pd.DataFrame: 合并后的数据帧
    """
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
    print(f"🔍 合并后的数据：{df_all.head()}")

    return df_all
