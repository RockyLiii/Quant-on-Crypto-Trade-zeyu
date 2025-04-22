# fetch_data.py
# 从 OKX API 获取多个币种的 K 线数据并单独保存
# 依赖: requests, pandas, argparse (请先安装: pip install requests pandas)
#usage: python fetch_data.py --update 主要用这个就行，间断时间不超过5天即可，可以考虑挂在后台
import requests
import pandas as pd
import time
import os
import json
import sys
import argparse

# --- 配置 ---
# 要获取的币种列表
SYMBOLS = ['CAT','BTC','DEGEN','DOGE','BONK','ELON','NOT','PEPE','PNUT','SHIB','SLERF','TRUMP','WIF','XRP','ETH','SOL','TRX','SUI','XLM','LINK','MEW','ONDO','ADA','AAVE','AVAX']

# OKX API 相关
API_ENDPOINT = "https://www.okx.com/api/v5/market/candles"
BAR = "5m" # K线周期
LIMIT = 300 # 每次 API 请求获取的最大 K 线条数
MAX_PAGES = 100 # 每个币种最多获取的页数
REQUEST_DELAY = 0.2 # 请求之间的延迟（秒）
REQUEST_TIMEOUT = 15 # 请求超时时间（秒）

# 数据保存目录
OUTPUT_DIR = "okx_data"

# --- 函数定义 ---
def fetch_okx_5m_data(symbol, bar=BAR, max_pages=MAX_PAGES, limit=LIMIT):
    """拉取单个币种的OKX K线数据，支持翻页。"""
    inst_id = f"{symbol}-USDT"
    all_data = []
    after = None  # 用于翻页的时间戳
    print(f"  Fetching {inst_id} ({bar}, max {max_pages} pages)...", end='')
    page_count = 0
    
    for _ in range(max_pages):
        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": limit
        }
        if after:
            params["after"] = after
        
        try:
            response = requests.get(API_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status() # 检查 HTTP 错误
            data = response.json()

            # 检查业务错误码
            if data.get("code") == "0":
                candles = data.get("data", [])
                if not candles:
                    break # 没有更多数据了
                # 检查第一条数据的列数是否为9 (只检查一次)
                if page_count == 0 and len(candles[0]) != 9:
                     print(f"\n  ❌ Error: API for {symbol} did not return 9 columns as expected. Got {len(candles[0])}. Stopping fetch.")
                     return pd.DataFrame() # 返回空 DF
                all_data.extend(candles)
                # OKX 返回的时间戳是升序的，after需要用最新的时间戳(列表最后一个)
                after = candles[-1][0] 
                page_count += 1
                time.sleep(REQUEST_DELAY) # 尊重速率限制
            else:
                print(f"\n  ❌ OKX API Error for {symbol}: {data.get('msg')} (Code: {data.get('code')})")
                break # API返回错误，停止该币种的获取

        except requests.exceptions.Timeout:
            print(f"\n  ❌ Timeout during request for {symbol}. Stopping fetch.")
            break 
        except requests.exceptions.RequestException as e:
            print(f"\n  ❌ Request failed for {symbol}: {e}. Stopping fetch.")
            break
        except json.JSONDecodeError:
            print(f"\n  ❌ JSON Decode Error for {symbol}. Stopping fetch.")
            break
        except Exception as e:
             print(f"\n  ❌ An unexpected error occurred for {symbol}: {e}. Stopping fetch.")
             break
             
    print(f" Done ({page_count} pages, {len(all_data)} candles).")

    if not all_data:
        return pd.DataFrame() # 如果获取失败或无数据，返回空DataFrame

    # 定义完整的9列名称 (根据OKX V5文档)
    columns = [
        "timestamp", "open", "high", "low", "close", "volume_base",
        "volume_quote", "volume_quote_2", "confirm" # 假设第8列是 volume_quote 的另一种表示
    ]
    # 确保数据有9列才创建DataFrame
    if len(all_data[0]) != 9:
         print(f"\n  ❌ Error: Data for {symbol} does not have 9 columns after fetch. Cannot create DataFrame.")
         return pd.DataFrame()
         
    df = pd.DataFrame(all_data, columns=columns)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    
    # 转换数值列 (包括新增的 volume_quote_2)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume_base', 'volume_quote', 'volume_quote_2', 'confirm']
    for col in numeric_cols:
        if col in df.columns: # Check if column exists before converting
             df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 通常API返回是时间降序，这里获取后按时间升序排
    df = df.sort_values("timestamp").reset_index(drop=True)
    # 可选：去重，虽然分页理论上不应重复，但保险起见
    df = df.drop_duplicates(subset=['timestamp'], keep='first') 
    return df

# --- 命令行参数 ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='从OKX获取K线数据')
    parser.add_argument('--update', action='store_true', 
                        help='更新模式：添加新数据到现有文件并移除未确认(confirm=0)的K线')
    return parser.parse_args()

# --- 主程序 ---
if __name__ == "__main__":
    args = parse_arguments()
    update_mode = args.update
    
    if update_mode:
        print(f"Starting data update for {len(SYMBOLS)} symbols (update mode)...")
    else:
        print(f"Starting data fetch for {len(SYMBOLS)} symbols (new data mode)...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True) # 创建输出目录

    total_symbols_processed = 0
    total_symbols_saved = 0

    for symbol in SYMBOLS:
        print(f"\n📥 Processing symbol: {symbol}")
        df_new = fetch_okx_5m_data(symbol, bar=BAR, max_pages=MAX_PAGES, limit=LIMIT)
        total_symbols_processed += 1

        if not df_new.empty:
            filename = os.path.join(OUTPUT_DIR, f"{symbol}-USDT_{BAR}_raw.csv")
            
            if update_mode and os.path.exists(filename):
                try:
                    # 读取现有数据
                    df_existing = pd.read_csv(filename)
                    
                    # 转换时间戳列为datetime以便合并
                    if 'timestamp' in df_existing.columns:
                        df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
                    
                    # 合并新旧数据
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    
                    # 去重
                    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
                    
                    # 移除所有confirm=0的行
                    df_filtered = df_combined[df_combined['confirm'] != 0]
                    
                    # 按时间排序
                    df_filtered = df_filtered.sort_values("timestamp").reset_index(drop=True)
                    
                    # 保存更新后的数据
                    df_filtered.to_csv(filename, index=False, encoding='utf-8')
                    print(f"  💾 Updated data saved to {filename}")
                    print(f"      原始行数: {len(df_existing)}, 新增行数: {len(df_new)}")
                    print(f"      合并后: {len(df_combined)}, 过滤后: {len(df_filtered)}")
                    total_symbols_saved += 1
                    
                except Exception as e:
                    print(f"  ❌ Failed to update data for {symbol}: {e}", file=sys.stderr)
            else:
                # 新模式或文件不存在，直接保存新数据
                try:
                    df_new.to_csv(filename, index=False, encoding='utf-8')
                    print(f"  💾 Raw data saved to {filename} ({len(df_new)} rows)")
                    total_symbols_saved += 1
                except Exception as e:
                    print(f"  ❌ Failed to save raw data for {symbol}: {e}", file=sys.stderr)
        else:
             print(f"  ⚠️ No data fetched or processed for {symbol}.")

    print(f"\n--- {'Update' if update_mode else 'Fetch'} complete ---")
    print(f"Processed: {total_symbols_processed} symbols")
    print(f"Successfully saved: {total_symbols_saved} files to '{OUTPUT_DIR}'")
