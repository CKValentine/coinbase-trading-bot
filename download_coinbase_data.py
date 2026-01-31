import csv
import time
from datetime import datetime
from coinbase.rest import RESTClient
import requests  # For CoinGecko and CoinMetrics
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Coinbase client without keys (for public data)
client = RESTClient()

# CoinMetrics API key from .env
coinmetrics_key = os.getenv("COINMETRICS_API_KEY")

def timestamp_to_unix(dt_str):
    """Convert YYYY-MM-DD string to Unix timestamp."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d")
    return int(dt.timestamp())

def fetch_candles(product_id, granularity, start, end, batch_size=300):
    data = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + (batch_size * get_granularity_seconds(granularity)), end)
        
        try:
            response = client.get_public_candles(
                product_id=product_id,
                start=str(current_start),
                end=str(current_end),
                granularity=granularity
            )
            candles = response['candles']
            data.extend(candles)
            print(f"Fetched {len(candles)} candles for {product_id} from {datetime.fromtimestamp(current_start)} to {datetime.fromtimestamp(current_end)}")
            
            current_start = current_end + 1
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching data for {product_id}: {e}")
            break
    
    return data

def get_granularity_seconds(granularity):
    mapping = {
        'ONE_HOUR': 3600,
    }
    return mapping.get(granularity, 86400)

def fetch_coingecko_metrics(asset, start_ts, end_ts, metrics):
    url = f"https://api.coingecko.com/api/v3/coins/{asset}/market_chart/range?vs_currency=usd&from={start_ts}&to={end_ts}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        result = {}
        for i, ts in enumerate(data['prices']):
            date_str = datetime.fromtimestamp(ts[0]/1000).strftime('%Y-%m-%d')
            result[date_str] = {
                'price': ts[1],
                'market_cap': data['market_caps'][i][1],
                'total_volume': data['total_volumes'][i][1]
            }
        return result
    else:
        print(f"Error fetching CoinGecko for {asset}: {response.text}")
        return {}

def fetch_coinmetrics_metrics(asset, start_date, end_date, metrics):
    url = f"https://api.coinmetrics.io/v4/timeseries/asset-metrics?api_key={coinmetrics_key}&assets={asset}&metrics={','.join(metrics)}&frequency=1d&start_time={start_date}&end_time={end_date}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['data']
        result = {}
        for d in data:
            time_str = d['time'].split('T')[0]
            result[time_str] = {m: d.get(m) for m in metrics}
        return result
    else:
        print(f"Error fetching CoinMetrics for {asset}: {response.text}")
        return {}

def add_advanced_features(df, product_id):
    asset_map = {
        'BTC-USD': 'bitcoin',
        'ETH-USD': 'ethereum',
        'SOL-USD': 'solana',
        'XRP-USD': 'xrp',
        'ADA-USD': 'cardano',
        'LINK-USD': 'chainlink'
    }
    asset = asset_map.get(product_id)
    if asset:
        start_ts = int(df['timestamp'].min().timestamp())
        end_ts = int(df['timestamp'].max().timestamp())
        coingecko_data = fetch_coingecko_metrics(asset, start_ts, end_ts, [])
        coinmetrics_data = fetch_coinmetrics_metrics(asset, df['timestamp'].min().strftime('%Y-%m-%d'), df['timestamp'].max().strftime('%Y-%m-%d'), ['AdrActCnt', 'TxCnt', 'SplyCur', 'FeeTotNtv', 'HashRate', 'IssTotNtv'])
        
        df['date'] = df['timestamp'].dt.date.astype(str)
        
        # CoinGecko metrics
        for m in ['market_cap', 'total_volume']:
            df[m] = df['date'].map({k: v.get(m) for k, v in coingecko_data.items() if v.get(m)})
            df[m] = df[m].ffill()
        
        # CoinMetrics metrics
        for m in ['AdrActCnt', 'TxCnt', 'SplyCur', 'FeeTotNtv', 'HashRate', 'IssTotNtv']:
            df[m] = df['date'].map({k: v.get(m) for k, v in coinmetrics_data.items() if v.get(m)})
            df[m] = df[m].ffill()
        
        df.drop('date', axis=1, inplace=True)
    
    return df

def save_to_csv(df, filename, mode='w'):
    if mode == 'a' and os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    print(f"Data saved to {filename} in mode {mode}")

# Configuration
pairs = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'LINK-USD']
granularity = 'ONE_HOUR'
start_date = '2016-01-01' # 10 years back
end_date = '2026-01-30' # Up to today

# Convert dates to Unix timestamps
start_timestamp = timestamp_to_unix(start_date)
end_timestamp = timestamp_to_unix(end_date) + 86399

# Fetch and save for each pair
for product_id in pairs:
    print(f"\nStarting download for {product_id}...")
    output_file = f"{product_id.lower().replace('-', '_')}_hourly.csv"

    # Check if CSV exists; if yes, append from last timestamp
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        if 'timestamp' in df_existing.columns:
            df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
            last_ts = int(df_existing['timestamp'].max().timestamp())
            print(f"Existing data found for {product_id}. Resuming from {datetime.fromtimestamp(last_ts)}")
        else:
            last_ts = start_timestamp
            print(f"Existing data found but no timestamp column. Starting from beginning.")
        start_timestamp_local = last_ts + get_granularity_seconds(granularity) # Start from next candle
        mode = 'a'
    else:
        start_timestamp_local = start_timestamp
        mode = 'w'
        print(f"No existing data for {product_id}. Starting from beginning.")

    candles_data = fetch_candles(product_id, granularity, start_timestamp_local, end_timestamp)
    if candles_data:
        df_new = pd.DataFrame(candles_data)
        df_new['timestamp'] = pd.to_datetime(df_new['start'], unit='s')
        df_new = add_advanced_features(df_new, product_id) # Add metrics
        save_to_csv(df_new, output_file, mode)
    else:
        print(f"No new data fetched for {product_id}. Skipping save.")
    print(f"Completed {product_id}\n")