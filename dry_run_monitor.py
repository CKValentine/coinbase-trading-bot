import pandas as pd
import numpy as np
import joblib
import time
import requests
import os
from datetime import datetime, timedelta
from ta import momentum, trend, volatility

# === CONFIGURATION ===
SYMBOL = 'BTC-USD'
MODEL_FILE = 'model_BTC_USD.pkl'
THRESHOLD = 0.60
LOG_FILE = "trade_log.csv"

def get_live_coinbase_data(product_id='BTC-USD'):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {'granularity': 3600} 
    response = requests.get(url, params=params)
    df = pd.DataFrame(response.json(), columns=['ts', 'low', 'high', 'open', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def grade_past_predictions(current_price):
    """Checks the log for trades from 8 hours ago and calculates their success."""
    if not os.path.exists(LOG_FILE): return
    
    df_log = pd.read_csv(LOG_FILE)
    df_log['ts'] = pd.to_datetime(df_log['ts'])
    
    # Target time is roughly 8 hours ago
    eight_hours_ago = datetime.now() - timedelta(hours=8)
    
    # Find logs from 8 hours ago that haven't been 'graded' yet
    mask = (df_log['ts'] <= eight_hours_ago) & (df_log['outcome_pct'].isna())
    
    for idx, row in df_log[mask].iterrows():
        entry_price = row['price']
        gain_loss = (current_price - entry_price) / entry_price
        df_log.at[idx, 'outcome_pct'] = f"{gain_loss:.2%}"
        print(f"✅ [GRADED] Signal from {row['ts'].strftime('%H:%M')} would be: {gain_loss:.2%}")
    
    df_log.to_csv(LOG_FILE, index=False)

def generate_signal():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Polling...")
    
    try:
        df = get_live_coinbase_data(SYMBOL)
        model = joblib.load(MODEL_FILE)
        
        # 1. Indicators + SMA Filter
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['atr_ratio'] = volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range() / df['close']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=14).mean()
        df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(1)
        df['macd_diff'] = trend.MACD(df['close']).macd_diff()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        features_list = ['log_return', 'atr_ratio', 'volume_ratio', 'rsi', 'rsi_slope', 'macd_diff']
        latest_row = df[features_list].iloc[-1:]
        
        prob = model.predict_proba(latest_row)[:, 1][0]
        price = df['close'].iloc[-1]
        sma_200 = df['sma_200'].iloc[-1]
        
        # 2. Outcome Grading
        grade_past_predictions(price)

        # 3. Decision Logic
        action = "HOLD"
        if prob > THRESHOLD:
            if price > sma_200:
                print(f"🚀 [SIGNAL] BUY ALERT! Confidence: {prob:.2%}")
                action = "BUY"
            else:
                print(f"⚠️ [FILTERED] High confidence ({prob:.2%}) but below 200-SMA.")
                action = "FILTERED"
        else:
            print(f"Price: ${price:,.2f} | Confidence: {prob:.2%}")

        # 4. Log to CSV
        log_entry = pd.DataFrame([{
            "ts": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "price": price,
            "conf": prob,
            "act": action,
            "outcome_pct": np.nan # To be filled in 8 hours
        }])
        log_entry.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    while True:
        generate_signal()
        current_hour = datetime.now().hour
        while datetime.now().hour == current_hour:
            time.sleep(30)