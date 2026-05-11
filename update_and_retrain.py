"""
update_and_retrain.py

1. Fetches new hourly candles for all pairs from last CSV timestamp to now
2. Appends to existing CSVs
3. Retrains all 6 models on the full updated dataset
4. Saves new .pkl files ready for app.py to load

Run manually:   python3 update_and_retrain.py
Scheduled via:  LaunchAgent (com.tradingbot.update.plist) — runs every Sunday at 2am
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from datetime import datetime
from coinbase.rest import RESTClient
from dotenv import load_dotenv
from xgboost import XGBClassifier
from ta import momentum, trend, volatility

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAIRS = [
    ('BTC-USD',  'btc_usd_hourly.csv'),
    ('ETH-USD',  'eth_usd_hourly.csv'),
    ('SOL-USD',  'sol_usd_hourly.csv'),
    ('XRP-USD',  'xrp_usd_hourly.csv'),
    ('ADA-USD',  'ada_usd_hourly.csv'),
    ('LINK-USD', 'link_usd_hourly.csv'),
]

PAIR_CONFIG = {
    'BTC-USD':  dict(target=0.005, horizon=8, threshold=0.53),
    'ETH-USD':  dict(target=0.010, horizon=8, threshold=0.58),
    'SOL-USD':  dict(target=0.010, horizon=8, threshold=0.58),
    'XRP-USD':  dict(target=0.010, horizon=8, threshold=0.58),
    'ADA-USD':  dict(target=0.010, horizon=8, threshold=0.58),
    'LINK-USD': dict(target=0.010, horizon=8, threshold=0.58),
}

FEATURES = [
    'log_return', 'atr_ratio', 'volume_ratio', 'rsi', 'rsi_slope', 'macd_diff',
    'bb_position', 'sma50_slope', 'price_vs_sma200',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'fear_greed', 'dxy_return', 'gold_return',
]

FEE       = 0.002
STOP_LOSS = -0.02


# ---------------------------------------------------------------------------
# Step 1: Update CSVs
# ---------------------------------------------------------------------------

def fetch_candles_range(product_id, start_ts, end_ts, batch=300):
    """Fetch hourly candles in batches from Coinbase public API."""
    client = RESTClient()
    all_candles = []
    current = start_ts

    while current < end_ts:
        batch_end = min(current + batch * 3600, end_ts)
        try:
            response = client.get_public_candles(
                product_id=product_id,
                start=str(current),
                end=str(batch_end),
                granularity='ONE_HOUR'
            )
            candles = response['candles']
            for c in candles:
                all_candles.append({
                    'start':     c['start'],
                    'low':       c['low'],
                    'high':      c['high'],
                    'open':      c['open'],
                    'close':     c['close'],
                    'volume':    c['volume'],
                    'timestamp': datetime.utcfromtimestamp(int(c['start'])).strftime('%Y-%m-%d %H:%M:%S'),
                })
            print(f"  Fetched {len(candles)} candles up to {datetime.utcfromtimestamp(batch_end)}")
        except Exception as e:
            print(f"  Error fetching {product_id}: {e}")

        current = batch_end + 1
        time.sleep(0.5)

    return all_candles


def update_csv(pair, csv_file):
    print(f"\n[DATA] {pair}")

    if not os.path.exists(csv_file):
        print(f"  CSV not found: {csv_file} — skipping")
        return False

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    last_ts = int(df['timestamp'].max().timestamp())
    now_ts  = int(time.time()) - 3600  # exclude current incomplete candle

    hours_missing = (now_ts - last_ts) // 3600
    print(f"  Last record: {df['timestamp'].max()}  ({hours_missing} hours to fetch)")

    if hours_missing < 2:
        print(f"  Already up to date")
        return True

    new_candles = fetch_candles_range(pair, last_ts + 3600, now_ts)

    if not new_candles:
        print(f"  No new candles fetched")
        return True

    df_new = pd.DataFrame(new_candles)
    for col in ['start', 'low', 'high', 'open', 'close', 'volume']:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined.drop_duplicates(subset='timestamp', inplace=True)
    df_combined.sort_values('timestamp', inplace=True)
    df_combined.to_csv(csv_file, index=False)

    print(f"  Added {len(df_new)} rows → total {len(df_combined):,} rows")
    return True


# ---------------------------------------------------------------------------
# Step 2: Load external data
# ---------------------------------------------------------------------------

def load_fear_greed():
    try:
        r = requests.get('https://api.alternative.me/fng/?limit=0&format=json', timeout=15)
        data = r.json()['data']
        df = pd.DataFrame(data)[['value', 'timestamp']]
        df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s').dt.date
        df['fear_greed'] = df['value'].astype(float) / 100.0
        df = df[['date', 'fear_greed']].set_index('date')
        print(f"  Fear & Greed: {len(df)} days")
        return df
    except Exception as e:
        print(f"  Fear & Greed failed: {e}")
        return pd.DataFrame()


def load_macro():
    try:
        dxy  = yf.download('DX-Y.NYB', start='2016-01-01', progress=False)['Close'].squeeze()
        gold = yf.download('GC=F',     start='2016-01-01', progress=False)['Close'].squeeze()
        macro = pd.DataFrame({'dxy_return': dxy.pct_change(), 'gold_return': gold.pct_change()})
        macro.index = pd.to_datetime(macro.index).date
        macro = macro.dropna()
        print(f"  Macro (DXY + Gold): {len(macro)} days")
        return macro
    except Exception as e:
        print(f"  Macro failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 3: Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df, fg, macro):
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['log_return']   = np.log(df['close'] / df['close'].shift(1)).shift(1)
    atr = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['atr_ratio']    = (atr / df['close']).shift(1)
    df['volume_ratio'] = (df['volume'] / df['volume'].rolling(14).mean()).shift(1)
    rsi             = momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi']       = rsi.shift(1)
    df['rsi_slope'] = (rsi - rsi.shift(1)).shift(1)
    df['macd_diff'] = trend.MACD(df['close']).macd_diff().shift(1)

    bb = volatility.BollingerBands(df['close'], window=20)
    df['bb_position'] = ((df['close'] - bb.bollinger_lband()) /
                         (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)).shift(1)
    sma50 = df['close'].rolling(50).mean()
    df['sma50_slope'] = (sma50 / sma50.shift(5) - 1).shift(1)
    sma200 = df['close'].rolling(200).mean()
    df['price_vs_sma200'] = (df['close'] / sma200 - 1).shift(1)

    df.index = pd.to_datetime(df.index)
    hour = df.index.hour
    dow  = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin']  = np.sin(2 * np.pi * dow  / 7)
    df['dow_cos']  = np.cos(2 * np.pi * dow  / 7)

    df['date'] = df.index.date
    if not fg.empty:
        df['fear_greed'] = df['date'].map(fg['fear_greed'].to_dict())
        df['fear_greed'] = df['fear_greed'].ffill()
    if not macro.empty:
        for col in ['dxy_return', 'gold_return']:
            df[col] = df['date'].map(macro[col].to_dict())
            df[col] = df[col].ffill()
    df.drop(columns=['date'], inplace=True)
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Step 4: Train + backtest one pair
# ---------------------------------------------------------------------------

def train_pair(pair, csv_file, cfg, fg, macro):
    print(f"\n[TRAIN] {pair}")

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df = engineer_features(df, fg, macro)

    df['future_return'] = df['close'].shift(-cfg['horizon']) / df['close'] - 1
    df['label']         = (df['future_return'] > cfg['target']).astype(int)
    df.dropna(inplace=True)

    print(f"  Rows: {len(df):,}  Positive: {df['label'].mean():.1%}")

    X, y = df[FEATURES], df['label']
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train         = y.iloc[:split]

    model = XGBClassifier(
        tree_method='hist', device='cpu',
        n_estimators=600, learning_rate=0.02,
        max_depth=4, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10,
        eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train)

    # Quick backtest
    test_df = df.iloc[split:].copy()
    test_df['prob']   = model.predict_proba(X_test)[:, 1]
    test_df['signal'] = (test_df['prob'] > cfg['threshold']).astype(int).shift(1)
    test_df.dropna(inplace=True)

    test_df['trade_return'] = np.where(
        test_df['signal'] == 1,
        np.maximum(test_df['future_return'], STOP_LOSS) - FEE, 0
    )
    test_df['equity'] = (1 + test_df['trade_return']).cumprod()
    rolling_max  = test_df['equity'].cummax()
    max_dd       = ((test_df['equity'] - rolling_max) / rolling_max).min()
    total_return = test_df['equity'].iloc[-1] - 1
    trades       = int(test_df['signal'].sum())
    win_rate     = (test_df[test_df['signal']==1]['trade_return'] > 0).mean() if trades > 0 else 0

    print(f"  Trades: {trades}  Win: {win_rate:.1%}  Profit: {total_return:.1%}  Drawdown: {max_dd:.1%}")

    model_file = f"model_{pair.replace('-', '_')}.pkl"
    joblib.dump({'model': model, 'features': FEATURES}, model_file)
    print(f"  Saved → {model_file}")

    return {'pair': pair, 'trades': trades, 'win_rate': win_rate,
            'profit': total_return, 'drawdown': max_dd}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start = datetime.now()
    print(f"{'='*50}")
    print(f"Update & Retrain — {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    # Step 1: Update CSVs
    print("\n--- Updating data ---")
    for pair, csv_file in PAIRS:
        update_csv(pair, csv_file)

    # Step 2: Load external data
    print("\n--- Loading external data ---")
    fg    = load_fear_greed()
    macro = load_macro()

    # Step 3: Retrain all models
    print("\n--- Retraining models ---")
    results = []
    for pair, csv_file in PAIRS:
        result = train_pair(pair, csv_file, PAIR_CONFIG[pair], fg, macro)
        results.append(result)

    # Summary
    elapsed = (datetime.now() - start).seconds
    print(f"\n{'='*50}")
    print(f"Done in {elapsed}s — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")
    print(f"{'Pair':>10} {'Trades':>8} {'Win Rate':>10} {'Profit':>10} {'Drawdown':>12}")
    for r in results:
        print(f"{r['pair']:>10} {r['trades']:>8} {r['win_rate']:>10.1%} {r['profit']:>10.1%} {r['drawdown']:>12.1%}")
    print(f"{'='*50}\n")

    # Telegram summary
    try:
        token   = os.getenv('TELEGRAM_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if token and chat_id:
            lines = ["📊 <b>Weekly retrain complete</b>"]
            for r in results:
                emoji = "✅" if r['win_rate'] >= 0.60 else "⚠️"
                lines.append(
                    f"{emoji} {r['pair']}: {r['trades']} trades, "
                    f"{r['win_rate']:.0%} win, {r['profit']:+.0%} profit"
                )
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "HTML"},
                timeout=5
            )
    except Exception as e:
        print(f"Telegram notify failed: {e}")
