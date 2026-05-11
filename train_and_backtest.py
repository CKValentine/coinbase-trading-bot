"""
train_and_backtest.py  —  v2

Per-pair tuned training with enriched features:
  Core OHLCV:   log_return, atr_ratio, volume_ratio, rsi, rsi_slope, macd_diff
  Trend:        bb_position, sma50_slope, price_vs_sma200
  Time:         hour_sin, hour_cos, dow_sin, dow_cos
  Macro:        fear_greed, dxy_return, gold_return  (where available)
"""

import pandas as pd
import numpy as np
import joblib
import os
import requests
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from xgboost import XGBClassifier
from ta import momentum, trend, volatility

# ---------------------------------------------------------------------------
# Per-pair configuration
# ---------------------------------------------------------------------------
PAIR_CONFIG = {
    'BTC-USD':  dict(csv='btc_usd_hourly.csv',  target=0.005, horizon=8, threshold=0.53, label='BTC'),
    'ETH-USD':  dict(csv='eth_usd_hourly.csv',  target=0.010, horizon=8, threshold=0.58, label='ETH'),
    'SOL-USD':  dict(csv='sol_usd_hourly.csv',  target=0.015, horizon=8, threshold=0.65, label='SOL'),
    'XRP-USD':  dict(csv='xrp_usd_hourly.csv',  target=0.010, horizon=8, threshold=0.58, label='XRP'),
    'ADA-USD':  dict(csv='ada_usd_hourly.csv',  target=0.010, horizon=8, threshold=0.58, label='ADA'),
    'LINK-USD': dict(csv='link_usd_hourly.csv', target=0.010, horizon=8, threshold=0.58, label='LINK'),
}

BASE_FEATURES = [
    'log_return', 'atr_ratio', 'volume_ratio',
    'rsi', 'rsi_slope', 'macd_diff',
    'bb_position', 'sma50_slope', 'price_vs_sma200',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

MACRO_FEATURES = ['fear_greed', 'dxy_return', 'gold_return']
FEE = 0.002


# ---------------------------------------------------------------------------
# External data loaders
# ---------------------------------------------------------------------------

def load_fear_greed():
    try:
        r = requests.get('https://api.alternative.me/fng/?limit=0&format=json', timeout=15)
        data = r.json()['data']
        df = pd.DataFrame(data)[['value', 'timestamp']]
        df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s').dt.date
        df['fear_greed'] = df['value'].astype(float) / 100.0
        df = df[['date', 'fear_greed']].set_index('date')
        print(f"  Fear & Greed: {len(df)} days loaded")
        return df
    except Exception as e:
        print(f"  Fear & Greed unavailable: {e}")
        return pd.DataFrame()


def load_macro():
    try:
        # .squeeze() converts yfinance MultiIndex columns to a plain Series
        dxy  = yf.download('DX-Y.NYB', start='2016-01-01', progress=False)['Close'].squeeze()
        gold = yf.download('GC=F',     start='2016-01-01', progress=False)['Close'].squeeze()
        macro = pd.DataFrame({'dxy_return': dxy.pct_change(), 'gold_return': gold.pct_change()})
        macro.index = pd.to_datetime(macro.index).date
        macro = macro.dropna()
        print(f"  Macro (DXY + Gold): {len(macro)} days loaded")
        return macro
    except Exception as e:
        print(f"  Macro data unavailable: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df, fg, macro):
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Core OHLCV (leak-proof: all shifted 1 bar)
    df['log_return']   = np.log(df['close'] / df['close'].shift(1)).shift(1)

    atr = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['atr_ratio']    = (atr / df['close']).shift(1)
    df['volume_ratio'] = (df['volume'] / df['volume'].rolling(14).mean()).shift(1)

    rsi             = momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi']       = rsi.shift(1)
    df['rsi_slope'] = (rsi - rsi.shift(1)).shift(1)
    df['macd_diff'] = trend.MACD(df['close']).macd_diff().shift(1)

    # Trend features
    bb = volatility.BollingerBands(df['close'], window=20)
    df['bb_position'] = ((df['close'] - bb.bollinger_lband()) /
                         (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)).shift(1)

    sma50 = df['close'].rolling(50).mean()
    df['sma50_slope'] = (sma50 / sma50.shift(5) - 1).shift(1)

    sma200 = df['close'].rolling(200).mean()
    df['price_vs_sma200'] = (df['close'] / sma200 - 1).shift(1)

    # Time features (cyclical)
    df.index = pd.to_datetime(df.index)
    hour = df.index.hour
    dow  = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin']  = np.sin(2 * np.pi * dow  / 7)
    df['dow_cos']  = np.cos(2 * np.pi * dow  / 7)

    # Macro (daily, forward-filled)
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
# Train + backtest one pair
# ---------------------------------------------------------------------------

def train_and_backtest(pair, cfg, fg, macro):
    print(f"\n{'='*45}")
    print(f"  {pair}")
    print(f"{'='*45}")

    if not os.path.exists(cfg['csv']):
        print(f"  SKIP — {cfg['csv']} not found")
        return

    df = pd.read_csv(cfg['csv'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df = engineer_features(df, fg, macro)

    available_macro = [f for f in MACRO_FEATURES if f in df.columns]
    features = BASE_FEATURES + available_macro
    print(f"  Features ({len(features)}): {features}")

    df['future_return'] = df['close'].shift(-cfg['horizon']) / df['close'] - 1
    df['label']         = (df['future_return'] > cfg['target']).astype(int)
    df.dropna(inplace=True)

    print(f"  Rows: {len(df):,}  |  Positive labels: {df['label'].mean():.1%}")

    X, y = df[features], df['label']
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        tree_method='hist', device='cpu',
        n_estimators=600, learning_rate=0.02,
        max_depth=4, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10,
        eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train)

    # Save model + feature list together
    model_file = f"model_{pair.replace('-', '_')}.pkl"
    joblib.dump({'model': model, 'features': features}, model_file)
    print(f"  Saved → {model_file}")

    # Backtest on test set
    test_df = df.iloc[split:].copy()
    test_df['prob']   = model.predict_proba(X_test)[:, 1]
    test_df['signal'] = (test_df['prob'] > cfg['threshold']).astype(int).shift(1)
    test_df.dropna(inplace=True)

    test_df['trade_return'] = np.where(
        test_df['signal'] == 1,
        np.maximum(test_df['future_return'], -0.02) - FEE,
        0
    )

    test_df['equity'] = (1 + test_df['trade_return']).cumprod()
    rolling_max  = test_df['equity'].cummax()
    max_drawdown = ((test_df['equity'] - rolling_max) / rolling_max).min()
    total_return = test_df['equity'].iloc[-1] - 1
    trades       = int(test_df['signal'].sum())
    win_rate     = (test_df[test_df['signal']==1]['trade_return'] > 0).mean() if trades > 0 else 0

    print(f"\n  Backtest ({cfg['threshold']:.0%} threshold, +{cfg['target']:.1%} target):")
    print(f"  Trades:       {trades}")
    print(f"  Win rate:     {win_rate:.2%}")
    print(f"  Net profit:   {total_return:.2%}")
    print(f"  Max drawdown: {max_drawdown:.2%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading external data...")
    fg    = load_fear_greed()
    macro = load_macro()

    print(f"\nTraining {len(PAIR_CONFIG)} models...")
    for pair, cfg in PAIR_CONFIG.items():
        train_and_backtest(pair, cfg, fg, macro)

    print(f"\n{'='*45}")
    print("Done. All models saved.")
    print("NOTE: app.py needs updating — models now saved as")
    print("dicts with 'model' and 'features' keys.")
    print(f"{'='*45}\n")
