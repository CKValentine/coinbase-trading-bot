import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from ta import momentum, trend, volatility

def prepare_data(csv_file):
    if not os.path.exists(csv_file):
        return None
    
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # --- 1. LEAK-PROOF FEATURES ---
    # We shift EVERY indicator by 1. 
    # This means at 1:00PM, the model ONLY sees data finalized at 12:00PM.
    df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi().shift(1)
    df['macd_diff'] = trend.MACD(df['close']).macd_diff().shift(1)
    
    # ATR Ratio (Volatility)
    atr = volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['atr_ratio'] = (atr / df['close']).shift(1)

    # --- 2. THE REALISTIC TARGET ---
    # Goal: Is price 1.0% higher in 8 hours?
    df['future_8h_return'] = df['close'].shift(-8) / df['close'] - 1
    df['label'] = np.where(df['future_8h_return'] > 0.01, 1, 0)

    df.dropna(inplace=True)
    return df

def run_leak_proof_backtest(pair, csv_file):
    df = prepare_data(csv_file)
    if df is None: return

    features = ['rsi', 'macd_diff', 'atr_ratio']
    X = df[features]
    y = df['label']

    # Chronological Split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train Model (Hardware Accelerated)
    model = XGBClassifier(tree_method='hist', device='cpu', n_estimators=500, learning_rate=0.02)
    model.fit(X_train, y_train)

    # --- 3. BACKTEST WITH STOP LOSS ---
    test_df = df.iloc[split:].copy()
    test_df['prob'] = model.predict_proba(X_test)[:, 1]
    
    # Adjusted Confidence Threshold (60%)
    test_df['signal'] = (test_df['prob'] > 0.60).astype(int)

    # Trading Rules
    fee = 0.002 # 0.2% total round-trip
    stop_loss = -0.02 # 2% Stop Loss

    # If signal is 1, return is either the 8h result OR the stop loss
    test_df['trade_return'] = np.where(
        test_df['signal'] == 1,
        np.maximum(test_df['future_8h_return'], stop_loss) - fee,
        0
    )

    # --- 4. PERFORMANCE METRICS ---
    # Cumulative Returns
    test_df['equity_curve'] = (1 + test_df['trade_return']).cumprod()
    
    # Max Drawdown Calculation
    rolling_max = test_df['equity_curve'].cummax()
    drawdown = (test_df['equity_curve'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    total_return = test_df['equity_curve'].iloc[-1] - 1
    trades = test_df['signal'].sum()
    win_rate = (test_df[test_df['signal'] == 1]['trade_return'] > 0).mean()

    print(f"\n{'='*30}")
    print(f"REPORT FOR {pair}")
    print(f"{'='*30}")
    print(f"Total Trades:    {int(trades)}")
    print(f"Win Rate:        {win_rate:.2%}")
    print(f"Net Profit:      {total_return:.2%}")
    print(f"Max Drawdown:    {max_drawdown:.2%}")
    print(f"{'='*30}\n")

if __name__ == "__main__":
    run_leak_proof_backtest('BTC-USD', 'btc_usd_hourly.csv')