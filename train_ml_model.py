import pandas as pd
import numpy as np
import joblib
import os
from ta import momentum, trend, volatility

def run_backtest(csv_file, pair, model_file, threshold=0.85):
    print(f"\n--- Reality-Check Backtest: {pair} ---")
    
    if not os.path.exists(csv_file) or not os.path.exists(model_file):
        print(f"Skipping: Missing files.")
        return

    df = pd.read_csv(csv_file)
    model = joblib.load(model_file)
    
    # 1. Feature Engineering (Must match train_ml_model.py exactly)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['atr'] = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['atr_ratio'] = df['atr'] / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=14).mean()
    df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_slope'] = df['rsi'] - df['rsi'].shift(1)
    df['macd_diff'] = trend.MACD(df['close']).macd_diff()

    # Define exact feature list
    features = ['log_return', 'atr_ratio', 'volume_ratio', 'rsi', 'rsi_slope', 'macd_diff']
    
    # Handle On-Chain features if present in the CSV
    for col in ['addr_growth', 'addr_momentum', 'tx_growth']:
        if col in df.columns:
            features.append(col)

    df.dropna(inplace=True)
    
    # 2. Probability Filtering
    # Instead of model.predict(), we get the percentage chance of success
    probs = model.predict_proba(df[features])[:, 1]
    df['raw_signal'] = (probs > threshold).astype(int)

    # 3. THE SHIFT: Execute at the START of the next hour
    # Signal at 1:00 PM -> Entry at 2:00 PM
    df['entry_signal'] = df['raw_signal'].shift(1)

    # 4. Trading Simulation
    fee = 0.001 
    # Return from current 'close' to 'close' 8 hours later
    df['next_8h_return'] = df['close'].shift(-8) / df['close'] - 1
    
    # Net return accounting for fees
    df['strategy_return'] = np.where(
        df['entry_signal'] == 1, 
        df['next_8h_return'] - (fee * 2), 
        0
    )

    # 5. Metrics
    total_trades = df['entry_signal'].sum()
    if total_trades > 0:
        win_rate = len(df[(df['entry_signal'] == 1) & (df['next_8h_return'] > 0)]) / total_trades
        cumulative_return = (1 + df['strategy_return']).prod() - 1
        
        print(f"  > Threshold: {threshold*100}%")
        print(f"  > Total Trades: {int(total_trades)}")
        print(f"  > Win Rate: {win_rate:.2%}")
        print(f"  > Net Profit: {cumulative_return:.2%}")
    else:
        print(f"  > No trades found at {threshold*100}% confidence.")

if __name__ == "__main__":
    run_backtest('btc_usd_hourly.csv', 'BTC-USD', 'model_BTC_USD.pkl', threshold=0.80)