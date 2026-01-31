from flask import Flask, request, jsonify
import os
import time
import pandas as pd
import numpy as np
import joblib
from coinbase.rest import RESTClient
from dotenv import load_dotenv
from ta import momentum, trend  # For RSI and MACD

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Security: Set a secret passphrase from .env
PASSPHRASE = os.getenv("TV_SECRET")

# Initialize Coinbase client with keys from .env
client = RESTClient(
    api_key=os.getenv("COINBASE_API_KEY"),
    api_secret=os.getenv("COINBASE_API_SECRET")
)

# Load models into a dictionary for easy access
models = {
    "BTC-USD": joblib.load('model_BTC_USD.pkl'),
    "ETH-USD": joblib.load('model_ETH_USD.pkl'),
    "SOL-USD": joblib.load('model_SOL_USD.pkl'),
    "XRP-USD": joblib.load('model_XRP_USD.pkl'),
    "ADA-USD": joblib.load('model_ADA_USD.pkl'),
    "LINK-USD": joblib.load('model_LINK_USD.pkl')
}

@app.route('/', methods=['GET'])
def home():
    return "Coinbase Trading Bot is running. Use /webhook for signals.", 200

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if data.get('passphrase') != PASSPHRASE:
        return jsonify({"status": "unauthorized"}), 401

    product_id = data['ticker']  # e.g., 'BTC-USD'
    
    # Load the specific model from dict
    model = models.get(product_id)
    if not model:
        return jsonify({"status": "error", "message": f"No model for {product_id}"}), 400

    # Fetch last 24 hours of hourly data for features
    granularity = 'ONE_HOUR'
    end = int(time.time())  # Current Unix timestamp
    start = end - (24 * 3600)  # 24 hours ago

    try:
        response = client.get_public_candles(
            product_id=product_id,
            start=str(start),
            end=str(end),
            granularity=granularity
        )
        candles = response['candles']
        df_recent = pd.DataFrame(candles)
        df_recent.columns = ['start', 'low', 'high', 'open', 'close', 'volume']  # Force columns if needed
        df_recent['start'] = df_recent['start'].astype(int)  # Ensure int for timestamp
        df_recent['timestamp'] = pd.to_datetime(df_recent['start'], unit='s')
        df_recent[['low', 'high', 'open', 'close', 'volume']] = df_recent[['low', 'high', 'open', 'close', 'volume']].astype(float)

        # Engineer features (matching training)
        df_recent['log_return'] = np.log(df_recent['close'] / df_recent['close'].shift(1))
        high_low = df_recent['high'] - df_recent['low']
        high_close = np.abs(df_recent['high'] - df_recent['close'].shift())
        low_close = np.abs(df_recent['low'] - df_recent['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_recent['atr'] = true_range.rolling(window=14).mean()
        df_recent['volume_ma'] = df_recent['volume'].rolling(window=14).mean()
        df_recent['volume_ratio'] = df_recent['volume'] / df_recent['volume_ma']
        df_recent['rsi'] = momentum.RSIIndicator(df_recent['close']).rsi()
        df_recent['rsi_slope'] = df_recent['rsi'] - df_recent['rsi'].shift(1)
        df_recent['atr_ratio'] = df_recent['atr'] / df_recent['close']
        macd = trend.MACD(df_recent['close'])
        df_recent['macd'] = macd.macd()
        df_recent['macd_signal'] = macd.macd_signal()

        # Drop NaNs and get latest features
        df_recent.dropna(inplace=True)
        if df_recent.empty:
            return jsonify({"status": "insufficient_data"}), 400
        
        features = ['log_return', 'atr_ratio', 'volume_ratio', 'rsi', 'rsi_slope', 'macd', 'macd_signal']
        latest_features = df_recent.iloc[-1][features].values.reshape(1, -1)

        # Predict probability of profitable trade
        prob = model.predict_proba(latest_features)[0][1]  # Prob of class 1
        print(f"Model Confidence for {product_id}: {prob:.2%}")

        # Confidence Filter: Trade only if >60% sure (per Gemini; adjust to 0.75 if too many trades)
        if prob > 0.60 and data['action'] == 'buy':
            # Execute market buy order
            order_response = client.market_order_buy(
                product_id=product_id,
                quote_size=data['size']  # USD amount
            )
            return jsonify({"status": "trade_executed", "order_id": order_response.get('order_id'), "prob": prob}), 200
        else:
            return jsonify({"status": "trade_filtered_out", "prob": prob}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)