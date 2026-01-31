from flask import Flask, request, jsonify
import os
import time
import pandas as pd
import numpy as np
import joblib
from coinbase.rest import RESTClient
from dotenv import load_dotenv
from ta import momentum, trend

load_dotenv()

app = Flask(__name__)

PASSPHRASE = os.getenv("TV_SECRET")

client = RESTClient(
    api_key=os.getenv("COINBASE_API_KEY"),
    api_secret=os.getenv("COINBASE_API_SECRET")
)

# Load models
models = {
    "BTC-USD": joblib.load('model_BTC_USD.pkl'),
    "ETH-USD": joblib.load('model_ETH_USD.pkl'),
    "SOL-USD": joblib.load('model_SOL_USD.pkl'),
    "XRP-USD": joblib.load('model_XRP_USD.pkl'),
    "ADA-USD": joblib.load('model_ADA_USD.pkl'),
    "LINK-USD": joblib.load('model_LINK_USD.pkl')
}

# Home route - shows when you visit the URL in browser
@app.route('/', methods=['GET'])
def home():
    return "✅ Coinbase Trading Bot is LIVE!<br>Use /webhook for TradingView signals.", 200

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if data.get('passphrase') != PASSPHRASE:
        return jsonify({"status": "unauthorized"}), 401

    product_id = data['ticker']
    model = models.get(product_id)

    if not model:
        return jsonify({"status": "error", "message": "No model found"}), 400

    # Fetch recent data
    try:
        response = client.get_public_candles(
            product_id=product_id,
            start=str(int(time.time() - 36000)),  # 10 hours
            end=str(int(time.time())),
            granularity='ONE_HOUR'
        )

        candles = response['candles']
        df = pd.DataFrame(candles)
        df.columns = ['start', 'low', 'high', 'open', 'close', 'volume']
        df['start'] = df['start'].astype(int)
        df['timestamp'] = pd.to_datetime(df['start'], unit='s')

        # Feature Engineering
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['atr_ratio'] = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range() / df['close']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(14).mean()
        df['rsi'] = momentum.RSIIndicator(df['close']).rsi()
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(1)
        macd = trend.MACD(df['close'])
        df['macd_diff'] = macd.macd() - macd.macd_signal()

        df.dropna(inplace=True)
        if df.empty:
            return jsonify({"status": "insufficient_data"}), 400

        features = ['log_return', 'atr_ratio', 'volume_ratio', 'rsi', 'rsi_slope', 'macd_diff']
        latest = df[features].iloc[-1:].copy()

        prob = model.predict_proba(latest)[0][1]

        print(f"[{product_id}] Confidence: {prob:.2%}")

        if prob > 0.60 and data['action'] == 'buy':
            order_response = client.market_order_buy(
                product_id=product_id,
                quote_size=str(data['size'])
            )
            return jsonify({"status": "trade_executed", "prob": prob}), 200
        else:
            return jsonify({"status": "trade_filtered_out", "prob": prob}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)