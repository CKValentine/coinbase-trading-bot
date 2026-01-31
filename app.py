from flask import Flask, request, jsonify
import os
import time
import pandas as pd
import numpy as np
import joblib
from coinbase.rest import RESTClient
from dotenv import load_dotenv
from ta import momentum, trend  # For RSI and MACD
import uuid  # Add this at the top for client_order_id

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

    product_id = data['ticker']
    model = models.get(product_id)
    
    # 1. Fetch Data (SDK Fix)
    try:
        response = client.get_public_candles(
            product_id=product_id,
            start=str(int(time.time() - 36000)), # Fetch 10 hours
            end=str(int(time.time())),
            granularity='ONE_HOUR'
        )
        
        # FIX: Access as object attribute
        candles = [vars(c) for c in response.candles] 
        df = pd.DataFrame(candles)
        
        # 2. Indicator logic (Ensure 'df' naming is consistent)
        df['close'] = df['close'].astype(float)
        df['rsi'] = momentum.RSIIndicator(df['close']).rsi()
        # ... (rest of your indicators) ...

        # 3. Model Prediction (Feature Name Fix)
        features = ['log_return', 'atr_ratio', 'volume_ratio', 'rsi', 'rsi_slope', 'macd', 'macd_signal']
        latest_row = df[features].iloc[-1:] # Keep as DataFrame to preserve column names

        prob = model.predict_proba(latest_row)[0][1]
        
        # 4. Execution (UUID & Quote Size Fix)
        if prob > 0.60 and data['action'] == 'buy':
            # Coinbase requires a unique ID for every order attempt
            order_id = str(uuid.uuid4()) 
            
            order_response = client.market_order_buy(
                client_order_id=order_id,
                product_id=product_id,
                quote_size=str(data['size']) # Must be a string
            )
            
            return jsonify({"status": "success", "order": str(order_response)}), 200
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)