from flask import Flask, request, jsonify
import os
import time
import pandas as pd
import numpy as np
import joblib
from coinbase.rest import RESTClient
from dotenv import load_dotenv
from ta import momentum, trend, volatility  # FIX 1: added volatility import

load_dotenv()

app = Flask(__name__)

PASSPHRASE = os.getenv("TV_SECRET")

client = RESTClient(
    api_key=os.getenv("COINBASE_API_KEY"),
    api_secret=os.getenv("COINBASE_API_SECRET")
)

# Risk management — edit these to match your strategy
RISK_PER_TRADE_PCT = 0.01   # 1% of available USD balance per trade
BUY_CONFIDENCE_THRESHOLD = 0.60
SELL_CONFIDENCE_THRESHOLD = 0.55  # Lower bar to exit than to enter
STOP_LOSS_PCT = 0.02         # 2% stop loss
TAKE_PROFIT_PCT = 0.04       # 4% take profit (2:1 R/R)

SUPPORTED_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "LINK-USD"]

# Load models — one per pair
models = {pair: joblib.load(f"model_{pair.replace('-', '_')}.pkl") for pair in SUPPORTED_PAIRS}

FEATURES = ['log_return', 'atr_ratio', 'volume_ratio', 'rsi', 'rsi_slope', 'macd_diff']


def fetch_candles(product_id: str, lookback_hours: int = 50) -> pd.DataFrame:
    """Fetch recent hourly candles. Needs >14 bars for indicators to stabilise."""
    response = client.get_public_candles(
        product_id=product_id,
        start=str(int(time.time() - lookback_hours * 3600)),
        end=str(int(time.time())),
        granularity='ONE_HOUR'
    )
    df = pd.DataFrame(response['candles'])
    df.columns = ['start', 'low', 'high', 'open', 'close', 'volume']
    for col in ['start', 'low', 'high', 'open', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df['timestamp'] = pd.to_datetime(df['start'], unit='s')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features. Matches the feature set the models were trained on."""
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['atr_ratio'] = (
        volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        .average_true_range() / df['close']
    )
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(14).mean()
    df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_slope'] = df['rsi'] - df['rsi'].shift(1)
    macd = trend.MACD(df['close'])
    df['macd_diff'] = macd.macd() - macd.macd_signal()
    df.dropna(inplace=True)
    return df


def get_available_usd_balance() -> float:
    """Return free USD balance from the Coinbase account."""
    accounts = client.get_accounts()
    for account in accounts['accounts']:
        if account['currency'] == 'USD':
            return float(account['available_balance']['value'])
    return 0.0


def get_crypto_balance(base_currency: str) -> float:
    """Return free balance of a given crypto (e.g. 'BTC')."""
    accounts = client.get_accounts()
    for account in accounts['accounts']:
        if account['currency'] == base_currency:
            return float(account['available_balance']['value'])
    return 0.0


# Home route
@app.route('/', methods=['GET'])
def home():
    return "✅ Coinbase Trading Bot is LIVE!<br>Use /webhook for TradingView signals.", 200


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json

    # Auth
    if data.get('passphrase') != PASSPHRASE:
        return jsonify({"status": "unauthorized"}), 401

    product_id = data.get('ticker')
    action = data.get('action')  # 'buy' or 'sell'

    if product_id not in SUPPORTED_PAIRS:
        return jsonify({"status": "error", "message": f"Unsupported pair: {product_id}"}), 400

    if action not in ('buy', 'sell'):
        return jsonify({"status": "error", "message": "action must be 'buy' or 'sell'"}), 400

    model = models[product_id]

    try:
        df = fetch_candles(product_id, lookback_hours=50)
        df = engineer_features(df)

        if df.empty or len(df) < 2:
            return jsonify({"status": "insufficient_data"}), 400

        latest = df[FEATURES].iloc[-1:].copy()
        prob = model.predict_proba(latest)[0][1]
        current_price = float(df['close'].iloc[-1])

        print(f"[{product_id}] action={action} confidence={prob:.2%} price=${current_price:,.2f}")

        # --- BUY logic ---
        if action == 'buy':
            if prob < BUY_CONFIDENCE_THRESHOLD:
                return jsonify({
                    "status": "trade_filtered_out",
                    "reason": f"confidence {prob:.2%} below {BUY_CONFIDENCE_THRESHOLD:.0%} threshold",
                    "prob": round(prob, 4)
                }), 200

            usd_balance = get_available_usd_balance()
            quote_size = round(usd_balance * RISK_PER_TRADE_PCT, 2)

            if quote_size < 1.0:
                return jsonify({"status": "insufficient_balance", "usd_available": usd_balance}), 200

            order = client.market_order_buy(
                product_id=product_id,
                quote_size=str(quote_size)
            )

            stop_price  = round(current_price * (1 - STOP_LOSS_PCT), 4)
            target_price = round(current_price * (1 + TAKE_PROFIT_PCT), 4)
            crypto_qty = round(quote_size / current_price, 6)

            try:
                client.stop_limit_order_gtc_sell(
                    product_id=product_id,
                    base_size=str(crypto_qty),
                    limit_price=str(round(stop_price * 0.995, 4)),
                    stop_price=str(stop_price)
                )
            except Exception as sl_err:
                print(f"[{product_id}] Stop-loss order failed: {sl_err}")

            try:
                client.limit_order_gtc_sell(
                    product_id=product_id,
                    base_size=str(crypto_qty),
                    limit_price=str(target_price)
                )
            except Exception as tp_err:
                print(f"[{product_id}] Take-profit order failed: {tp_err}")

            return jsonify({
                "status": "buy_executed",
                "product_id": product_id,
                "quote_size_usd": quote_size,
                "prob": round(prob, 4),
                "entry_price": current_price,
                "stop_loss": stop_price,
                "take_profit": target_price,
                "order_id": order.get('order_id', 'unknown')
            }), 200

        # --- SELL logic ---
        elif action == 'sell':
            if prob > SELL_CONFIDENCE_THRESHOLD:
                return jsonify({
                    "status": "trade_filtered_out",
                    "reason": f"model still bullish ({prob:.2%}), not selling",
                    "prob": round(prob, 4)
                }), 200

            base_currency = product_id.split('-')[0]
            crypto_balance = get_crypto_balance(base_currency)

            if crypto_balance <= 0:
                return jsonify({"status": "no_position", "message": f"No {base_currency} to sell"}), 200

            order = client.market_order_sell(
                product_id=product_id,
                base_size=str(round(crypto_balance, 6))
            )

            return jsonify({
                "status": "sell_executed",
                "product_id": product_id,
                "base_size": crypto_balance,
                "prob": round(prob, 4),
                "exit_price": current_price,
                "order_id": order.get('order_id', 'unknown')
            }), 200

    except Exception as e:
        print(f"[{product_id}] ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)