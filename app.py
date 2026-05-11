from flask import Flask, request, jsonify
import os
import json
import time
import threading
import requests as http_requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from coinbase.rest import RESTClient
from dotenv import load_dotenv
from ta import momentum, trend, volatility

load_dotenv()

app = Flask(__name__)

PASSPHRASE = os.getenv("TV_SECRET")

client = RESTClient(
    api_key=os.getenv("COINBASE_API_KEY"),
    api_secret=os.getenv("COINBASE_API_SECRET")
)

# ---------------------------------------------------------------------------
# Per-pair configuration — thresholds match train_and_backtest.py
# ---------------------------------------------------------------------------
PAIR_CONFIG = {
    "BTC-USD":  dict(buy_threshold=0.53, sell_threshold=0.48),
    "ETH-USD":  dict(buy_threshold=0.58, sell_threshold=0.52),
    "XRP-USD":  dict(buy_threshold=0.58, sell_threshold=0.52),
    "ADA-USD":  dict(buy_threshold=0.58, sell_threshold=0.52),
    "LINK-USD": dict(buy_threshold=0.58, sell_threshold=0.52),
}

SUPPORTED_PAIRS    = list(PAIR_CONFIG.keys())
RISK_PER_TRADE_PCT = 0.20
STOP_LOSS_PCT      = 0.02
TAKE_PROFIT_PCT    = 0.04

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
_raw     = {pair: joblib.load(f"model_{pair.replace('-', '_')}.pkl") for pair in SUPPORTED_PAIRS}
models   = {pair: _raw[pair]['model']    for pair in SUPPORTED_PAIRS}
features = {pair: _raw[pair]['features'] for pair in SUPPORTED_PAIRS}

# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------
POSITIONS_FILE  = "positions.json"
_positions_lock = threading.Lock()


def load_positions() -> dict:
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_positions(positions: dict):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2)


def has_open_position(product_id: str) -> bool:
    """Check local state, auto-clearing if SL/TP has already filled on Coinbase."""
    with _positions_lock:
        positions = load_positions()

    if product_id not in positions:
        return False

    pos = positions[product_id]
    for order_id in [pos.get('sl_order_id'), pos.get('tp_order_id')]:
        if not order_id:
            continue
        try:
            order  = client.get_order(order_id)
            status = order['order']['status']
            if status in ('FILLED', 'CANCELLED'):
                with _positions_lock:
                    positions = load_positions()
                    positions.pop(product_id, None)
                    save_positions(positions)
                print(f"[{product_id}] Position auto-cleared — order {order_id} is {status}")
                return False
        except Exception as e:
            print(f"[{product_id}] Could not check order {order_id}: {e}")

    return True


def open_position(product_id, entry_price, qty, stop_price,
                  target_price, sl_order_id, tp_order_id):
    with _positions_lock:
        positions = load_positions()
        positions[product_id] = {
            "entry_price":  entry_price,
            "qty":          qty,
            "stop_price":   stop_price,
            "target_price": target_price,
            "sl_order_id":  sl_order_id,
            "tp_order_id":  tp_order_id,
            "opened_at":    datetime.utcnow().isoformat(),
        }
        save_positions(positions)


def close_position(product_id: str):
    with _positions_lock:
        positions = load_positions()
        positions.pop(product_id, None)
        save_positions(positions)


# ---------------------------------------------------------------------------
# Market data & feature engineering
# ---------------------------------------------------------------------------

def fetch_candles(product_id: str, lookback_hours: int = 250) -> pd.DataFrame:
    response = client.get_public_candles(
        product_id=product_id,
        start=str(int(time.time() - lookback_hours * 3600)),
        end=str(int(time.time())),
        granularity='ONE_HOUR'
    )
    candles = response['candles']
    df = pd.DataFrame([{
        'start':  c['start'],
        'low':    c['low'],
        'high':   c['high'],
        'open':   c['open'],
        'close':  c['close'],
        'volume': c['volume']
    } for c in candles])
    for col in ['start', 'low', 'high', 'open', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df['timestamp'] = pd.to_datetime(df['start'], unit='s')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_fear_greed() -> float:
    try:
        r = http_requests.get(
            'https://api.alternative.me/fng/?limit=1&format=json', timeout=5
        )
        return float(r.json()['data'][0]['value']) / 100.0
    except Exception:
        return 0.5


def engineer_features(df: pd.DataFrame, pair_features: list) -> pd.DataFrame:
    df = df.copy()
    df.set_index('timestamp', inplace=True)

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

    hour = df.index.hour
    dow  = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin']  = np.sin(2 * np.pi * dow  / 7)
    df['dow_cos']  = np.cos(2 * np.pi * dow  / 7)

    if 'fear_greed' in pair_features:
        df['fear_greed'] = get_fear_greed()
    if 'dxy_return' in pair_features:
        df['dxy_return'] = 0.0
    if 'gold_return' in pair_features:
        df['gold_return'] = 0.0

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

def get_available_usd_balance() -> float:
    cursor = None
    while True:
        kwargs = {'limit': 250}
        if cursor:
            kwargs['cursor'] = cursor
        response = client.get_accounts(**kwargs)
        for account in response['accounts']:
            if account['currency'] == 'USD':
                return float(account['available_balance']['value'])
        if not response.get('has_next') or not response.get('cursor'):
            break
        cursor = response['cursor']
    return 0.0


def get_crypto_balance(base_currency: str) -> float:
    cursor = None
    while True:
        kwargs = {'limit': 250}
        if cursor:
            kwargs['cursor'] = cursor
        response = client.get_accounts(**kwargs)
        for account in response['accounts']:
            if account['currency'] == base_currency:
                return float(account['available_balance']['value'])
        if not response.get('has_next') or not response.get('cursor'):
            break
        cursor = response['cursor']
    return 0.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return (
        "✅ Coinbase Trading Bot is LIVE!<br>"
        "POST /webhook — TradingView signals<br>"
        "GET  /status  — open positions & account info"
    ), 200


@app.route('/status', methods=['GET'])
def status():
    positions = load_positions()
    try:
        usd = get_available_usd_balance()
    except Exception:
        usd = None
    return jsonify({
        "open_positions":  positions,
        "usd_available":   usd,
        "supported_pairs": SUPPORTED_PAIRS,
        "timestamp":       datetime.utcnow().isoformat()
    }), 200


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json

    if data.get('passphrase') != PASSPHRASE:
        return jsonify({"status": "unauthorized"}), 401

    product_id = data.get('ticker')
    action     = data.get('action')

    if product_id not in SUPPORTED_PAIRS:
        return jsonify({"status": "error", "message": f"Unsupported pair: {product_id}"}), 400

    if action not in ('buy', 'sell'):
        return jsonify({"status": "error", "message": "action must be 'buy' or 'sell'"}), 400

    cfg           = PAIR_CONFIG[product_id]
    model         = models[product_id]
    pair_features = features[product_id]

    try:
        df = fetch_candles(product_id, lookback_hours=250)
        df = engineer_features(df, pair_features)

        if df.empty or len(df) < 2:
            return jsonify({"status": "insufficient_data"}), 400

        latest        = df[pair_features].iloc[-1:].copy()
        prob          = float(model.predict_proba(latest)[0][1])
        current_price = float(df['close'].iloc[-1])

        print(f"[{product_id}] action={action} prob={prob:.2%} price=${current_price:,.4f}")

        # --- BUY ---
        if action == 'buy':
            if prob < cfg['buy_threshold']:
                return jsonify({
                    "status": "filtered",
                    "reason": f"prob {prob:.2%} < threshold {cfg['buy_threshold']:.0%}",
                    "prob":   round(prob, 4)
                }), 200

            if has_open_position(product_id):
                return jsonify({
                    "status": "skipped",
                    "reason": f"position already open for {product_id}",
                    "existing": load_positions().get(product_id, {})
                }), 200

            usd_balance = get_available_usd_balance()
            quote_size  = round(usd_balance * RISK_PER_TRADE_PCT, 2)

            if quote_size < 1.0:
                return jsonify({"status": "insufficient_balance", "usd_available": usd_balance}), 200

            order = client.market_order_buy(
                product_id=product_id,
                quote_size=str(quote_size)
            )

            stop_price   = round(current_price * (1 - STOP_LOSS_PCT), 4)
            target_price = round(current_price * (1 + TAKE_PROFIT_PCT), 4)
            crypto_qty   = round(quote_size / current_price, 6)

            sl_order_id = tp_order_id = None

            try:
                sl = client.stop_limit_order_gtc_sell(
                    product_id=product_id,
                    base_size=str(crypto_qty),
                    limit_price=str(round(stop_price * 0.995, 4)),
                    stop_price=str(stop_price)
                )
                sl_order_id = sl.get('order_id')
            except Exception as e:
                print(f"[{product_id}] SL order failed: {e}")

            try:
                tp = client.limit_order_gtc_sell(
                    product_id=product_id,
                    base_size=str(crypto_qty),
                    limit_price=str(target_price)
                )
                tp_order_id = tp.get('order_id')
            except Exception as e:
                print(f"[{product_id}] TP order failed: {e}")

            open_position(
                product_id, current_price, crypto_qty,
                stop_price, target_price, sl_order_id, tp_order_id
            )

            return jsonify({
                "status":         "buy_executed",
                "product_id":     product_id,
                "quote_size_usd": quote_size,
                "prob":           round(prob, 4),
                "entry_price":    current_price,
                "stop_loss":      stop_price,
                "take_profit":    target_price,
                "order_id":       order.get('order_id', 'unknown'),
                "sl_order_id":    sl_order_id,
                "tp_order_id":    tp_order_id,
            }), 200

        # --- SELL ---
        elif action == 'sell':
            if prob > cfg['sell_threshold']:
                return jsonify({
                    "status": "filtered",
                    "reason": f"model still bullish ({prob:.2%}), not selling",
                    "prob":   round(prob, 4)
                }), 200

            base_currency  = product_id.split('-')[0]
            crypto_balance = get_crypto_balance(base_currency)

            if crypto_balance <= 0:
                close_position(product_id)
                return jsonify({"status": "no_position", "message": f"No {base_currency} to sell"}), 200

            order = client.market_order_sell(
                product_id=product_id,
                base_size=str(round(crypto_balance, 6))
            )

            close_position(product_id)

            return jsonify({
                "status":     "sell_executed",
                "product_id": product_id,
                "base_size":  crypto_balance,
                "prob":       round(prob, 4),
                "exit_price": current_price,
                "order_id":   order.get('order_id', 'unknown'),
            }), 200

    except Exception as e:
        print(f"[{product_id}] ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
