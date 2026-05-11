from flask import Flask, request, jsonify
import os
import json
import time
import threading
import requests as http_requests
import yfinance as yf
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
# Per-pair configuration
# ---------------------------------------------------------------------------
PAIR_CONFIG = {
    "BTC-USD":  dict(buy_threshold=0.53, sell_threshold=0.48),
    "ETH-USD":  dict(buy_threshold=0.58, sell_threshold=0.52),
    "SOL-USD":  dict(buy_threshold=0.58, sell_threshold=0.52),  # re-enabled
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
# Position & trade history tracking
# ---------------------------------------------------------------------------
POSITIONS_FILE    = "positions.json"
TRADE_HISTORY_FILE = "trade_history.json"
_positions_lock   = threading.Lock()
_history_lock     = threading.Lock()


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


def load_trade_history() -> list:
    if not os.path.exists(TRADE_HISTORY_FILE):
        return []
    try:
        with open(TRADE_HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def log_trade(entry: dict):
    with _history_lock:
        history = load_trade_history()
        history.append(entry)
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)


def has_open_position(product_id: str) -> bool:
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
                    pos = positions.pop(product_id, None)
                    save_positions(positions)
                print(f"[{product_id}] Position auto-cleared — order {order_id} is {status}")
                return False
        except Exception as e:
            print(f"[{product_id}] Could not check order {order_id}: {e}")

    return True


def open_position(product_id, entry_price, qty, stop_price,
                  target_price, sl_order_id, tp_order_id, quote_size, prob):
    with _positions_lock:
        positions = load_positions()
        positions[product_id] = {
            "entry_price":  entry_price,
            "qty":          qty,
            "quote_size":   quote_size,
            "stop_price":   stop_price,
            "target_price": target_price,
            "sl_order_id":  sl_order_id,
            "tp_order_id":  tp_order_id,
            "prob":         prob,
            "opened_at":    datetime.utcnow().isoformat(),
        }
        save_positions(positions)

    log_trade({
        "type":         "buy",
        "product_id":   product_id,
        "entry_price":  entry_price,
        "qty":          qty,
        "quote_size":   quote_size,
        "stop_price":   stop_price,
        "target_price": target_price,
        "prob":         prob,
        "timestamp":    datetime.utcnow().isoformat(),
        "status":       "open",
    })


def close_position(product_id: str, exit_price: float = None, reason: str = "signal"):
    with _positions_lock:
        positions = load_positions()
        pos = positions.pop(product_id, None)
        save_positions(positions)

    if pos and exit_price:
        entry  = pos.get('entry_price', 0)
        qty    = pos.get('qty', 0)
        pnl    = round((exit_price - entry) * qty, 4)
        pnl_pct = round((exit_price / entry - 1) * 100, 2) if entry else 0
        log_trade({
            "type":       "sell",
            "product_id": product_id,
            "entry_price": entry,
            "exit_price": exit_price,
            "qty":        qty,
            "pnl":        pnl,
            "pnl_pct":   pnl_pct,
            "reason":     reason,
            "timestamp":  datetime.utcnow().isoformat(),
            "status":     "closed",
        })


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


def get_live_macro() -> tuple:
    try:
        dxy  = yf.Ticker('DX-Y.NYB').history(period='2d', interval='1h')['Close']
        gold = yf.Ticker('GC=F').history(period='2d', interval='1h')['Close']
        return float(dxy.iloc[-1] / dxy.iloc[-2] - 1), float(gold.iloc[-1] / gold.iloc[-2] - 1)
    except Exception as e:
        print(f"Macro fetch failed: {e}")
        return 0.0, 0.0


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
    if 'dxy_return' in pair_features or 'gold_return' in pair_features:
        dxy_ret, gold_ret = get_live_macro()
        if 'dxy_return' in pair_features:
            df['dxy_return'] = dxy_ret
        if 'gold_return' in pair_features:
            df['gold_return'] = gold_ret

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


def get_current_price(product_id: str) -> float:
    try:
        response = client.get_public_candles(
            product_id=product_id,
            start=str(int(time.time() - 3600)),
            end=str(int(time.time())),
            granularity='ONE_HOUR'
        )
        candles = response['candles']
        return float(candles[0]['close']) if candles else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return (
        "✅ Coinbase Trading Bot is LIVE!<br>"
        "POST /webhook   — TradingView signals<br>"
        "GET  /status    — JSON status<br>"
        "GET  /dashboard — Trade dashboard"
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


@app.route('/dashboard', methods=['GET'])
def dashboard():
    positions = load_positions()
    history   = load_trade_history()
    try:
        usd = round(get_available_usd_balance(), 2)
    except Exception:
        usd = 0.0

    # Enrich open positions with current price and unrealised P&L
    enriched = {}
    for pair, pos in positions.items():
        current = get_current_price(pair)
        entry   = pos.get('entry_price', 0)
        qty     = pos.get('qty', 0)
        unreal  = round((current - entry) * qty, 4) if entry and qty else 0
        unreal_pct = round((current / entry - 1) * 100, 2) if entry else 0
        enriched[pair] = {**pos, "current_price": current,
                          "unrealised_pnl": unreal, "unrealised_pct": unreal_pct}

    # Trade stats
    closed = [t for t in history if t.get('type') == 'sell']
    total_trades = len(closed)
    wins         = len([t for t in closed if t.get('pnl', 0) > 0])
    win_rate     = round(wins / total_trades * 100, 1) if total_trades else 0
    total_pnl    = round(sum(t.get('pnl', 0) for t in closed), 4)
    win_color    = "green" if win_rate >= 50 else "red"
    pnl_color    = "green" if total_pnl >= 0 else "red"
    pnl_sign     = "+" if total_pnl >= 0 else ""

    # Build open positions rows
    pos_rows = ""
    if not enriched:
        pos_rows = '<p class="empty">No open positions</p>'
    else:
        rows = ""
        for pair, p in enriched.items():
            pct      = p["unrealised_pct"]
            pc       = "green" if p["unrealised_pnl"] >= 0 else "red"
            opened   = p["opened_at"][:16].replace("T", " ")
            conf     = round(p.get("prob", 0) * 100, 1)
            rows += (
                "<tr>"
                "<td><strong>" + pair + "</strong></td>"
                "<td>$" + "{:,.4f}".format(p["entry_price"]) + "</td>"
                "<td>$" + "{:,.4f}".format(p["current_price"]) + "</td>"
                '<td><span class="badge ' + pc + '">' + "{:+.2f}".format(pct) + "%</span></td>"
                "<td>$" + "{:,.4f}".format(p["stop_price"]) + "</td>"
                "<td>$" + "{:,.4f}".format(p["target_price"]) + "</td>"
                "<td>" + str(conf) + "%</td>"
                "<td>" + opened + "</td>"
                "</tr>"
            )
        pos_rows = (
            "<table><tr><th>Pair</th><th>Entry</th><th>Current</th>"
            "<th>P&amp;L</th><th>Stop</th><th>Target</th>"
            "<th>Confidence</th><th>Opened</th></tr>"
            + rows + "</table>"
        )

    # Build trade history rows
    hist_rows = ""
    if not closed:
        hist_rows = '<p class="empty">No closed trades yet</p>'
    else:
        rows = ""
        for t in reversed(closed[-20:]):
            pc      = "green" if t.get("pnl", 0) >= 0 else "red"
            pct_val = t.get("pnl_pct", 0)
            ts      = t["timestamp"][:16].replace("T", " ")
            rows += (
                "<tr>"
                "<td><strong>" + t["product_id"] + "</strong></td>"
                "<td>$" + "{:,.4f}".format(t.get("entry_price", 0)) + "</td>"
                "<td>$" + "{:,.4f}".format(t.get("exit_price", 0)) + "</td>"
                '<td><span class="badge ' + pc + '">' + "{:+.2f}".format(pct_val) + "%</span></td>"
                '<td><span class="badge blue">' + t.get("reason", "signal") + "</span></td>"
                "<td>" + ts + "</td>"
                "</tr>"
            )
        hist_rows = (
            "<table><tr><th>Pair</th><th>Entry</th><th>Exit</th>"
            "<th>P&amp;L</th><th>Reason</th><th>Time</th></tr>"
            + rows + "</table>"
        )

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Trading Bot Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 20px; }
  h1 { font-size: 20px; font-weight: 500; margin-bottom: 20px; color: #fff; }
  h2 { font-size: 13px; font-weight: 500; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: #1a1a1a; border: 0.5px solid #333; border-radius: 10px; padding: 16px; }
  .card .label { font-size: 12px; color: #888; margin-bottom: 6px; }
  .card .value { font-size: 24px; font-weight: 500; color: #fff; }
  .card .value.green { color: #4caf50; }
  .card .value.red { color: #f44336; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; color: #888; font-weight: 400; border-bottom: 0.5px solid #333; }
  td { padding: 10px 12px; border-bottom: 0.5px solid #222; }
  tr:last-child td { border-bottom: none; }
  .section { background: #1a1a1a; border: 0.5px solid #333; border-radius: 10px; padding: 16px; margin-bottom: 20px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
  .badge.green { background: #1a3a1a; color: #4caf50; }
  .badge.red { background: #3a1a1a; color: #f44336; }
  .badge.blue { background: #1a2a3a; color: #64b5f6; }
  .timestamp { font-size: 11px; color: #555; margin-top: 8px; }
  .empty { color: #555; font-size: 13px; padding: 16px 0; }
</style>
</head>
<body>
<h1>&#x1F916; Coinbase Trading Bot</h1>
<div class="grid">
  <div class="card"><div class="label">USD Available</div><div class="value">$""" + "{:,.2f}".format(usd) + """</div></div>
  <div class="card"><div class="label">Open Positions</div><div class="value">""" + str(len(enriched)) + """</div></div>
  <div class="card"><div class="label">Total Trades</div><div class="value">""" + str(total_trades) + """</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value """ + win_color + """">""" + str(win_rate) + """%</div></div>
  <div class="card"><div class="label">Total P&amp;L</div><div class="value """ + pnl_color + """">$""" + pnl_sign + "{:,.4f}".format(total_pnl) + """</div></div>
</div>
<div class="section"><h2>Open Positions</h2>""" + pos_rows + """</div>
<div class="section"><h2>Trade History</h2>""" + hist_rows + """</div>
<div class="timestamp">Auto-refreshes every 60s &middot; """ + now + """ UTC</div>
</body>
</html>"""
    return html, 200


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
                stop_price, target_price, sl_order_id, tp_order_id,
                quote_size, round(prob, 4)
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

            close_position(product_id, exit_price=current_price, reason="signal")

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
