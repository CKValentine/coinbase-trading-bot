import os
import ccxt
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Coinbase API setup
exchange = ccxt.coinbase({
    'apiKey': os.getenv('COINBASE_API_KEY'),
    'secret': os.getenv('COINBASE_API_SECRET'),
    'password': os.getenv('COINBASE_API_PASSPHRASE'),
    'enableRateLimit': True,
})

WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')

# Risk management params
RISK_PER_TRADE = 0.01  # 1% of account per trade
TARGET_ROI = 0.025  # 2.5% target per trade
STOP_LOSS_PCT = 0.015  # 1.5% stop loss

def get_account_balance():
    balance = exchange.fetch_balance()
    return balance['USDT']['free']  # Assuming base currency is USDT

def place_order(symbol, side, amount):
    try:
        order = exchange.create_order(symbol, 'market', side, amount)
        print(f"Order placed: {side} {amount} {symbol}")
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

def set_stop_loss(order, symbol):
    entry_price = order['price']
    sl_price = entry_price * (1 - STOP_LOSS_PCT) if order['side'] == 'buy' else entry_price * (1 + STOP_LOSS_PCT)
    exchange.create_order(symbol, 'stop_limit', 'sell' if order['side'] == 'buy' else 'buy', order['amount'], sl_price, {'trigger_price': sl_price})

def set_take_profit(order, symbol):
    entry_price = order['price']
    tp_price = entry_price * (1 + TARGET_ROI) if order['side'] == 'buy' else entry_price * (1 - TARGET_ROI)
    exchange.create_order(symbol, 'limit', 'sell' if order['side'] == 'buy' else 'buy', order['amount'], tp_price)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if data.get('secret') != WEBHOOK_SECRET:
        return jsonify({'status': 'error', 'message': 'Invalid secret'}), 403
    
    action = data.get('action')  # 'buy' or 'sell'
    symbol = data.get('symbol')  # e.g., 'SOL-USDT'
    balance = get_account_balance()
    amount = (balance * RISK_PER_TRADE) / exchange.fetch_ticker(symbol)['last']  # Calculate position size
    
    order = place_order(symbol, action, amount)
    if order:
        set_stop_loss(order, symbol)
        set_take_profit(order, symbol)
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(port=5000)