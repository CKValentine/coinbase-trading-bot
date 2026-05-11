#!/bin/bash
# run_update.sh — called by LaunchAgent every Sunday at 2am
# Updates CSVs, retrains models, and restarts the bot

export HOME=/Users/ckvalentine
export PATH=/usr/local/bin:/usr/bin:/bin
export PWD=/Users/ckvalentine/coinbase-trading-bot

cd /Users/ckvalentine/coinbase-trading-bot

echo "=== Starting weekly update: $(date) ==="

# Run update and retrain
/usr/bin/python3 update_and_retrain.py

# Restart the bot to load new models
launchctl unload /Users/ckvalentine/Library/LaunchAgents/com.tradingbot.plist
sleep 3
launchctl load /Users/ckvalentine/Library/LaunchAgents/com.tradingbot.plist

echo "=== Update complete: $(date) ==="
