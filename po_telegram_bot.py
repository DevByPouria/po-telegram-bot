import os
import json
import gc
import threading
import requests
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import ta
from twelvedata import TDClient

app = Flask(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY") # کلید API را در Environment Variable رندر ذخیره کن

# ================== ۱۵ جفت‌ارز اصلی بازار (فرمت TwelveData) ==================
SYMBOLS = {
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD",
    "USD/CHF": "USD/CHF",
    "EUR/JPY": "EUR/JPY",
    "GBP/JPY": "GBP/JPY",
    "EUR/GBP": "EUR/GBP",
    "EUR/CHF": "EUR/CHF",
    "GBP/CHF": "GBP/CHF",
    "AUD/CAD": "AUD/CAD",
    "AUD/JPY": "AUD/JPY",
    "CAD/JPY": "CAD/JPY",
    "CHF/JPY": "CHF/JPY",
}

THRESHOLDS = [75, 78, 80, 82, 85]
PERIOD = 3000 # تعداد کندل‌ها (برای پوشش حدود ۶۰ روز در تایم‌فریم ۵ دقیقه)
INTERVAL = "5min"
EXPIRY = 3

backtest_running = False

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("BOT_TOKEN یا CHAT_ID تنظیم نشده")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"خطا در ارسال: {e}")

# ... (توابع الگوهای کندلی، زمینه روند، و score_signal دقیقاً مثل نسخه ۵ قبلی باقی می‌مانند) ...
# برای اختصار، فرض می‌کنیم که این توابع (bullish_engulfing, hammer, morning_star, piercing, 
# bearish_engulfing, shooting_star, evening_star, dark_cloud, trend_context_bullish, 
# trend_context_bearish, calc_indicators, score_signal) در اینجا تعریف شده‌اند.
# فقط تابع backtest_symbol تغییر می‌کند:

def backtest_symbol(name, symbol):
    try:
        print(f"⏳ دانلود {name} از TwelveData...")
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        ts = td.time_series(
            symbol=symbol,
            interval=INTERVAL,
            outputsize=PERIOD,
            timezone="UTC"
        )
        df = ts.as_pandas()
        
        if df is None or df.empty or len(df) < 250:
            return {"symbol": name, "error": "داده کافی نیست"}
        
        df = df.rename(columns=str.lower)
        df = df.sort_index() # اطمینان از مرتب بودن بر اساس زمان
        df = calc_indicators(df)
        print(f"✅ {name}: {len(df)} کندل دریافت شد")
        
        all_signals = []
        for i in range(50, len(df) - EXPIRY):
            sc, direction = score_signal(df, i)
            if direction is None or sc == 0:
                continue
            entry = df['close'].iloc[i]
            exit_p = df['close'].iloc[i + EXPIRY]
            if direction == 'CALL':
                win = exit_p > entry
            else:
                win = exit_p < entry
            all_signals.append({"score": sc, "win": win})
        
        results = {}
        for th in THRESHOLDS:
            filtered = [s for s in all_signals if s["score"] >= th]
            wins = sum(1 for s in filtered if s["win"])
            losses = len(filtered) - wins
            wr = (wins / len(filtered) * 100) if len(filtered) > 0 else 0
            results[th] = {"signals": len(filtered), "wins": wins, "losses": losses, "win_rate": round(wr, 2)}
        
        del df, all_signals
        gc.collect()
        return {"symbol": name, "results": results}
    except Exception as e:
        return {"symbol": name, "error": str(e)}

# ... (بقیه کد شامل run_backtest_background, روت‌ها, و main بدون تغییر باقی می‌ماند) ...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
