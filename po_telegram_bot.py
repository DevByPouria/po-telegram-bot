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
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

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
INTERVAL = "5min"
OUTPUTSIZE = 3000
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

# ================== الگوهای کندلی ==================
def bullish_engulfing(df, i):
    if i < 1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    return (p['close'] < p['open'] and c['close'] > c['open']
            and c['close'] > p['open'] and c['open'] < p['close'])

def bearish_engulfing(df, i):
    if i < 1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    return (p['close'] > p['open'] and c['close'] < c['open']
            and c['close'] < p['open'] and c['open'] > p['close'])

def hammer(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return False
    lower_wick = min(c['close'], c['open']) - c['low']
    upper_wick = c['high'] - max(c['close'], c['open'])
    return lower_wick >= 2.5 * body and upper_wick <= body * 0.7

def shooting_star(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return False
    upper_wick = c['high'] - max(c['close'], c['open'])
    lower_wick = min(c['close'], c['open']) - c['low']
    return upper_wick >= 2.5 * body and lower_wick <= body * 0.7

def morning_star(df, i):
    if i < 2: return False
    c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    body1 = abs(c1['close'] - c1['open'])
    body2 = abs(c2['close'] - c2['open'])
    body3 = abs(c3['close'] - c3['open'])
    if body1 == 0: return False
    return (c1['close'] < c1['open'] and body1 > body2 * 1.2
            and c3['close'] > c3['open'] and body3 > body2 * 1.2
            and c3['close'] > (c1['open'] + c1['close']) / 2)

def evening_star(df, i):
    if i < 2: return False
    c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    body1 = abs(c1['close'] - c1['open'])
    body2 = abs(c2['close'] - c2['open'])
    body3 = abs(c3['close'] - c3['open'])
    if body1 == 0: return False
    return (c1['close'] > c1['open'] and body1 > body2 * 1.2
            and c3['close'] < c3['open'] and body3 > body2 * 1.2
            and c3['close'] < (c1['open'] + c1['close']) / 2)

def piercing(df, i):
    if i < 1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    if p['close'] >= p['open']: return False
    if c['close'] <= c['open']: return False
    mid = (p['open'] + p['close']) / 2
    return c['open'] < p['close'] and c['close'] > mid and c['close'] < p['open']

def dark_cloud(df, i):
    if i < 1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    if p['close'] <= p['open']: return False
    if c['close'] >= c['open']: return False
    mid = (p['open'] + p['close']) / 2
    return c['open'] > p['close'] and c['close'] < mid and c['close'] > p['open']

def trend_context_bullish(df, i, lookback=5):
    if i < lookback: return False
    count = 0
    for j in range(i-lookback, i):
        if df.iloc[j]['close'] < df.iloc[j]['open']:
            count += 1
    return count >= 3

def trend_context_bearish(df, i, lookback=5):
    if i < lookback: return False
    count = 0
    for j in range(i-lookback, i):
        if df.iloc[j]['close'] > df.iloc[j]['open']:
            count += 1
    return count >= 3

def calc_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['atr_ma'] = df['atr'].rolling(50).mean()
    return df

def score_signal(df, i):
    row = df.iloc[i]
    sc_call, sc_put = 0, 0

    if not pd.isna(row['rsi']):
        if row['rsi'] < 22: sc_call += 25
        elif row['rsi'] < 28: sc_call += 15
        elif row['rsi'] > 78: sc_put += 25
        elif row['rsi'] > 72: sc_put += 15

    if not pd.isna(row['bb_low']) and not pd.isna(row['bb_high']):
        price = row['close']
        if price <= row['bb_low']: sc_call += 20
        elif price >= row['bb_high']: sc_put += 20

    bull = bullish_engulfing(df, i) or hammer(df, i) or morning_star(df, i) or piercing(df, i)
    bear = bearish_engulfing(df, i) or shooting_star(df, i) or evening_star(df, i) or dark_cloud(df, i)
    if bull: sc_call += 30
    if bear: sc_put += 30

    if trend_context_bullish(df, i): sc_call += 25
    if trend_context_bearish(df, i): sc_put += 25

    if not pd.isna(row['atr']) and not pd.isna(row['atr_ma']):
        if row['atr'] < row['atr_ma'] * 0.6:
            return 0, None

    if sc_call > sc_put and sc_call > 0:
        return sc_call, 'CALL'
    elif sc_put > sc_call and sc_put > 0:
        return sc_put, 'PUT'
    return 0, None

def backtest_symbol(name, symbol):
    try:
        print(f"⏳ دانلود {name} از TwelveData...")
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        ts = td.time_series(
            symbol=symbol,
            interval=INTERVAL,
            outputsize=OUTPUTSIZE,
            timezone="UTC"
        )
        df = ts.as_pandas()

        if df is None or df.empty or len(df) < 250:
            return {"symbol": name, "error": "داده کافی نیست"}

        df = df.rename(columns=str.lower)
        df = df.sort_index()
        df = calc_indicators(df)
        print(f"✅ {name}: {len(df)} کندل")

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

def run_backtest_background():
    global backtest_running
    if backtest_running:
        send_telegram("⚠️ بک‌تست قبلی هنوز در حال اجراست.")
        return
    backtest_running = True
    try:
        send_telegram("📊 <b>بک‌تست TwelveData شروع شد</b>\n"
                      "⏱ ۵ دقیقه | اکسپایر ۱۵ دقیقه | ۱۵ جفت‌ارز\n"
                      "🎯 آستانه‌ها: 75/78/80/82/85")

        total = {th: {"wins": 0, "losses": 0, "signals": 0} for th in THRESHOLDS}

        for name, sym in SYMBOLS.items():
            r = backtest_symbol(name, sym)
            if "error" in r:
                send_telegram(f"❌ <b>{r['symbol']}</b>: {r['error']}")
                continue

            msg = f"<b>{r['symbol']}</b>\n"
            for th in THRESHOLDS:
                res = r["results"][th]
                msg += f"  {th}: {res['signals']} | {res['wins']}W/{res['losses']}L | <b>{res['win_rate']}%</b>\n"
                total[th]["wins"] += res["wins"]
                total[th]["losses"] += res["losses"]
                total[th]["signals"] += res["signals"]
            send_telegram(msg)

        final = "🏁 <b>جمع کل:</b>\n\n"
        for th in THRESHOLDS:
            t = total[th]
            tot = t["wins"] + t["losses"]
            wr = (t["wins"] / tot * 100) if tot > 0 else 0
            final += f"<b>آستانه {th}:</b>\n"
            final += f"  📈 {t['signals']} سیگنال | ✅ {t['wins']} | ❌ {t['losses']}\n"
            final += f"  🎯 وین ریت: <b>{round(wr, 2)}%</b>\n\n"
        send_telegram(final)
    finally:
        backtest_running = False

@app.route('/')
def health():
    return "Signal Server is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw = request.get_data(as_text=True)
        try: signal = json.loads(raw)
        except: signal = {"raw": raw}
        action = signal.get("action", "?")
        symbol = signal.get("symbol", "?")
        price = signal.get("price", "?")
        msg = f"🟢 <b>سیگنال</b>\n📊 {symbol}\n💰 {price}\n📌 {action}"
        send_telegram(msg)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/backtest', methods=['GET'])
def backtest_route():
    threading.Thread(target=run_backtest_background, daemon=True).start()
    return jsonify({"status": "ok", "message": "بک‌تست شروع شد"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
