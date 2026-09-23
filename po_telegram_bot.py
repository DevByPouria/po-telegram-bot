import os
import json
import gc
import threading
import requests
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import ta

app = Flask(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# ================== همه جفت‌ارزهای Pocket Option ==================
SYMBOLS = {
    "AUD/CAD": "AUDCAD=X",
    "AUD/USD": "AUDUSD=X",
    "CHF/JPY": "CHFJPY=X",
    "EUR/CHF": "EURCHF=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "GBP/CHF": "GBPCHF=X",
    "EUR/USD": "EURUSD=X",
    "GBP/JPY": "GBPJPY=X",
    "GBP/CAD": "GBPCAD=X",
    "GBP/AUD": "GBPAUD=X",
    "AUD/JPY": "AUDJPY=X",
    "CAD/JPY": "CADJPY=X",
    "USD/CHF": "USDCHF=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/CHF": "AUDCHF=X",
    "CAD/CHF": "CADCHF=X",
    "USD/CAD": "USDCAD=X",
    "EUR/CAD": "EURCAD=X",
    "EUR/AUD": "EURAUD=X",
}

THRESHOLDS = [60, 70, 80, 90]
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
    return p['close'] < p['open'] and c['close'] > c['open'] and c['close'] > p['open'] and c['open'] < p['close']

def bearish_engulfing(df, i):
    if i < 1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    return p['close'] > p['open'] and c['close'] < c['open'] and c['close'] < p['open'] and c['open'] > p['close']

def hammer(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return False
    low_wick = min(c['close'], c['open']) - c['low']
    high_wick = c['high'] - max(c['close'], c['open'])
    return low_wick > 2 * body and high_wick < body

def shooting_star(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return False
    low_wick = min(c['close'], c['open']) - c['low']
    high_wick = c['high'] - max(c['close'], c['open'])
    return high_wick > 2 * body and low_wick < body

# ================== محاسبه اندیکاتورها ==================
def calc_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['atr_ma'] = df['atr'].rolling(50).mean()
    return df

# ================== امتیازدهی بازگشتی ==================
def score_signal(df, i):
    row = df.iloc[i]
    sc_call, sc_put = 0, 0
    
    # 1. RSI اشباع - ۳۵ امتیاز
    if not pd.isna(row['rsi']):
        if row['rsi'] < 25: sc_call += 35
        elif row['rsi'] < 30: sc_call += 20
        elif row['rsi'] > 75: sc_put += 35
        elif row['rsi'] > 70: sc_put += 20
    
    # 2. Bollinger Bands - ۳۵ امتیاز
    if not pd.isna(row['bb_low']) and not pd.isna(row['bb_high']):
        price = row['close']
        if price <= row['bb_low']: sc_call += 35
        elif price >= row['bb_high']: sc_put += 35
    
    # 3. الگوی کندلی - ۳۰ امتیاز
    if bullish_engulfing(df, i) or hammer(df, i): sc_call += 30
    if bearish_engulfing(df, i) or shooting_star(df, i): sc_put += 30
    
    # فیلتر ATR: نوسان کافی
    if not pd.isna(row['atr']) and not pd.isna(row['atr_ma']):
        if row['atr'] < row['atr_ma'] * 0.5:
            return 0, None
    
    if sc_call > sc_put and sc_call > 0:
        return sc_call, 'CALL'
    elif sc_put > sc_call and sc_put > 0:
        return sc_put, 'PUT'
    return 0, None

def backtest_symbol(name, yf_sym, period="30d", interval="5m", expiry=3):
    try:
        print(f"⏳ دانلود {name}...")
        df = yf.download(yf_sym, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or len(df) < 250:
            return {"symbol": name, "error": "داده کافی نیست"}
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        df = calc_indicators(df)
        print(f"✅ {name}: {len(df)} کندل")
        
        all_signals = []
        for i in range(50, len(df) - expiry):
            sc, direction = score_signal(df, i)
            if direction is None or sc == 0:
                continue
            entry = df['close'].iloc[i]
            exit_p = df['close'].iloc[i + expiry]
            if direction == 'CALL':
                win = exit_p > entry
            else:
                win = exit_p < entry
            all_signals.append({"score": sc, "direction": direction, "win": win})
        
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
        send_telegram("📊 <b>بک‌تست استراتژی بازگشتی شروع شد</b>\n⏱ ۵ دقیقه | اکسپایر ۱۵ دقیقه | ۳۰ روز\n🧠 RSI + Bollinger + الگوی کندلی")
        
        total = {th: {"wins": 0, "losses": 0, "signals": 0} for th in THRESHOLDS}
        
        for name, sym in SYMBOLS.items():
            r = backtest_symbol(name, sym)
            if "error" in r:
                send_telegram(f"❌ <b>{r['symbol']}</b>: {r['error']}")
                continue
            
            msg = f"<b>{r['symbol']}</b>\n"
            for th in THRESHOLDS:
                res = r["results"][th]
                msg += f"  آستانه {th}: {res['signals']} سیگنال | {res['wins']}W/{res['losses']}L | <b>{res['win_rate']}%</b>\n"
                total[th]["wins"] += res["wins"]
                total[th]["losses"] += res["losses"]
                total[th]["signals"] += res["signals"]
            send_telegram(msg)
        
        final = "🏁 <b>جمع کل بر اساس آستانه:</b>\n\n"
        for th in THRESHOLDS:
            t = total[th]
            tot = t["wins"] + t["losses"]
            wr = (t["wins"] / tot * 100) if tot > 0 else 0
            final += f"<b>آستانه {th}:</b>\n"
            final += f"  📈 {t['signals']} سیگنال | ✅ {t['wins']} | ❌ {t['losses']} | 🎯 <b>{round(wr, 2)}%</b>\n\n"
        send_telegram(final)
    finally:
        backtest_running = False

# ================== روت‌ها ==================
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
