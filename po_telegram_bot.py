import os
import json
import requests
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import ta

app = Flask(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURJPY": "EURJPY=X",
    "XAUUSD": "GC=F",
}

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
    low_wick = min(c['close'], c['open']) - c['low']
    high_wick = c['high'] - max(c['close'], c['open'])
    if body == 0: return False
    return low_wick > 2 * body and high_wick < body

def shooting_star(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    low_wick = min(c['close'], c['open']) - c['low']
    high_wick = c['high'] - max(c['close'], c['open'])
    if body == 0: return False
    return high_wick > 2 * body and low_wick < body

# ================== حمایت و مقاومت ==================
def near_support(df, i, lookback=30):
    if i < lookback: return False
    recent_low = df['low'].iloc[i-lookback:i].min()
    price = df['close'].iloc[i]
    return abs(price - recent_low) / price < 0.001

def near_resistance(df, i, lookback=30):
    if i < lookback: return False
    recent_high = df['high'].iloc[i-lookback:i].max()
    price = df['close'].iloc[i]
    return abs(price - recent_high) / price < 0.001

# ================== محاسبه اندیکاتورها ==================
def calc_indicators(df):
    df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 14, 3).stoch()
    return df

# ================== امتیازدهی ==================
def score_signal(df, i):
    row = df.iloc[i]
    sc_call, sc_put = 0, 0
    
    # 1. روند - ۲۵ امتیاز
    if not pd.isna(row['ema200']):
        if row['close'] > row['ema200']: sc_call += 25
        elif row['close'] < row['ema200']: sc_put += 25
    
    # 2. RSI - ۲۰ امتیاز
    if not pd.isna(row['rsi']):
        if row['rsi'] < 30: sc_call += 20
        elif row['rsi'] > 70: sc_put += 20
    
    # 3. الگوی کندلی - ۲۵ امتیاز
    if bullish_engulfing(df, i) or hammer(df, i): sc_call += 25
    if bearish_engulfing(df, i) or shooting_star(df, i): sc_put += 25
    
    # 4. استوکاستیک - ۱۵ امتیاز
    if not pd.isna(row['stoch']):
        if row['stoch'] < 20: sc_call += 15
        elif row['stoch'] > 80: sc_put += 15
    
    # 5. حمایت/مقاومت - ۱۵ امتیاز
    if near_support(df, i): sc_call += 15
    if near_resistance(df, i): sc_put += 15
    
    if sc_call >= 75 and sc_call > sc_put:
        return sc_call, 'CALL'
    elif sc_put >= 75 and sc_put > sc_call:
        return sc_put, 'PUT'
    return max(sc_call, sc_put), None

# ================== بک‌تست ==================
def backtest_symbol(name, yf_sym, period="7d", interval="1m", expiry=5, threshold=75):
    try:
        df = yf.download(yf_sym, period=period, interval=interval, progress=False)
        if df.empty or len(df) < 250:
            return {"symbol": name, "error": "داده کافی نیست"}
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        df = calc_indicators(df)
        
        wins, losses, signals = 0, 0, 0
        for i in range(200, len(df) - expiry):
            sc, direction = score_signal(df, i)
            if direction is None or sc < threshold:
                continue
            signals += 1
            entry = df['close'].iloc[i]
            exit_p = df['close'].iloc[i + expiry]
            if direction == 'CALL':
                if exit_p > entry: wins += 1
                else: losses += 1
            else:
                if exit_p < entry: wins += 1
                else: losses += 1
        
        total = wins + losses
        wr = (wins / total * 100) if total > 0 else 0
        return {"symbol": name, "signals": signals, "wins": wins, "losses": losses, "win_rate": round(wr, 2)}
    except Exception as e:
        return {"symbol": name, "error": str(e)}

def run_backtest():
    msg = "📊 <b>بک‌تست استراتژی امتیازدهی</b>\n\n"
    msg += f"⏱ تایم‌فریم: ۱ دقیقه | اکسپایر: ۵ دقیقه | آستانه: ۷۵\n\n"
    
    total_w, total_l, total_s = 0, 0, 0
    for name, sym in SYMBOLS.items():
        r = backtest_symbol(name, sym)
        if "error" in r:
            msg += f"❌ {r['symbol']}: {r['error']}\n"
            continue
        msg += f"<b>{r['symbol']}</b>: {r['signals']} سیگنال | {r['wins']}W/{r['losses']}L | <b>{r['win_rate']}%</b>\n"
        total_w += r['wins']; total_l += r['losses']; total_s += r['signals']
    
    tot = total_w + total_l
    overall = (total_w / tot * 100) if tot > 0 else 0
    msg += f"\n🎯 <b>جمع کل:</b> {total_s} سیگنال | {total_w}W/{total_l}L | <b>{round(overall, 2)}%</b>"
    send_telegram(msg)
    print(msg)

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
    try:
        run_backtest()
        return jsonify({"status": "ok", "message": "نتایج به تلگرام ارسال شد"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
