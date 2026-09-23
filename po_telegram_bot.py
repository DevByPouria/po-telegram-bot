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

SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURJPY": "EURJPY=X",
    "XAUUSD": "GC=F",
}

THRESHOLDS = [75, 80, 85, 90]
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

def calc_indicators(df):
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 14, 3).stoch()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['atr_ma'] = df['atr'].rolling(50).mean()
    return df

def score_signal(df, i):
    row = df.iloc[i]
    sc_call, sc_put = 0, 0
    
    # 1. روند قوی - EMA50 و EMA200 هم‌جهت (۳۰ امتیاز)
    if not pd.isna(row['ema50']) and not pd.isna(row['ema200']):
        if row['ema50'] > row['ema200'] and row['close'] > row['ema50']:
            sc_call += 30
        elif row['ema50'] < row['ema200'] and row['close'] < row['ema50']:
            sc_put += 30
    
    # 2. RSI اشباع (۲۰ امتیاز)
    if not pd.isna(row['rsi']):
        if row['rsi'] < 30: sc_call += 20
        elif row['rsi'] > 70: sc_put += 20
    
    # 3. الگوی کندلی (۲۰ امتیاز)
    if bullish_engulfing(df, i) or hammer(df, i): sc_call += 20
    if bearish_engulfing(df, i) or shooting_star(df, i): sc_put += 20
    
    # 4. استوکاستیک (۱۵ امتیاز)
    if not pd.isna(row['stoch']):
        if row['stoch'] < 20: sc_call += 15
        elif row['stoch'] > 80: sc_put += 15
    
    # 5. حمایت/مقاومت (۱۵ امتیاز)
    if near_support(df, i): sc_call += 15
    if near_resistance(df, i): sc_put += 15
    
    # فیلتر ATR: نوسان کافی
    atr_ok = True
    if not pd.isna(row['atr']) and not pd.isna(row['atr_ma']):
        if row['atr'] < row['atr_ma'] * 0.5:
            atr_ok = False
    
    if not atr_ok:
        return 0, None
    
    if sc_call > sc_put:
        return sc_call, 'CALL'
    elif sc_put > sc_call:
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
        
        # ذخیره تمام سیگنال‌ها با امتیاز
        all_signals = []
        for i in range(200, len(df) - expiry):
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
        
        # برای هر آستانه، نتایج رو جدا کن
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
        send_telegram("📊 <b>بک‌تست نسخه ۲ شروع شد</b>\n⏱ ۵ دقیقه | اکسپایر ۱۵ دقیقه | ۳۰ روز\n🎯 تست آستانه‌ها: 75, 80, 85, 90")
        
        # ساختار: {threshold: {wins, losses, signals}}
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
        
        # جمع کل برای هر آستانه
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
