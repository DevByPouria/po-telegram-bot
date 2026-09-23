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

# ================== ۱۰ جفت‌ارز (۵ ین + ۵ اصلی) ==================
SYMBOLS = {
    # جفت‌های ین (بهترین عملکرد قبلی)
    "USD/JPY": "USD/JPY",
    "EUR/JPY": "EUR/JPY",
    "GBP/JPY": "GBP/JPY",
    "AUD/JPY": "AUD/JPY",
    "CHF/JPY": "CHF/JPY",
    # جفت‌های اصلی
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/CHF": "USD/CHF",
    "USD/CAD": "USD/CAD",
    "AUD/USD": "AUD/USD",
}

INTERVAL = "5min"
OUTPUTSIZE = 5000
EXPIRY = 3

# ================== فیلتر ساعت (۸-۲۰ UTC) ==================
SESSION_START_UTC = 8
SESSION_END_UTC = 20

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

def is_session_active(dt):
    return SESSION_START_UTC <= dt.hour < SESSION_END_UTC

# ================== الگوهای کندلی ==================
def bullish_engulfing(df, i):
    if i < 1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    return (p['close'] < p['open'] and c['close'] > c['open']
            and c['close'] > p['open'] and c['open'] < p['close']
            and abs(c['close'] - c['open']) > abs(p['close'] - p['open']) * 1.1)

def bearish_engulfing(df, i):
    if i < 1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    return (p['close'] > p['open'] and c['close'] < c['open']
            and c['close'] < p['open'] and c['open'] > p['close']
            and abs(c['close'] - c['open']) > abs(p['close'] - p['open']) * 1.1)

def hammer(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return False
    lower_wick = min(c['close'], c['open']) - c['low']
    upper_wick = c['high'] - max(c['close'], c['open'])
    return lower_wick >= 2 * body and upper_wick <= body * 0.8

def shooting_star(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return False
    upper_wick = c['high'] - max(c['close'], c['open'])
    lower_wick = min(c['close'], c['open']) - c['low']
    return upper_wick >= 2 * body and lower_wick <= body * 0.8

def bullish_pattern(df, i):
    return bullish_engulfing(df, i) or hammer(df, i)

def bearish_pattern(df, i):
    return bearish_engulfing(df, i) or shooting_star(df, i)

# ================== اندیکاتورها ==================
def calc_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['adx'] = adx_ind.adx()
    df['di_plus'] = adx_ind.adx_pos()
    df['di_minus'] = adx_ind.adx_neg()
    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    return df

# ================== هسته تشخیص سیگنال ==================
def get_signal(df, i):
    row = df.iloc[i]
    
    if pd.isna(row['adx']) or pd.isna(row['rsi']) or pd.isna(row['bb_high']) or pd.isna(row['ema20']):
        return None
    
    adx = row['adx']
    rsi = row['rsi']
    price = row['close']
    di_plus = row['di_plus']
    di_minus = row['di_minus']
    
    # ==================== حالت ۱: روند قوی (ADX > 25) ====================
    if adx > 25:
        # CALL: روند صعودی + پول‌بک به EMA20
        if di_plus > di_minus and price > row['ema50']:
            distance_to_ema20 = abs(price - row['ema20']) / price
            if distance_to_ema20 < 0.0015 and 35 < rsi < 65:
                if bullish_pattern(df, i):
                    return "CALL"
        
        # PUT: روند نزولی + پول‌بک به EMA20
        if di_minus > di_plus and price < row['ema50']:
            distance_to_ema20 = abs(price - row['ema20']) / price
            if distance_to_ema20 < 0.0015 and 35 < rsi < 65:
                if bearish_pattern(df, i):
                    return "PUT"
    
    # ==================== حالت ۲: رنج (ADX ≤ 25) ====================
    else:
        # CALL: اشباع فروش + زیر BB پایین
        if rsi < 30 and price <= row['bb_low']:
            if bullish_pattern(df, i):
                return "CALL"
        
        # PUT: اشباع خرید + بالای BB بالا
        if rsi > 70 and price >= row['bb_high']:
            if bearish_pattern(df, i):
                return "PUT"
    
    return None

def backtest_symbol(name, symbol):
    try:
        print(f"⏳ دانلود {name}...")
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

        total_signals = 0
        wins = 0
        losses = 0
        trend_calls = 0
        trend_puts = 0
        range_calls = 0
        range_puts = 0
        days_with_data = set()

        for i in range(50, len(df) - EXPIRY):
            try:
                candle_time = df.index[i]
                if hasattr(candle_time, 'hour'):
                    if not is_session_active(candle_time):
                        continue
                    days_with_data.add(candle_time.date())
            except:
                pass
            
            direction = get_signal(df, i)
            if direction is None:
                continue
            
            entry = df['close'].iloc[i]
            exit_p = df['close'].iloc[i + EXPIRY]
            adx_val = df['adx'].iloc[i]
            
            total_signals += 1
            
            if adx_val > 25:
                if direction == 'CALL': trend_calls += 1
                else: trend_puts += 1
            else:
                if direction == 'CALL': range_calls += 1
                else: range_puts += 1
            
            if direction == 'CALL':
                is_win = exit_p > entry
            else:
                is_win = exit_p < entry
            
            if is_win:
                wins += 1
            else:
                losses += 1

        total = wins + losses
        wr = (wins / total * 100) if total > 0 else 0
        num_days = len(days_with_data) if days_with_data else 1
        signals_per_day = round(total_signals / num_days, 1)

        del df
        gc.collect()
        return {
            "symbol": name,
            "signals": total_signals,
            "signals_per_day": signals_per_day,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wr, 2),
            "trend_calls": trend_calls,
            "trend_puts": trend_puts,
            "range_calls": range_calls,
            "range_puts": range_puts,
            "days": num_days,
        }
    except Exception as e:
        return {"symbol": name, "error": str(e)}

def run_backtest_background():
    global backtest_running
    if backtest_running:
        send_telegram("⚠️ بک‌تست قبلی هنوز در حال اجراست.")
        return
    backtest_running = True
    try:
        send_telegram("📊 <b>بک‌تست نسخه نهایی شروع شد</b>\n\n"
                      "🧠 ADX Regime + Pullback + Reversion\n"
                      "⏰ سشن ۸-۲۰ UTC (۱۲ ساعت)\n"
                      "🎯 ۱۰ جفت‌ارز (۵ ین + ۵ اصلی)\n"
                      "⏱ ۵ دقیقه | اکسپایر ۱۵ دقیقه\n"
                      "🎯 هدف: روزی ۵-۱۰ سیگنال")

        total_w, total_l, total_s = 0, 0, 0
        all_days = set()

        for name, sym in SYMBOLS.items():
            r = backtest_symbol(name, sym)
            if "error" in r:
                send_telegram(f"❌ <b>{r['symbol']}</b>: {r['error']}")
                continue

            msg = (f"<b>{r['symbol']}</b>\n"
                   f"📈 سیگنال: {r['signals']} (روزی {r['signals_per_day']})\n"
                   f"✅ {r['wins']}W / ❌ {r['losses']}L\n"
                   f"🎯 وین ریت: <b>{r['win_rate']}%</b>\n"
                   f"📊 روند: {r['trend_calls']}C/{r['trend_puts']}P | "
                   f"رنج: {r['range_calls']}C/{r['range_puts']}P")
            send_telegram(msg)

            total_w += r['wins']
            total_l += r['losses']
            total_s += r['signals']

        tot = total_w + total_l
        overall = (total_w / tot * 100) if tot > 0 else 0
        # تخمین روزها بر اساس ۵۲ روز معاملاتی
        estimated_days = 52
        signals_per_day_total = round(total_s / estimated_days, 1) if estimated_days > 0 else 0
        
        final = (f"🏁 <b>جمع کل:</b>\n\n"
                 f"📈 سیگنال: {total_s}\n"
                 f"📅 تخمین روزانه: ~{signals_per_day_total} سیگنال\n"
                 f"✅ برد: {total_w} | ❌ باخت: {total_l}\n"
                 f"🎯 <b>وین ریت: {round(overall, 2)}%</b>")
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
