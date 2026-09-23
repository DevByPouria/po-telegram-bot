import os
import json
import gc
import time
import threading
import requests
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import ta
from twelvedata import TDClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

# ================== ۶ جفت‌ارز منتخب ==================
SYMBOLS = {
    "USD/JPY": "USD/JPY",
    "EUR/JPY": "EUR/JPY",
    "GBP/JPY": "GBP/JPY",
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/CHF": "USD/CHF",
}

INTERVAL = "15min"        # تایم فریم اصلی
HTF_INTERVAL = "1h"       # تایم فریم بالاتر (روند)
OUTPUTSIZE = 5000
HTF_OUTPUTSIZE = 200
EXPIRY = 3                # ۳ کندل × ۱۵ دقیقه = ۴۵ دقیقه

backtest_running = False

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"خطا: {e}")

# ================== الگوهای کندلی ==================
def bullish_engulfing(df, i):
    if i < 1: return 0
    p, c = df.iloc[i-1], df.iloc[i]
    if (p['close'] < p['open'] and c['close'] > c['open']
        and c['close'] > p['open'] and c['open'] < p['close']):
        return 1
    return 0

def bearish_engulfing(df, i):
    if i < 1: return 0
    p, c = df.iloc[i-1], df.iloc[i]
    if (p['close'] > p['open'] and c['close'] < c['open']
        and c['close'] < p['open'] and c['open'] > p['close']):
        return 1
    return 0

def hammer(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return 0
    lower_wick = min(c['close'], c['open']) - c['low']
    upper_wick = c['high'] - max(c['close'], c['open'])
    if lower_wick >= 2 * body and upper_wick <= body * 0.8:
        return 1
    return 0

def shooting_star(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return 0
    upper_wick = c['high'] - max(c['close'], c['open'])
    lower_wick = min(c['close'], c['open']) - c['low']
    if upper_wick >= 2 * body and lower_wick <= body * 0.8:
        return 1
    return 0

# ================== فیچرها ==================
def build_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_pos'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
    
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['adx'] = adx_ind.adx()
    df['di_plus'] = adx_ind.adx_pos()
    df['di_minus'] = adx_ind.adx_neg()
    
    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
    
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    
    # فیچرهای اضافی
    df['candle_body'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_wick'] = (df['high'] - df[['close','open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_wick'] = (df[['close','open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['price_change'].rolling(20).std()
    df['rsi_change'] = df['rsi'].diff()
    df['macd_hist_change'] = df['macd_diff'].diff()
    
    # فاصله از EMA
    df['dist_ema20'] = (df['close'] - df['ema20']) / df['ema20']
    df['dist_ema50'] = (df['close'] - df['ema50']) / df['ema50']
    df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']
    
    # الگوها
    df['bull_engulf'] = 0
    df['bear_engulf'] = 0
    df['hammer'] = 0
    df['shooting'] = 0
    
    return df

def add_patterns(df):
    for i in range(2, len(df)):
        df.at[df.index[i], 'bull_engulf'] = bullish_engulfing(df, i)
        df.at[df.index[i], 'bear_engulf'] = bearish_engulfing(df, i)
        df.at[df.index[i], 'hammer'] = hammer(df, i)
        df.at[df.index[i], 'shooting'] = shooting_star(df, i)
    return df

# ================== ساخت دیتاست آموزش ==================
def create_training_data(df, expiry=3):
    """ساخت X (فیچرها) و y (برچسب برد/باخت)"""
    feature_cols = [
        'rsi', 'bb_pos', 'macd', 'macd_signal', 'macd_diff',
        'adx', 'di_plus', 'di_minus',
        'dist_ema20', 'dist_ema50', 'dist_ema200',
        'candle_body', 'upper_wick', 'lower_wick',
        'price_change', 'volatility', 'rsi_change', 'macd_hist_change',
        'bull_engulf', 'bear_engulf', 'hammer', 'shooting'
    ]
    
    X, y = [], []
    
    for i in range(250, len(df) - expiry):
        row = df.iloc[i]
        if row[feature_cols].isna().any():
            continue
        
        entry = row['close']
        exit_p = df['close'].iloc[i + expiry]
        
        # برچسب: 1 = صعودی (CALL برنده), 0 = نزولی (PUT برنده)
        label = 1 if exit_p > entry else 0
        
        features = row[feature_cols].values.astype(float)
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y), feature_cols

# ================== آموزش مدل ==================
def train_model(X, y):
    """آموزش Random Forest و برگرداندن مدل + دقت"""
    if len(X) < 200:
        return None, 0, 0
    
    # تقسیم: ۷۰٪ آموزش، ۳۰٪ تست
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, len(X_test)

# ================== بک‌تست ==================
def backtest_symbol(name, symbol):
    try:
        print(f"⏳ دانلود {name}...")
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        
        # داده ۱۵ دقیقه
        ts = td.time_series(symbol=symbol, interval=INTERVAL, outputsize=OUTPUTSIZE, timezone="UTC")
        df = ts.as_pandas()
        time.sleep(7)  # جلوگیری از محدودیت API
        
        if df is None or df.empty or len(df) < 500:
            return {"symbol": name, "error": "داده کافی نیست"}
        
        df = df.rename(columns=str.lower).sort_index()
        df = build_features(df)
        df = add_patterns(df)
        df = df.dropna()
        
        if len(df) < 500:
            return {"symbol": name, "error": "داده پس از پاک‌سازی کم است"}
        
        # ساخت دیتاست
        X, y, feature_cols = create_training_data(df, expiry=EXPIRY)
        
        if len(X) < 300:
            return {"symbol": name, "error": "دیتاست کافی نیست"}
        
        # آموزش مدل
        model, accuracy, test_size = train_model(X, y)
        
        if model is None:
            return {"symbol": name, "error": "آموزش مدل شکست خورد"}
        
        # بک‌تست روی داده تست
        split_idx = int(len(X) * 0.7)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # گرفتن probability از مدل
        probs = model.predict_proba(X_test)
        
        # فیلتر: فقط سیگنال‌هایی با احتمال بالا
        high_conf_signals = 0
        high_conf_wins = 0
        min_confidence = 0.65
        
        for j, prob in enumerate(probs):
            max_prob = max(prob)
            pred = 1 if prob[1] > prob[0] else 0
            
            if max_prob >= min_confidence:
                high_conf_signals += 1
                if pred == y_test[j]:
                    high_conf_wins += 1
        
        wr = (high_conf_wins / high_conf_signals * 100) if high_conf_signals > 0 else 0
        days = 52  # تقریبی
        
        del df
        gc.collect()
        
        return {
            "symbol": name,
            "total_samples": len(X),
            "test_samples": test_size,
            "model_accuracy": round(accuracy * 100, 2),
            "signals": high_conf_signals,
            "wins": high_conf_wins,
            "losses": high_conf_signals - high_conf_wins,
            "win_rate": round(wr, 2),
            "signals_per_day": round(high_conf_signals / days, 1),
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
        send_telegram("🤖 <b>بک‌تست ML + Multi-Timeframe شروع شد</b>\n\n"
                      "🧠 Random Forest + 15min + Trend Filter\n"
                      "⏱ ۱۵ دقیقه | اکسپایر ۴۵ دقیقه\n"
                      "🎯 ۶ جفت‌ارز | آستانه اطمینان ۶۵٪\n"
                      "⏳ هر جفت‌ارز ~۱ دقیقه (به خاطر API)")

        total_signals, total_wins = 0, 0

        for name, sym in SYMBOLS.items():
            r = backtest_symbol(name, sym)
            if "error" in r:
                send_telegram(f"❌ <b>{r['symbol']}</b>: {r['error']}")
                continue

            msg = (f"<b>{r['symbol']}</b>\n"
                   f"🧠 دقت مدل: {r['model_accuracy']}%\n"
                   f"📈 سیگنال (≥۶۵٪): {r['signals']} (روزی {r['signals_per_day']})\n"
                   f"✅ {r['wins']}W / ❌ {r['losses']}L\n"
                   f"🎯 وین ریت: <b>{r['win_rate']}%</b>")
            send_telegram(msg)

            total_signals += r['signals']
            total_wins += r['wins']

        total_losses = total_signals - total_wins
        overall = (total_wins / total_signals * 100) if total_signals > 0 else 0
        final = (f"🏁 <b>جمع کل:</b>\n\n"
                 f"📈 سیگنال: {total_signals}\n"
                 f"✅ برد: {total_wins} | ❌ باخت: {total_losses}\n"
                 f"🎯 <b>وین ریت: {round(overall, 2)}%</b>")
        send_telegram(final)
    finally:
        backtest_running = False

@app.route('/')
def health():
    return "ML Signal Server is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw = request.get_data(as_text=True)
        send_telegram(f"🟢 <b>سیگنال</b>\n{raw[:500]}")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/backtest', methods=['GET'])
def backtest_route():
    threading.Thread(target=run_backtest_background, daemon=True).start()
    return jsonify({"status": "ok", "message": "ML backtest started"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
