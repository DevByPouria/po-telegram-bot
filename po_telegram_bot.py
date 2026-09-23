import os
import gc
import time
import threading
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import ta
from twelvedata import TDClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

# ================== تنظیمات ==================
SYMBOLS = {
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "EUR/JPY": "EUR/JPY",
}

INTERVAL = "15min"
HTF_INTERVAL = "1h"
OUTPUTSIZE = 5000
HTF_OUTPUTSIZE = 800
EXPIRY_CANDLES = 2  # ۲ کندل ۱۵ دقیقه = ۳۰ دقیقه
THRESHOLD = 70  # آستانه اطمینان
N_FOLDS = 4
MIN_TRAIN_RATIO = 0.4

# ================== تنظیمات بانک و ریسک ==================
INITIAL_BANKROLL = 1000.0
PAYOUT = 0.85  # ۸۵٪ سود در هر برد
STAKE_PCT = 0.01  # ۱٪ ریسک

# ================== تایم‌زون ایران ==================
IRAN_TZ = timezone(timedelta(hours=3, minutes=30))

# ================== وضعیت ==================
backtest_running = False
live_running = False
trained_models = {}  # {symbol: model}

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=15)
    except Exception as e:
        print(f"telegram error: {e}")

def to_iran(dt):
    """تبدیل datetime به وقت ایران"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IRAN_TZ)

def fmt_iran(dt):
    """فرمت خوانای وقت ایران"""
    return to_iran(dt).strftime("%H:%M")

# ================== الگوهای کندلی ==================
def bull_engulf(df, i):
    if i < 1: return 0
    p, c = df.iloc[i-1], df.iloc[i]
    return int(p['close'] < p['open'] and c['close'] > c['open'] and c['close'] > p['open'] and c['open'] < p['close'])

def bear_engulf(df, i):
    if i < 1: return 0
    p, c = df.iloc[i-1], df.iloc[i]
    return int(p['close'] > p['open'] and c['close'] < c['open'] and c['close'] < p['open'] and c['open'] > p['close'])

def hammer_p(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return 0
    lw = min(c['close'], c['open']) - c['low']
    uw = c['high'] - max(c['close'], c['open'])
    return int(lw >= 2 * body and uw <= body * 0.8)

def shooting_p(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return 0
    uw = c['high'] - max(c['close'], c['open'])
    lw = min(c['close'], c['open']) - c['low']
    return int(uw >= 2 * body and lw <= body * 0.8)

# ================== Features ==================
def build_ltf_features(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_pos'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
    m = ta.trend.MACD(df['close'])
    df['macd'] = m.macd()
    df['macd_signal'] = m.macd_signal()
    df['macd_diff'] = m.macd_diff()
    a = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['adx'] = a.adx()
    df['di_plus'] = a.adx_pos()
    df['di_minus'] = a.adx_neg()
    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['candle_body'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_wick'] = (df['high'] - df[['close','open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_wick'] = (df[['close','open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['price_change'].rolling(20).std()
    df['rsi_change'] = df['rsi'].diff()
    df['macd_hist_change'] = df['macd_diff'].diff()
    df['dist_ema20'] = (df['close'] - df['ema20']) / df['ema20']
    df['dist_ema50'] = (df['close'] - df['ema50']) / df['ema50']
    df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['bull_engulf'] = 0
    df['bear_engulf'] = 0
    df['hammer'] = 0
    df['shooting'] = 0
    for i in range(2, len(df)):
        df.iloc[i, df.columns.get_loc('bull_engulf')] = bull_engulf(df, i)
        df.iloc[i, df.columns.get_loc('bear_engulf')] = bear_engulf(df, i)
        df.iloc[i, df.columns.get_loc('hammer')] = hammer_p(df, i)
        df.iloc[i, df.columns.get_loc('shooting')] = shooting_p(df, i)
    return df

def build_htf_features(df_htf):
    df = df_htf.copy()
    df['htf_ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['htf_trend'] = (df['close'] > df['htf_ema50']).astype(int)
    df['htf_rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['htf_adx'] = adx.adx()
    df['htf_ema_dist'] = (df['close'] - df['htf_ema50']) / df['htf_ema50']
    df['close_time'] = df.index + pd.Timedelta(hours=1)
    cols = ['htf_trend', 'htf_rsi', 'htf_adx', 'htf_ema_dist', 'close_time']
    return df[cols].set_index('close_time')

def merge_htf_ltf(df_ltf, df_htf_features):
    left = df_ltf.reset_index()
    if 'time' not in left.columns:
        left['time'] = df_ltf.index
    right = df_htf_features.reset_index().rename(columns={'close_time': 'time'})
    left = left.sort_values('time')
    right = right.sort_values('time')
    merged = pd.merge_asof(left, right, on='time', direction='backward', allow_exact_matches=True)
    return merged.set_index('time')

FEATURE_COLS = [
    'rsi', 'bb_pos', 'macd', 'macd_signal', 'macd_diff',
    'adx', 'di_plus', 'di_minus',
    'dist_ema20', 'dist_ema50', 'dist_ema200',
    'candle_body', 'upper_wick', 'lower_wick',
    'price_change', 'volatility', 'rsi_change', 'macd_hist_change',
    'bull_engulf', 'bear_engulf', 'hammer', 'shooting',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'htf_trend', 'htf_rsi', 'htf_adx', 'htf_ema_dist',
]

def create_dataset(df, expiry):
    X, y, meta = [], [], []
    for i in range(250, len(df) - expiry):
        row = df.iloc[i]
        if row[FEATURE_COLS].isna().any():
            continue
        entry = row['close']
        exit_p = df['close'].iloc[i + expiry]
        if exit_p == entry:
            continue
        label = 1 if exit_p > entry else 0
        X.append(row[FEATURE_COLS].values.astype(float))
        y.append(label)
        meta.append({'time': df.index[i], 'entry': entry})
    return np.array(X), np.array(y), meta

def purged_walk_forward(X, y, meta, expiry, n_folds):
    n = len(X)
    embargo = expiry
    remaining = n - int(n * MIN_TRAIN_RATIO)
    fold_size = remaining // n_folds
    oos_probs, oos_y, oos_meta = [], [], []
    fold_info = []
    for fold in range(n_folds):
        train_end = int(n * MIN_TRAIN_RATIO) + fold * fold_size
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start or train_end < 200:
            continue
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        m_te = meta[test_start:test_end]
        if len(X_te) < 20:
            continue
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=20, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        acc = accuracy_score(y_te, model.predict(X_te))
        fold_info.append(round(acc * 100, 2))
        oos_probs.extend(probs)
        oos_y.extend(y_te)
        oos_meta.extend(m_te)
    return np.array(oos_probs), np.array(oos_y), oos_meta, fold_info

# ================== Backtest ==================
def backtest_symbol(name, symbol):
    try:
        print(f"⏳ {name}...")
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        ts = td.time_series(symbol=symbol, interval=INTERVAL, outputsize=OUTPUTSIZE, timezone="UTC")
        df = ts.as_pandas()
        time.sleep(7)
        ts_h = td.time_series(symbol=symbol, interval=HTF_INTERVAL, outputsize=HTF_OUTPUTSIZE, timezone="UTC")
        df_h = ts_h.as_pandas()
        time.sleep(7)
        if df is None or df.empty or len(df) < 500:
            return {"symbol": name, "error": "داده کم"}
        df = df.rename(columns=str.lower).sort_index()
        df_h = df_h.rename(columns=str.lower).sort_index()
        htf_feat = build_htf_features(df_h)
        df = build_ltf_features(df)
        df = merge_htf_ltf(df, htf_feat)
        df = df.dropna()
        if len(df) < 500:
            return {"symbol": name, "error": "داده کم بعد فیلتر"}
        
        X, y, meta = create_dataset(df, EXPIRY_CANDLES)
        if len(X) < 400:
            return {"symbol": name, "error": "نمونه کم"}
        
        oos_probs, oos_y, oos_meta, fold_info = purged_walk_forward(X, y, meta, EXPIRY_CANDLES, N_FOLDS)
        if len(oos_probs) < 50:
            return {"symbol": name, "error": "OOS کم"}
        
        # محاسبه با threshold
        bankroll = INITIAL_BANKROLL
        signals, wins = 0, 0
        for j, prob in enumerate(oos_probs):
            mp = max(prob)
            if mp * 100 < THRESHOLD:
                continue
            pred = 1 if prob[1] > prob[0] else 0
            is_win = int(pred == oos_y[j])
            signals += 1
            wins += is_win
            stake = bankroll * STAKE_PCT
            if is_win:
                bankroll += stake * PAYOUT
            else:
                bankroll -= stake
        
        wr = (wins / signals * 100) if signals > 0 else 0
        ret_pct = (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
        
        # تعداد روز
        if len(oos_meta) >= 2:
            days = (oos_meta[-1]['time'] - oos_meta[0]['time']).days
            days = max(days, 1)
        else:
            days = 1
        sig_per_day = round(signals / days, 1)
        
        # آموزش مدل نهایی روی همه داده برای live
        final_model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=20, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        final_model.fit(X, y)
        trained_models[name] = final_model
        
        return {
            "symbol": name,
            "signals": signals,
            "wins": wins,
            "wr": round(wr, 2),
            "signals_per_day": sig_per_day,
            "return_pct": round(ret_pct, 2),
            "final_bankroll": round(bankroll, 2),
            "folds": fold_info,
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"symbol": name, "error": str(e)}

def run_backtest_background():
    global backtest_running
    if backtest_running:
        send_telegram("⚠️ بک‌تست قبلی در جریان است")
        return
    backtest_running = True
    try:
        send_telegram(
            "🔬 <b>بک‌تست نهایی</b>\n\n"
            "🎯 ۴ جفت‌ارز منتخب\n"
            "⏱ اکسپایر: ۳۰ دقیقه\n"
            "🎯 آستانه: ۷۰٪\n"
            "💰 بانک اولیه: ۱۰۰۰$ | ریسک: ۱٪\n\n"
            "⏳ در حال اجرا..."
        )
        
        results = []
        for name, sym in SYMBOLS.items():
            r = backtest_symbol(name, sym)
            if "error" in r:
                send_telegram(f"❌ <b>{name}</b>: {r['error']}")
                continue
            results.append(r)
            
            # گزارش ساده برای هر جفت
            msg = (
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📌 <b>{r['symbol']}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"🎯 <b>وین ریت:</b> {r['wr']}%\n"
                f"📈 <b>سیگنال:</b> {r['signals']} (روزی ~{r['signals_per_day']})\n"
                f"💰 <b>بانک:</b> 1000$ → {r['final_bankroll']}$\n"
                f"📊 <b>سود:</b> {r['return_pct']}%\n"
                f"📉 <b>دقت فولدها:</b> {r['folds']}"
            )
            send_telegram(msg)
        
        # خلاصه نهایی
        if results:
            total_signals = sum(r['signals'] for r in results)
            total_wins = sum(r['wins'] for r in results)
            avg_wr = round(total_wins / total_signals * 100, 2) if total_signals > 0 else 0
            total_ret = sum(r['return_pct'] for r in results)
            avg_ret = round(total_ret / len(results), 2)
            
            final = (
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🏆 <b>خلاصه کل</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 وین ریت میانگین: <b>{avg_wr}%</b>\n"
                f"📈 مجموع سیگنال: {total_signals}\n"
                f"💰 میانگین بازدهی: <b>+{avg_ret}%</b>\n\n"
                f"✅ آماده برای تست لایو"
            )
            send_telegram(final)
    finally:
        backtest_running = False

# ================== Live Signal ==================
def analyze_live(symbol):
    """تحلیل لحظه‌ای برای یک نماد"""
    try:
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        ts = td.time_series(symbol=symbol, interval=INTERVAL, outputsize=500, timezone="UTC")
        df = ts.as_pandas()
        if df is None or df.empty or len(df) < 250:
            return None
        
        ts_h = td.time_series(symbol=symbol, interval=HTF_INTERVAL, outputsize=200, timezone="UTC")
        df_h = ts_h.as_pandas()
        
        df = df.rename(columns=str.lower).sort_index()
        df_h = df_h.rename(columns=str.lower).sort_index()
        
        htf_feat = build_htf_features(df_h)
        df = build_ltf_features(df)
        df = merge_htf_ltf(df, htf_feat)
        df = df.dropna()
        
        if len(df) < 250:
            return None
        
        # آخرین کندل بسته شده
        last_row = df.iloc[-1]
        if last_row[FEATURE_COLS].isna().any():
            return None
        
        model = trained_models.get(symbol)
        if model is None:
            return None
        
        X = last_row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        prob = model.predict_proba(X)[0]
        confidence = max(prob) * 100
        
        if confidence < THRESHOLD:
            return None
        
        direction = "CALL" if prob[1] > prob[0] else "PUT"
        entry_price = float(last_row['close'])
        entry_time = df.index[-1]
        expiry_time = entry_time + timedelta(minutes=30)
        
        return {
            "symbol": symbol.replace("=X", ""),
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry_price": entry_price,
            "entry_time": entry_time,
            "expiry_time": expiry_time,
        }
    except Exception as e:
        print(f"analyze error for {symbol}: {e}")
        return None

def live_loop():
    """حلقه اصلی تحلیل زنده - هر ۱۵ دقیقه، ۵ ثانیه بعد از بسته شدن کندل"""
    global live_running
    last_signal_time = {}
    
    while live_running:
        try:
            now_utc = datetime.now(timezone.utc)
            minute = now_utc.minute
            second = now_utc.second
            
            # بررسی: آیا دقیقه بر ۱۵ بخش‌پذیر است و ۵ ثانیه گذشته؟
            if minute % 15 == 0 and 5 <= second <= 20:
                # جلوگیری از اجرای چندباره
                slot_key = now_utc.strftime("%Y%m%d%H%M")
                if slot_key == last_signal_time.get("_slot"):
                    time.sleep(5)
                    continue
                last_signal_time["_slot"] = slot_key
                
                print(f"🔍 بررسی سیگنال‌ها در {now_utc}")
                
                for name, sym in SYMBOLS.items():
                    try:
                        result = analyze_live(sym)
                        if result is None:
                            continue
                        
                        # جلوگیری از ارسال تکراری
                        sig_key = f"{result['symbol']}_{result['entry_time'].strftime('%Y%m%d%H%M')}"
                        if sig_key == last_signal_time.get(result['symbol']):
                            continue
                        last_signal_time[result['symbol']] = sig_key
                        
                        # ارسال به تلگرام
                        emoji = "🟢" if result['direction'] == "CALL" else "🔴"
                        msg = (
                            f"{emoji} <b>سیگنال {result['direction']}</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 نماد: <b>{result['symbol']}</b>\n"
                            f"🕐 ورود: <b>{fmt_iran(result['entry_time'])}</b> (به وقت ایران)\n"
                            f"⏱ اکسپایر: <b>۳۰ دقیقه</b>\n"
                            f"🔔 پایان: <b>{fmt_iran(result['expiry_time'])}</b>\n"
                            f"💵 قیمت: {result['entry_price']:.5f}\n"
                            f"🎯 اطمینان: <b>{result['confidence']}%</b>"
                        )
                        send_telegram(msg)
                        print(f"✅ سیگنال ارسال شد: {result['symbol']} {result['direction']}")
                    except Exception as e:
                        print(f"error in analyze {name}: {e}")
                    time.sleep(2)  # بین جفت‌ارزها
            
            time.sleep(5)
        except Exception as e:
            print(f"live_loop error: {e}")
            time.sleep(10)

@app.route('/')
def health():
    return "Signal Server is running!"

@app.route('/backtest', methods=['GET'])
def backtest_route():
    threading.Thread(target=run_backtest_background, daemon=True).start()
    return jsonify({"status": "ok", "message": "بک‌تست شروع شد"}), 200

@app.route('/start_live', methods=['GET'])
def start_live_route():
    global live_running
    if live_running:
        return jsonify({"status": "already running"}), 200
    if not trained_models:
        return jsonify({"status": "error", "message": "اول بک‌تست بزن"}), 400
    live_running = True
    threading.Thread(target=live_loop, daemon=True).start()
    send_telegram("🟢 حالت سیگنال زنده فعال شد. از این به بعد سیگنال‌ها ارسال می‌شن.")
    return jsonify({"status": "started"}), 200

@app.route('/stop_live', methods=['GET'])
def stop_live_route():
    global live_running
    live_running = False
    send_telegram("🔴 حالت سیگنال زنده متوقف شد.")
    return jsonify({"status": "stopped"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
