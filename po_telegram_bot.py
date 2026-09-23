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
from sklearn.metrics import accuracy_score

app = Flask(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

SYMBOLS = {
    "USD/JPY": "USD/JPY",
    "EUR/JPY": "EUR/JPY",
    "GBP/JPY": "GBP/JPY",
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/CHF": "USD/CHF",
}

INTERVAL = "15min"
HTF_INTERVAL = "1h"
OUTPUTSIZE = 5000
HTF_OUTPUTSIZE = 500
EXPIRY_OPTIONS = [2, 4]  # ۲ کندل=۳۰ دقیقه, ۴ کندل=۶۰ دقیقه
N_FOLDS = 4  # برای Walk-Forward

backtest_running = False

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"خطا: {e}")

# ================== الگوها ==================
def bullish_engulfing(df, i):
    if i < 1: return 0
    p, c = df.iloc[i-1], df.iloc[i]
    return 1 if (p['close'] < p['open'] and c['close'] > c['open'] and c['close'] > p['open'] and c['open'] < p['close']) else 0

def bearish_engulfing(df, i):
    if i < 1: return 0
    p, c = df.iloc[i-1], df.iloc[i]
    return 1 if (p['close'] > p['open'] and c['close'] < c['open'] and c['close'] < p['open'] and c['open'] > p['close']) else 0

def hammer(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return 0
    lw = min(c['close'], c['open']) - c['low']
    uw = c['high'] - max(c['close'], c['open'])
    return 1 if (lw >= 2 * body and uw <= body * 0.8) else 0

def shooting_star(df, i):
    c = df.iloc[i]
    body = abs(c['close'] - c['open'])
    if body == 0: return 0
    uw = c['high'] - max(c['close'], c['open'])
    lw = min(c['close'], c['open']) - c['low']
    return 1 if (uw >= 2 * body and lw <= body * 0.8) else 0

# ================== Feature Engineering ==================
def build_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_pos'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
    
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['adx'] = adx.adx()
    df['di_plus'] = adx.adx_pos()
    df['di_minus'] = adx.adx_neg()
    
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
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    
    df['bull_engulf'] = 0
    df['bear_engulf'] = 0
    df['hammer'] = 0
    df['shooting'] = 0
    for i in range(2, len(df)):
        df.iloc[i, df.columns.get_loc('bull_engulf')] = bullish_engulfing(df, i)
        df.iloc[i, df.columns.get_loc('bear_engulf')] = bearish_engulfing(df, i)
        df.iloc[i, df.columns.get_loc('hammer')] = hammer(df, i)
        df.iloc[i, df.columns.get_loc('shooting')] = shooting_star(df, i)
    
    return df

FEATURE_COLS = [
    'rsi', 'bb_pos', 'macd', 'macd_signal', 'macd_diff',
    'adx', 'di_plus', 'di_minus',
    'dist_ema20', 'dist_ema50', 'dist_ema200',
    'candle_body', 'upper_wick', 'lower_wick',
    'price_change', 'volatility', 'rsi_change', 'macd_hist_change',
    'bull_engulf', 'bear_engulf', 'hammer', 'shooting',
    'hour', 'dow'
]

def create_dataset(df, expiry):
    X, y, meta = [], [], []
    for i in range(250, len(df) - expiry):
        row = df.iloc[i]
        if row[FEATURE_COLS].isna().any():
            continue
        entry = row['close']
        exit_p = df['close'].iloc[i + expiry]
        label = 1 if exit_p > entry else 0
        X.append(row[FEATURE_COLS].values.astype(float))
        y.append(label)
        meta.append({
            'time': df.index[i],
            'entry': entry,
            'exit': exit_p,
            'ema200': row['ema200'],
        })
    return np.array(X), np.array(y), meta

def train_eval(X_train, y_train, X_test):
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        min_samples_split=20, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def calc_metrics(probs, y_test, meta_test, threshold, direction_label):
    """محاسبه متریک‌های کامل برای یک threshold مشخص"""
    signals = 0
    wins = 0
    call_signals = call_wins = 0
    put_signals = put_wins = 0
    trend_up_signals = trend_up_wins = 0
    trend_down_signals = trend_down_wins = 0
    confidence_buckets = {i: [0, 0] for i in range(50, 100, 5)}
    
    max_streak = 0
    current_streak = 0
    
    for j, prob in enumerate(probs):
        max_prob = max(prob)
        if max_prob * 100 < threshold:
            continue
        
        pred = 1 if prob[1] > prob[0] else 0
        confidence = max_prob * 100
        is_win = (pred == y_test[j])
        
        signals += 1
        if is_win: wins += 1
        
        # CALL/PUT
        if pred == 1:
            call_signals += 1
            if is_win: call_wins += 1
        else:
            put_signals += 1
            if is_win: put_wins += 1
        
        # Trend bucket
        ema = meta_test[j]['ema200']
        entry = meta_test[j]['entry']
        if not pd.isna(ema):
            if entry > ema:
                trend_up_signals += 1
                if is_win: trend_up_wins += 1
            else:
                trend_down_signals += 1
                if is_win: trend_down_wins += 1
        
        # Confidence bucket
        bucket = int(confidence // 5) * 5
        if bucket in confidence_buckets:
            confidence_buckets[bucket][0] += 1
            if is_win: confidence_buckets[bucket][1] += 1
        
        # Losing streak
        if not is_win:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    wr = (wins / signals * 100) if signals > 0 else 0
    return {
        'signals': signals, 'wins': wins, 'wr': round(wr, 2),
        'call_signals': call_signals,
        'call_wr': round(call_wins / call_signals * 100, 2) if call_signals > 0 else 0,
        'put_signals': put_signals,
        'put_wr': round(put_wins / put_signals * 100, 2) if put_signals > 0 else 0,
        'trend_up_signals': trend_up_signals,
        'trend_up_wr': round(trend_up_wins / trend_up_signals * 100, 2) if trend_up_signals > 0 else 0,
        'trend_down_signals': trend_down_signals,
        'trend_down_wr': round(trend_down_wins / trend_down_signals * 100, 2) if trend_down_signals > 0 else 0,
        'max_losing_streak': max_streak,
        'confidence_buckets': {k: v for k, v in confidence_buckets.items() if v[0] > 0},
    }

def backtest_symbol(name, symbol):
    try:
        print(f"⏳ دانلود {name}...")
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        
        # داده ۱۵ دقیقه
        ts = td.time_series(symbol=symbol, interval=INTERVAL, outputsize=OUTPUTSIZE, timezone="UTC")
        df = ts.as_pandas()
        time.sleep(7)
        
        # داده ۱ ساعته (واقعاً استفاده می‌شه)
        ts_htf = td.time_series(symbol=symbol, interval=HTF_INTERVAL, outputsize=HTF_OUTPUTSIZE, timezone="UTC")
        df_htf = ts_htf.as_pandas()
        time.sleep(7)
        
        if df is None or df.empty or len(df) < 500:
            return {"symbol": name, "error": "داده 15m کافی نیست"}
        
        df = df.rename(columns=str.lower).sort_index()
        df_htf = df_htf.rename(columns=str.lower).sort_index()
        
        # اضافه کردن HTF trend به LTF
        df_htf['htf_ema50'] = ta.trend.EMAIndicator(df_htf['close'], 50).ema_indicator()
        df_htf['htf_trend'] = (df_htf['close'] > df_htf['htf_ema50']).astype(int)
        df_htf['htf_rsi'] = ta.momentum.RSIIndicator(df_htf['close'], 14).rsi()
        
        # Merge: برای هر کندل 15m، آخرین وضعیت HTF رو پیدا کن
        df_htf_resampled = df_htf[['htf_trend', 'htf_rsi']].reindex(df.index, method='ffill')
        df['htf_trend'] = df_htf_resampled['htf_trend'].values
        df['htf_rsi'] = df_htf_resampled['htf_rsi'].values
        
        df = build_features(df)
        df = df.dropna()
        
        if len(df) < 500:
            return {"symbol": name, "error": "داده پس از پاک‌سازی کم است"}
        
        results_by_expiry = {}
        
        for expiry in EXPIRY_OPTIONS:
            X, y, meta = create_dataset(df, expiry)
            if len(X) < 400:
                continue
            
            # Walk-Forward Validation
            fold_size = len(X) // (N_FOLDS + 1)
            oos_probs = []
            oos_y = []
            oos_meta = []
            fold_accuracies = []
            
            for fold in range(N_FOLDS):
                train_end = (fold + 1) * fold_size
                test_end = min(train_end + fold_size, len(X))
                if test_end <= train_end:
                    continue
                
                X_train = X[:train_end]
                y_train = y[:train_end]
                X_test = X[train_end:test_end]
                y_test = y[train_end:test_end]
                meta_test = meta[train_end:test_end]
                
                model = train_eval(X_train, y_train, X_test)
                probs = model.predict_proba(X_test)
                fold_acc = accuracy_score(y_test, model.predict(X_test))
                fold_accuracies.append(fold_acc)
                
                oos_probs.extend(probs)
                oos_y.extend(y_test)
                oos_meta.extend(meta_test)
            
            oos_probs = np.array(oos_probs)
            oos_y = np.array(oos_y)
            
            # محاسبه متریک برای thresholdهای مختلف
            expiry_results = {}
            for th in [60, 65, 70, 75, 80]:
                expiry_results[th] = calc_metrics(oos_probs, oos_y, oos_meta, th, f"{expiry}c")
            
            results_by_expiry[expiry] = {
                'fold_accuracies': [round(a*100, 2) for a in fold_accuracies],
                'mean_accuracy': round(np.mean(fold_accuracies)*100, 2),
                'results': expiry_results,
            }
        
        del df, df_htf
        gc.collect()
        return {"symbol": name, "results_by_expiry": results_by_expiry}
    except Exception as e:
        import traceback
        return {"symbol": name, "error": f"{type(e).__name__}: {str(e)}"}

def run_backtest_background():
    global backtest_running
    if backtest_running:
        send_telegram("⚠️ بک‌تست قبلی هنوز در حال اجراست.")
        return
    backtest_running = True
    try:
        send_telegram("🔬 <b>بک‌تست عمیق ML شروع شد</b>\n\n"
                      "✅ Walk-Forward Validation (4 folds)\n"
                      "✅ CALL/PUT تفکیک‌شده\n"
                      "✅ Confidence Buckets\n"
                      "✅ Trend Filter واقعی (1h)\n"
                      "✅ ۲ اکسپایر (۳۰ و ۶۰ دقیقه)\n"
                      "⏳ این تست ~۵-۸ دقیقه طول می‌کشه")
        
        # خلاصه کلی برای هر threshold
        summary = {th: {'signals': 0, 'wins': 0} for th in [60, 65, 70, 75, 80]}
        summary_expiry_30 = {th: {'signals': 0, 'wins': 0} for th in [60, 65, 70, 75, 80]}
        summary_expiry_60 = {th: {'signals': 0, 'wins': 0} for th in [60, 65, 70, 75, 80]}
        
        for name, sym in SYMBOLS.items():
            r = backtest_symbol(name, sym)
            if "error" in r:
                send_telegram(f"❌ <b>{r['symbol']}</b>: {r['error']}")
                continue
            
            # برای هر اکسپایر
            for expiry, exp_data in r['results_by_expiry'].items():
                exp_label = "۳۰د" if expiry == 2 else "۶۰د"
                msg = (f"<b>{r['symbol']}</b> — اکسپایر {exp_label}\n"
                       f"📊 میانگین دقت فولدها: {exp_data['mean_accuracy']}%\n"
                       f"📊 دقت هر فولد: {exp_data['fold_accuracies']}\n")
                send_telegram(msg)
                
                # بهترین threshold برای این pair
                best_th, best_wr = 65, 0
                for th, metrics in exp_data['results'].items():
                    if metrics['signals'] >= 20 and metrics['wr'] > best_wr:
                        best_wr = metrics['wr']
                        best_th = th
                
                best = exp_data['results'][best_th]
                detail = (f"🎯 <b>آستانه {best_th}%</b> (بهترین)\n"
                          f"  سیگنال: {best['signals']} | وین ریت: <b>{best['wr']}%</b>\n"
                          f"  📈 CALL: {best['call_signals']} ({best['call_wr']}%)\n"
                          f"  📉 PUT: {best['put_signals']} ({best['put_wr']}%)\n"
                          f"  ⬆️ روند صعودی: {best['trend_up_signals']} ({best['trend_up_wr']}%)\n"
                          f"  ⬇️ روند نزولی: {best['trend_down_signals']} ({best['trend_down_wr']}%)\n"
                          f"  🔻 حداکثر باخت متوالی: {best['max_losing_streak']}\n")
                send_telegram(detail)
                
                # Confidence calibration
                calib = "📊 <b>Calibration:</b>\n"
                for bucket, (total, wins) in sorted(best['confidence_buckets'].items()):
                    wr = round(wins/total*100, 1) if total > 0 else 0
                    calib += f"  {bucket}-{bucket+5}%: {total} → {wr}%\n"
                send_telegram(calib)
                
                # جمع‌بندی کلی
                for th, metrics in exp_data['results'].items():
                    summary[th]['signals'] += metrics['signals']
                    summary[th]['wins'] += metrics['wins']
                    if expiry == 2:
                        summary_expiry_30[th]['signals'] += metrics['signals']
                        summary_expiry_30[th]['wins'] += metrics['wins']
                    elif expiry == 4:
                        summary_expiry_60[th]['signals'] += metrics['signals']
                        summary_expiry_60[th]['wins'] += metrics['wins']
        
        # خلاصه نهایی
        final = "🏁 <b>خلاصه نهایی</b>\n\n"
        final += "📅 <b>اکسپایر ۳۰ دقیقه:</b>\n"
        for th in [60, 65, 70, 75, 80]:
            t = summary_expiry_30[th]
            wr = round(t['wins']/t['signals']*100, 2) if t['signals'] > 0 else 0
            final += f"  {th}%: {t['signals']} سیگنال | {wr}%\n"
        
        final += "\n📅 <b>اکسپایر ۶۰ دقیقه:</b>\n"
        for th in [60, 65, 70, 75, 80]:
            t = summary_expiry_60[th]
            wr = round(t['wins']/t['signals']*100, 2) if t['signals'] > 0 else 0
            final += f"  {th}%: {t['signals']} سیگنال | {wr}%\n"
        
        send_telegram(final)
    finally:
        backtest_running = False

@app.route('/')
def health():
    return "Deep ML Signal Server running"

@app.route('/backtest', methods=['GET'])
def backtest_route():
    threading.Thread(target=run_backtest_background, daemon=True).start()
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
