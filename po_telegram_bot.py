import os
import gc
import time
import threading
import requests
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
HTF_OUTPUTSIZE = 800
EXPIRY_OPTIONS = [2, 3, 4]
N_FOLDS = 4
MIN_TRAIN_RATIO = 0.4

THRESHOLDS_TO_REPORT = [60, 65, 70, 75, 80]
PAYOUTS_TO_TEST = [0.75, 0.80, 0.85, 0.90]

INITIAL_BANKROLL = 1000.0
STAKE_PCT = 0.01  # 1% ریسک در هر معامله

backtest_running = False

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=15)
    except Exception as e:
        print(f"telegram error: {e}")

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

# ================== Features روی 15m ==================
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
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    # Patterns
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

# ================== Features روی 1H با timestamp اصلاح‌شده ==================
def build_htf_features(df_htf):
    """
    برای هر کندل 1H، ویژگی‌ها را محاسبه می‌کند.
    IMPORTANT: timestamp کندل 1H را به «زمان بسته شدن» تغییر می‌دهیم.
    اگر timestamp در TwelveData = شروع کندل باشد،
    بسته شدن = شروع + 1h.
    سپس در merge_asof فقط از کندل‌های بسته‌شده استفاده می‌کنیم.
    """
    df = df_htf.copy()
    df['htf_ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['htf_trend'] = (df['close'] > df['htf_ema50']).astype(int)
    df['htf_rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['htf_adx'] = adx.adx()
    df['htf_ema_dist'] = (df['close'] - df['htf_ema50']) / df['htf_ema50']
    
    # 🔴 حیاتی: timestamp را به «زمان بسته شدن» تغییر بده
    # فرض: timestamp TwelveData = شروع کندل 1H
    df['close_time'] = df.index + pd.Timedelta(hours=1)
    
    cols = ['htf_trend', 'htf_rsi', 'htf_adx', 'htf_ema_dist', 'close_time']
    out = df[cols].copy()
    out = out.set_index('close_time')
    return out

# ================== ادغام HTF با LTF ==================
def merge_htf_ltf(df_ltf, df_htf_features):
    """
    برای هر کندل 15m، آخرین کندل 1H که «بسته شده» را attach می‌کند.
    merge_asof با direction='backward' فقط از کندل‌های با close_time <= current_time استفاده می‌کند.
    """
    left = df_ltf.reset_index().rename(columns={'index': 'time'})
    # اگر index نام دارد (datetime)، این خط را امن می‌کند
    if 'time' not in left.columns:
        left['time'] = df_ltf.index
    
    right = df_htf_features.reset_index().rename(columns={'close_time': 'time'})
    
    left = left.sort_values('time')
    right = right.sort_values('time')
    
    merged = pd.merge_asof(
        left, right,
        on='time',
        direction='backward',  # فقط کندل‌های بسته‌شده قبلی
        allow_exact_matches=True
    )
    merged = merged.set_index('time')
    return merged

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

# ================== Dataset با حذف Tie ==================
def create_dataset(df, expiry):
    X, y, meta = [], [], []
    for i in range(250, len(df) - expiry):
        row = df.iloc[i]
        if row[FEATURE_COLS].isna().any():
            continue
        entry = row['close']
        exit_p = df['close'].iloc[i + expiry]
        
        # 🔴 Tie handling: حذف اگر exit == entry
        if exit_p == entry:
            continue
        label = 1 if exit_p > entry else 0
        
        X.append(row[FEATURE_COLS].values.astype(float))
        y.append(label)
        meta.append({
            'time': df.index[i],
            'entry': entry,
            'exit': exit_p,
            'htf_trend': int(row['htf_trend']) if not pd.isna(row['htf_trend']) else 0,
        })
    return np.array(X), np.array(y), meta

# ================== Baselines ==================
def compute_baselines(df, expiry):
    """سه baseline ساده برای مقایسه"""
    n = len(df) - expiry
    
    # 1. Always CALL
    always_call_correct = 0
    total = 0
    for i in range(250, n):
        entry = df['close'].iloc[i]
        exit_p = df['close'].iloc[i + expiry]
        if exit_p == entry:
            continue
        total += 1
        if exit_p > entry:
            always_call_correct += 1
    
    # 2. Previous candle direction
    prev_correct = 0
    prev_total = 0
    for i in range(251, n):
        prev_close = df['close'].iloc[i-1]
        prev_open = df['open'].iloc[i-1]
        entry = df['close'].iloc[i]
        exit_p = df['close'].iloc[i + expiry]
        if exit_p == entry:
            continue
        prev_total += 1
        pred_up = prev_close > prev_open
        actual_up = exit_p > entry
        if pred_up == actual_up:
            prev_correct += 1
    
    # 3. EMA200 trend
    ema_correct = 0
    ema_total = 0
    for i in range(250, n):
        row = df.iloc[i]
        if pd.isna(row['ema200']):
            continue
        entry = row['close']
        exit_p = df['close'].iloc[i + expiry]
        if exit_p == entry:
            continue
        ema_total += 1
        pred_up = entry > row['ema200']
        actual_up = exit_p > entry
        if pred_up == actual_up:
            ema_correct += 1
    
    return {
        'always_call': round(always_call_correct / total * 100, 2) if total > 0 else 0,
        'prev_candle': round(prev_correct / prev_total * 100, 2) if prev_total > 0 else 0,
        'ema200_trend': round(ema_correct / ema_total * 100, 2) if ema_total > 0 else 0,
    }

# ================== محاسبه متریک‌ها ==================
def calc_metrics(probs, y_test, meta_test, threshold, payout, one_at_a_time=False, expiry=3):
    """
    محاسبه متریک‌ها با:
    - بانک‌رول واقعی
    - حالت one_at_a_time (فقط یک معامله باز در لحظه)
    """
    bankroll = INITIAL_BANKROLL
    equity_curve = [bankroll]
    
    signals = 0
    wins = 0
    call_sig = call_wins = 0
    put_sig = put_wins = 0
    up_sig = up_wins = 0
    dn_sig = dn_wins = 0
    conf_buckets = {b: [0, 0] for b in range(50, 100, 5)}
    conf_sum = 0.0
    max_streak = 0
    cur_streak = 0
    last_trade_end_time = None
    
    for j, prob in enumerate(probs):
        mp = max(prob)
        if mp * 100 < threshold:
            continue
        
        pred = 1 if prob[1] > prob[0] else 0
        conf = mp * 100
        is_win = int(pred == y_test[j])
        
        # one-at-a-time: اگر معامله قبلی هنوز بسته نشده، رد کن
        if one_at_a_time:
            current_time = meta_test[j]['time']
            if last_trade_end_time is not None:
                if current_time < last_trade_end_time:
                    continue
        
        signals += 1
        wins += is_win
        conf_sum += conf
        
        # Bankroll update
        stake = bankroll * STAKE_PCT
        if is_win:
            bankroll += stake * payout
        else:
            bankroll -= stake
        equity_curve.append(bankroll)
        
        if one_at_a_time:
            last_trade_end_time = meta_test[j]['time'] + pd.Timedelta(minutes=15 * expiry)
        
        if pred == 1:
            call_sig += 1; call_wins += is_win
        else:
            put_sig += 1; put_wins += is_win
        
        htf = meta_test[j].get('htf_trend', -1)
        if htf == 1:
            up_sig += 1; up_wins += is_win
        elif htf == 0:
            dn_sig += 1; dn_wins += is_win
        
        b = int(conf // 5) * 5
        if b in conf_buckets:
            conf_buckets[b][0] += 1
            conf_buckets[b][1] += is_win
        
        if not is_win:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    
    # Max Drawdown روی equity واقعی
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak: peak = e
        if peak > 0:
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
    
    wr = (wins / signals * 100) if signals > 0 else 0
    avg_conf = (conf_sum / signals) if signals > 0 else 0
    # Expectancy per trade on unit stake
    exp_val = (wr/100 * payout) - ((100-wr)/100)
    
    return {
        'signals': signals, 'wins': wins, 'wr': round(wr, 2),
        'call_sig': call_sig,
        'call_wr': round(call_wins / call_sig * 100, 2) if call_sig > 0 else 0,
        'put_sig': put_sig,
        'put_wr': round(put_wins / put_sig * 100, 2) if put_sig > 0 else 0,
        'up_sig': up_sig,
        'up_wr': round(up_wins / up_sig * 100, 2) if up_sig > 0 else 0,
        'dn_sig': dn_sig,
        'dn_wr': round(dn_wins / dn_sig * 100, 2) if dn_sig > 0 else 0,
        'max_streak': max_streak,
        'max_dd': round(max_dd * 100, 2),
        'final_bankroll': round(bankroll, 2),
        'return_pct': round((bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100, 2),
        'expectancy': round(exp_val, 4),
        'avg_conf': round(avg_conf, 2),
        'conf_buckets': {k: v for k, v in conf_buckets.items() if v[0] > 0},
    }

# ================== Purged Walk-Forward ==================
def purged_walk_forward(X, y, meta, expiry, n_folds):
    n = len(X)
    embargo = expiry
    remaining = n - int(n * MIN_TRAIN_RATIO)
    fold_size = remaining // n_folds
    
    oos_probs, oos_y, oos_meta = [], [], []
    fold_info = []
    feature_importances = []
    
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
            n_estimators=100, max_depth=10,
            min_samples_split=20, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        acc = accuracy_score(y_te, model.predict(X_te))
        
        fold_info.append({
            'fold': fold + 1,
            'train_size': len(X_tr),
            'test_size': len(X_te),
            'accuracy': round(acc * 100, 2),
        })
        feature_importances.append(model.feature_importances_)
        
        oos_probs.extend(probs)
        oos_y.extend(y_te)
        oos_meta.extend(m_te)
    
    avg_importance = None
    if feature_importances:
        avg_importance = np.mean(feature_importances, axis=0)
    
    return np.array(oos_probs), np.array(oos_y), oos_meta, fold_info, avg_importance

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
            return {"symbol": name, "error": "داده 15m کم"}
        
        df = df.rename(columns=str.lower).sort_index()
        df_h = df_h.rename(columns=str.lower).sort_index()
        
        # HTF features با timestamp بسته شدن
        htf_feat = build_htf_features(df_h)
        
        # LTF features
        df = build_ltf_features(df)
        
        # ادغام با merge_asof
        df = merge_htf_ltf(df, htf_feat)
        df = df.dropna()
        
        if len(df) < 500:
            return {"symbol": name, "error": "داده پس از پاک‌سازی کم"}
        
        # Baselines
        baselines = compute_baselines(df, expiry=3)
        
        results_by_expiry = {}
        for expiry in EXPIRY_OPTIONS:
            X, y, meta = create_dataset(df, expiry)
            if len(X) < 400:
                continue
            
            oos_probs, oos_y, oos_meta, fold_info, importance = purged_walk_forward(X, y, meta, expiry, N_FOLDS)
            if len(oos_probs) < 50:
                continue
            
            exp_results = {}
            for th in THRESHOLDS_TO_REPORT:
                # حالت all signals
                all_sigs = calc_metrics(oos_probs, oos_y, oos_meta, th, 0.85, one_at_a_time=False, expiry=expiry)
                # حالت one-at-a-time
                one_sig = calc_metrics(oos_probs, oos_y, oos_meta, th, 0.85, one_at_a_time=True, expiry=expiry)
                
                # payout variations (فقط برای حالت all)
                payout_results = {}
                for p in PAYOUTS_TO_TEST:
                    m = calc_metrics(oos_probs, oos_y, oos_meta, th, p, one_at_a_time=False, expiry=expiry)
                    payout_results[p] = {
                        'final_bankroll': m['final_bankroll'],
                        'return_pct': m['return_pct'],
                        'expectancy': m['expectancy'],
                    }
                
                exp_results[th] = {
                    'all': all_sigs,
                    'one': one_sig,
                    'payouts': payout_results,
                }
            
            # Top features
            top_feats = []
            if importance is not None:
                feat_imp = list(zip(FEATURE_COLS, importance))
                feat_imp.sort(key=lambda x: -x[1])
                top_feats = [(f, round(float(v)*100, 2)) for f, v in feat_imp[:10]]
            
            results_by_expiry[expiry] = {
                'fold_info': fold_info,
                'n_oos': len(oos_probs),
                'results': exp_results,
                'top_features': top_feats,
            }
        
        del df, df_h
        gc.collect()
        return {"symbol": name, "results_by_expiry": results_by_expiry, "baselines": baselines}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"symbol": name, "error": f"{type(e).__name__}: {str(e)}"}

def run_backtest_background():
    global backtest_running
    if backtest_running:
        send_telegram("⚠️ بک‌تست قبلی در جریان است")
        return
    backtest_running = True
    try:
        send_telegram(
            "🔬 <b>بک‌تست نسخه ۴ (اصلاح کامل)</b>\n\n"
            "✅ Tie handling\n"
            "✅ HTF merge_asof بدون leakage\n"
            "✅ Bankroll واقعی (1000$, 1% ریسک)\n"
            "✅ Baselines: always CALL / prev candle / EMA200\n"
            "✅ Feature Importance\n"
            "✅ Payout 75/80/85/90\n"
            "✅ One-at-a-time simulation\n"
            "✅ Fold-by-fold metrics\n"
            "⏳ ۸-۱۲ دقیقه"
        )
        
        # جمع‌بندی
        summary = {exp: {th: {'signals': 0, 'wins': 0, 'ret': 0} for th in THRESHOLDS_TO_REPORT} for exp in EXPIRY_OPTIONS}
        
        for name, sym in SYMBOLS.items():
            r = backtest_symbol(name, sym)
            if "error" in r:
                send_telegram(f"❌ <b>{name}</b>: {r['error']}")
                continue
            
            # Baselines
            b = r['baselines']
            send_telegram(
                f"<b>{name}</b> — Baselines\n"
                f"  Always CALL: {b['always_call']}%\n"
                f"  Prev candle: {b['prev_candle']}%\n"
                f"  EMA200 trend: {b['ema200_trend']}%"
            )
            
            for expiry, ed in r['results_by_expiry'].items():
                label = {2: "30د", 3: "45د", 4: "60د"}.get(expiry, f"{expiry*15}د")
                
                # Fold-by-fold
                fold_str = " | ".join([f"F{f['fold']}:{f['accuracy']}%" for f in ed['fold_info']])
                send_telegram(f"<b>{name}</b> — {label}\n📊 فولدها: {fold_str}")
                
                # Top features
                if ed['top_features']:
                    tf = "\n".join([f"  {f}: {v}%" for f, v in ed['top_features'][:5]])
                    send_telegram(f"🔝 Top 5 features:\n{tf}")
                
                # Threshold 70 (canonical)
                for th in [65, 70, 75]:
                    if th not in ed['results']:
                        continue
                    m_all = ed['results'][th]['all']
                    m_one = ed['results'][th]['one']
                    if m_all['signals'] < 15:
                        continue
                    
                    msg = (
                        f"🎯 <b>ث {th}%</b>\n"
                        f"  All sigs: {m_all['signals']} | WR <b>{m_all['wr']}%</b> | "
                        f"Ret {m_all['return_pct']}% | DD {m_all['max_dd']}%\n"
                        f"  One-at-a-time: {m_one['signals']} | WR {m_one['wr']}% | "
                        f"Ret {m_one['return_pct']}% | DD {m_one['max_dd']}%\n"
                        f"  📈 CALL {m_all['call_sig']} ({m_all['call_wr']}%) | "
                        f"📉 PUT {m_all['put_sig']} ({m_all['put_wr']}%)\n"
                        f"  ⬆️ UP {m_all['up_sig']} ({m_all['up_wr']}%) | "
                        f"⬇️ DN {m_all['dn_sig']} ({m_all['dn_wr']}%)\n"
                        f"  🔻 استریک: {m_all['max_streak']} | "
                        f"💰 Exp: {m_all['expectancy']}"
                    )
                    send_telegram(msg)
                    
                    # Payout variations
                    pv = ed['results'][th]['payouts']
                    pv_str = "💵 <b>Payout variations:</b>\n"
                    for p, v in pv.items():
                        pv_str += f"  {int(p*100)}%: bankroll ${v['final_bankroll']} ({v['return_pct']}%)\n"
                    send_telegram(pv_str)
                    
                    # Reliability برای th=70
                    if th == 70:
                        cal = "📊 <b>Reliability:</b>\n"
                        for bk, (tt, w) in sorted(m_all['conf_buckets'].items()):
                            if tt > 0:
                                cal += f"  {bk}-{bk+5}%: {tt} → {round(w/tt*100, 1)}%\n"
                        send_telegram(cal)
                
                # جمع‌بندی
                for th, res in ed['results'].items():
                    summary[expiry][th]['signals'] += res['all']['signals']
                    summary[expiry][th]['wins'] += res['all']['wins']
                    summary[expiry][th]['ret'] += res['all']['return_pct']
        
        # خلاصه نهایی
        final = "🏁 <b>خلاصه نهایی</b>\n"
        for exp in EXPIRY_OPTIONS:
            label = {2: "30د", 3: "45د", 4: "60د"}.get(exp, f"{exp*15}د")
            final += f"\n📅 <b>{label}:</b>\n"
            for th in THRESHOLDS_TO_REPORT:
                s = summary[exp][th]
                wr = round(s['wins']/s['signals']*100, 2) if s['signals'] > 0 else 0
                final += f"  {th}%: {s['signals']} | WR {wr}% | ΣRet {round(s['ret'], 1)}%\n"
        send_telegram(final)
    finally:
        backtest_running = False

@app.route('/')
def health():
    return "ML v4 running"

@app.route('/backtest', methods=['GET'])
def backtest_route():
    threading.Thread(target=run_backtest_background, daemon=True).start()
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
