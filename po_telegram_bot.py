import os
import gc
import json
import time
import threading
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ================== SETTINGS ==================
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
EXPIRY_CANDLES = 2
THRESHOLD = 65
N_FOLDS = 4
MIN_TRAIN_RATIO = 0.4

# ================== OPTIMIZATION ==================
LIVE_LTF_SIZE = 450              # ✅ کافی برای EMA200 + buffer
LIVE_HTF_SIZE = 250              # ✅ کافی برای EMA50 + buffer
HTF_CACHE_DURATION = 3900        # 65 دقیقه
MAX_PARALLEL_WORKERS = 2         # ✅ کاهش به 2 (rate limit)
HTF_REQUEST_DELAY = 2.0          # ✅ فاصله بین درخواست های HTF

INITIAL_BANKROLL = 1000.0
PAYOUT = 0.85
STAKE_PCT = 0.01

MAX_SAME_DIRECTION_STREAK = 5

IRAN_TZ = timezone(timedelta(hours=3, minutes=30))

# ================== STATE ==================
backtest_running = False
live_running = False
trained_models = {}
auto_start_done = False

pending_signals = []
completed_signals = []
signal_id_counter = 0

last_analysis = {}
direction_streak = {}
daily_summary_sent = {}
last_update_id = 0

SIGNALS_FILE = "signals_data.json"

# ================== LOCKS ==================
htf_cache = {}
htf_cache_lock = threading.Lock()       # قفل خواندن کش
htf_fetch_lock = threading.Lock()       # ✅ قفل درخواست جدید HTF
signal_lock = threading.Lock()          # ✅ قفل pending_signals + counter
analysis_lock = threading.Lock()        # ✅ قفل last_analysis
save_lock = threading.Lock()            # ✅ قفل save_signals
streak_lock = threading.Lock()          # ✅ قفل direction_streak


# ================== HTF CACHE (با قفل کامل) ==================
def get_htf_data(symbol):
    """
    دریافت داده 1h از کش یا درخواست جدید
    ✅ با قفل کامل برای جلوگیری از rate limit
    """
    # چک کش
    with htf_cache_lock:
        now = time.time()
        if symbol in htf_cache:
            cache_time, cache_data = htf_cache[symbol]
            if now - cache_time < HTF_CACHE_DURATION:
                return cache_data

    # درخواست جدید — با قفل کامل
    with htf_fetch_lock:
        # چک دوباره (شاید thread دیگه درخواست داده)
        with htf_cache_lock:
            now = time.time()
            if symbol in htf_cache:
                cache_time, cache_data = htf_cache[symbol]
                if now - cache_time < HTF_CACHE_DURATION:
                    return cache_data

        try:
            print(f"[HTF] Fetching {symbol}...")
            td = TDClient(apikey=TWELVE_DATA_API_KEY)
            ts_h = td.time_series(
                symbol=symbol,
                interval=HTF_INTERVAL,
                outputsize=LIVE_HTF_SIZE,
                timezone="UTC"
            )
            df_h = ts_h.as_pandas()

            if df_h is not None and not df_h.empty:
                with htf_cache_lock:
                    htf_cache[symbol] = (time.time(), df_h)
                print(f"[HTF] Cached {symbol} ({len(df_h)} candles)")
                time.sleep(HTF_REQUEST_DELAY)
                return df_h
        except Exception as e:
            print(f"[HTF] Error {symbol}: {e}")
            time.sleep(HTF_REQUEST_DELAY)

    return None


def send_telegram(message, reply_markup=None):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"telegram error: {e}")


def send_telegram_async(message, reply_markup=None):
    """✅ ارسال در thread جداگانه (غیر بلاک)"""
    threading.Thread(
        target=send_telegram,
        args=(message, reply_markup),
        daemon=True
    ).start()


def get_main_keyboard():
    return {
        "keyboard": [
            [{"text": "آمار"}, {"text": "وضعیت"}],
            [{"text": "تحلیل"}, {"text": "امروز"}],
            [{"text": "خلاصه روز"}, {"text": "راهنما"}],
            [{"text": "توقف"}, {"text": "شروع"}],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
    }


def send_telegram_with_keyboard(message):
    send_telegram(message, reply_markup=get_main_keyboard())


def to_iran(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IRAN_TZ)


def fmt_iran(dt):
    return to_iran(dt).strftime("%H:%M")


# ================== SAVE / LOAD ==================
def _serialize_signal(s):
    s_copy = s.copy()
    for key in ['entry_time', 'expiry_time', 'saved_at']:
        if key in s_copy and hasattr(s_copy[key], 'isoformat'):
            s_copy[key] = s_copy[key].isoformat()
    return s_copy


def _deserialize_signal(s):
    for key in ['entry_time', 'expiry_time', 'saved_at']:
        if key in s and isinstance(s[key], str):
            try:
                s[key] = datetime.fromisoformat(s[key])
            except:
                pass
    return s


def save_signals():
    """✅ با قفل"""
    with save_lock:
        try:
            pending_serialized = [_serialize_signal(s) for s in pending_signals]
            completed_serialized = [_serialize_signal(s) for s in completed_signals]
            data = {
                "pending": pending_serialized,
                "completed": completed_serialized,
                "signal_counter": signal_id_counter,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"save_signals error: {e}")


def load_signals():
    global pending_signals, completed_signals, signal_id_counter
    try:
        if not os.path.exists(SIGNALS_FILE):
            return
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        pending_signals = [_deserialize_signal(s) for s in data.get("pending", [])]
        completed_signals = [_deserialize_signal(s) for s in data.get("completed", [])]
        signal_id_counter = data.get("signal_counter", 0)
        print(f"Signals loaded: {len(pending_signals)} pending, {len(completed_signals)} completed")
    except Exception as e:
        print(f"load_signals error: {e}")


# ================== TELEGRAM COMMANDS ==================
def send_help_command():
    msg = (
        "دستورات موجود\n====================\n\n"
        "از دکمه های پایین استفاده کن یا:\n\n"
        "/stats - آمار کلی\n/status - وضعیت ربات\n/analysis - آخرین تحلیل\n"
        "/today - آمار امروز\n/daily - خلاصه امروز\n/pause - توقف\n/resume - شروع\n/help - راهنما"
    )
    send_telegram_with_keyboard(msg)


def send_stats_command():
    total = len(completed_signals)
    wins = sum(1 for s in completed_signals if s.get('is_win') is True)
    losses = sum(1 for s in completed_signals if s.get('is_win') is False)
    ties = sum(1 for s in completed_signals if s.get('is_win') is None)
    decided = wins + losses
    wr = round(wins / decided * 100, 1) if decided > 0 else 0

    msg = "آمار کلی\n====================\n\n"
    msg += f"کل سیگنال: {total}\nبرد: {wins} | باخت: {losses}"
    if ties > 0:
        msg += f" | مساوی: {ties}"
    msg += f"\nوین ریت: {wr}%\nدر انتظار: {len(pending_signals)}\n\n"

    if completed_signals:
        msg += "آخرین 5 سیگنال:\n"
        for s in completed_signals[-5:]:
            emoji = "برد" if s.get('is_win') is True else ("باخت" if s.get('is_win') is False else "مساوی")
            msg += f"  {s['symbol']} {s['direction']} - {emoji}\n"

    send_telegram_with_keyboard(msg)


def send_status_command():
    msg = "وضعیت ربات\n====================\n\n"
    msg += f"مدل ها: {len(trained_models)}\n"
    msg += f"Live: {'فعال' if live_running else 'غیرفعال'}\n"
    msg += f"Backtest: {'در حال اجرا' if backtest_running else 'متوقف'}\n"
    msg += f"در انتظار: {len(pending_signals)}\n"
    msg += f"تکمیل شده: {len(completed_signals)}\n"
    msg += f"HTF Cache: {len(htf_cache)} items\n\n"
    if trained_models:
        msg += "جفت ارزها:\n"
        for name in trained_models.keys():
            msg += f"  {name}\n"
    send_telegram_with_keyboard(msg)


def send_analysis_command():
    msg = "آخرین تحلیل\n====================\n\n"
    with analysis_lock:
        if not last_analysis:
            msg += "هنوز تحلیلی انجام نشده."
        else:
            for sym, info in last_analysis.items():
                conf = info.get('confidence', 0)
                direction = info.get('direction', '?')
                status = info.get('status', '?')
                time_ir = info.get('time_iran', '?')
                if status == 'signal':
                    status_fa = 'سیگنال'
                elif status == 'rejected':
                    status_fa = 'رد شده'
                else:
                    status_fa = 'خطا'
                msg += f"{sym}\n  اطمینان: {conf}%\n  جهت: {direction}\n  وضعیت: {status_fa}\n  زمان: {time_ir}\n\n"
    msg += f"آستانه: {THRESHOLD}%"
    send_telegram_with_keyboard(msg)


def send_today_command():
    now_iran = to_iran(datetime.now(timezone.utc))
    today_start = now_iran.replace(hour=0, minute=0, second=0, microsecond=0)
    today_signals = []
    for s in completed_signals:
        try:
            if to_iran(s['entry_time']) >= today_start:
                today_signals.append(s)
        except:
            continue

    if not today_signals:
        send_telegram_with_keyboard("امروز هنوز سیگنالی تکمیل نشده.")
        return

    total = len(today_signals)
    wins = sum(1 for s in today_signals if s.get('is_win') is True)
    losses = sum(1 for s in today_signals if s.get('is_win') is False)
    decided = wins + losses
    wr = round(wins / decided * 100, 1) if decided > 0 else 0

    msg = f"آمار امروز ({now_iran.strftime('%Y-%m-%d')})\n====================\n\n"
    msg += f"کل: {total}\nبرد: {wins} | باخت: {losses}\nوین ریت: {wr}%\n\nبه تفکیک:\n"

    pair_stats = {}
    for s in today_signals:
        sym = s['symbol']
        if sym not in pair_stats:
            pair_stats[sym] = {'total': 0, 'wins': 0, 'losses': 0}
        pair_stats[sym]['total'] += 1
        if s.get('is_win') is True:
            pair_stats[sym]['wins'] += 1
        elif s.get('is_win') is False:
            pair_stats[sym]['losses'] += 1

    for sym, stats in pair_stats.items():
        d = stats['wins'] + stats['losses']
        pwr = round(stats['wins'] / d * 100, 1) if d > 0 else 0
        msg += f"  {sym}: {stats['total']} ({pwr}%)\n"

    send_telegram_with_keyboard(msg)


def handle_command(update):
    global live_running
    try:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        chat_id = msg.get("chat", {}).get("id")
        text = (msg.get("text") or "").strip()
        if str(chat_id) != str(CHAT_ID) or not text:
            return

        cmd = text.split()[0].lower()
        if cmd.startswith("/"):
            cmd = cmd.split("@")[0]

        if text == "آمار" or cmd == "/stats":
            send_stats_command()
        elif text == "وضعیت" or cmd == "/status":
            send_status_command()
        elif text == "تحلیل" or cmd == "/analysis":
            send_analysis_command()
        elif text == "امروز" or cmd == "/today":
            send_today_command()
        elif text == "خلاصه روز" or cmd == "/daily":
            send_daily_summary()
        elif text == "راهنما" or cmd == "/help":
            send_help_command()
        elif text == "توقف" or cmd in ["/pause", "/stop"]:
            if live_running:
                live_running = False
                send_telegram_with_keyboard("ربات موقتا متوقف شد.")
            else:
                send_telegram_with_keyboard("ربات از قبل متوقف بود.")
        elif text == "شروع" or cmd in ["/resume", "/start"]:
            if not trained_models:
                send_telegram_with_keyboard("مدل ها آموزش ندیدن.")
            elif live_running:
                send_telegram_with_keyboard("ربات فعاله.")
            else:
                live_running = True
                threading.Thread(target=live_loop, daemon=True).start()
                send_telegram_with_keyboard("ربات دوباره فعال شد.")
        else:
            send_telegram_with_keyboard(f"دستور ناشناخته: {text}")
    except Exception as e:
        print(f"handle_command error: {e}")


def poll_telegram_commands():
    global last_update_id
    time.sleep(30)
    while True:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            params = {"offset": last_update_id + 1, "timeout": 30}
            response = requests.get(url, params=params, timeout=40)
            data = response.json()
            if data.get("ok"):
                for update in data.get("result", []):
                    last_update_id = update["update_id"]
                    handle_command(update)
        except Exception as e:
            print(f"poll error: {e}")
            time.sleep(10)


# ================== CANDLE PATTERNS ==================
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

def backtest_symbol(name, symbol):
    try:
        print(f"Loading {name}...")
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        ts = td.time_series(symbol=symbol, interval=INTERVAL, outputsize=OUTPUTSIZE, timezone="UTC")
        df = ts.as_pandas()
        time.sleep(7)
        ts_h = td.time_series(symbol=symbol, interval=HTF_INTERVAL, outputsize=HTF_OUTPUTSIZE, timezone="UTC")
        df_h = ts_h.as_pandas()
        time.sleep(7)
        if df is None or df.empty or len(df) < 500:
            return {"symbol": name, "error": "not enough data"}
        df = df.rename(columns=str.lower).sort_index()
        df_h = df_h.rename(columns=str.lower).sort_index()
        htf_feat = build_htf_features(df_h)
        df = build_ltf_features(df)
        df = merge_htf_ltf(df, htf_feat)
        df = df.dropna()
        if len(df) < 500:
            return {"symbol": name, "error": "not enough data after filter"}

        X, y, meta = create_dataset(df, EXPIRY_CANDLES)
        if len(X) < 400:
            return {"symbol": name, "error": "not enough samples"}

        oos_probs, oos_y, oos_meta, fold_info = purged_walk_forward(X, y, meta, EXPIRY_CANDLES, N_FOLDS)
        if len(oos_probs) < 50:
            return {"symbol": name, "error": "not enough OOS"}

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

        if len(oos_meta) >= 2:
            days = (oos_meta[-1]['time'] - oos_meta[0]['time']).days
            days = max(days, 1)
        else:
            days = 1
        sig_per_day = round(signals / days, 1)

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
        return
    backtest_running = True
    try:
        send_telegram(
            "BACKTEST STARTED\n\n"
            "4 pairs selected\nExpiry: 30 min\n"
            f"Threshold: {THRESHOLD}%\n\nTakes 5-8 minutes..."
        )

        results = []
        for name, sym in SYMBOLS.items():
            r = backtest_symbol(name, sym)
            if "error" in r:
                send_telegram(f"ERROR {name}: {r['error']}")
                continue
            results.append(r)
            msg = (
                f"====================\n{r['symbol']}\n====================\n\n"
                f"Win Rate: {r['wr']}%\nSignals: {r['signals']} (~{r['signals_per_day']}/day)\n"
                f"Bank: $1000 -> ${r['final_bankroll']}\nReturn: {r['return_pct']}%\n"
                f"Folds: {r['folds']}"
            )
            send_telegram(msg)

        if results:
            total_signals = sum(r['signals'] for r in results)
            total_wins = sum(r['wins'] for r in results)
            avg_wr = round(total_wins / total_signals * 100, 2) if total_signals > 0 else 0
            avg_ret = round(sum(r['return_pct'] for r in results) / len(results), 2)
            final = (
                f"====================\nSUMMARY\n====================\n\n"
                f"Avg Win Rate: {avg_wr}%\nTotal Signals: {total_signals}\nAvg Return: +{avg_ret}%"
            )
            send_telegram(final)
    finally:
        backtest_running = False


# ================== ANALYZE LIVE ==================
def analyze_live(symbol):
    try:
        start = time.time()
        # 1. دریافت 15min
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        ts = td.time_series(symbol=symbol, interval=INTERVAL, outputsize=LIVE_LTF_SIZE, timezone="UTC")
        df = ts.as_pandas()
        if df is None or df.empty or len(df) < 250:
            print(f"analyze {symbol}: not enough data ({len(df) if df is not None else 0})")
            return None

        # 2. دریافت 1h از کش
        df_h = get_htf_data(symbol)
        if df_h is None or df_h.empty:
            print(f"analyze {symbol}: no HTF data")
            return None

        df = df.rename(columns=str.lower).sort_index()
        df_h = df_h.rename(columns=str.lower).sort_index()

        htf_feat = build_htf_features(df_h)
        df = build_ltf_features(df)
        df = merge_htf_ltf(df, htf_feat)
        df = df.dropna()

        if len(df) < 200:
            print(f"analyze {symbol}: after dropna only {len(df)} rows")
            return None

        last_row = df.iloc[-1]
        if last_row[FEATURE_COLS].isna().any():
            print(f"analyze {symbol}: NaN in features")
            return None

        model = trained_models.get(symbol)
        if model is None:
            print(f"analyze {symbol}: no model")
            return None

        X = last_row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        prob = model.predict_proba(X)[0]
        confidence = max(prob) * 100

        entry_time = df.index[-1]
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        entry_time = entry_time + timedelta(minutes=15)
        expiry_time = entry_time + timedelta(minutes=30)

        direction_tmp = "CALL" if prob[1] > prob[0] else "PUT"

        # ✅ با قفل
        with analysis_lock:
            last_analysis[symbol] = {
                "confidence": round(float(confidence), 2),
                "direction": direction_tmp,
                "time_utc": entry_time.strftime("%Y-%m-%d %H:%M"),
                "time_iran": fmt_iran(entry_time),
                "status": "signal" if confidence >= THRESHOLD else "rejected",
                "threshold": THRESHOLD,
            }

        elapsed = round(time.time() - start, 2)
        print(f"analyze {symbol}: {confidence:.1f}% {direction_tmp} ({elapsed}s)")

        if confidence < THRESHOLD:
            return None

        direction = "CALL" if prob[1] > prob[0] else "PUT"
        entry_price = float(last_row['close'])

        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry_price": entry_price,
            "entry_time": entry_time,
            "expiry_time": expiry_time,
        }
    except Exception as e:
        print(f"analyze error for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        with analysis_lock:
            last_analysis[symbol] = {
                "confidence": 0,
                "direction": "ERROR",
                "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "time_iran": fmt_iran(datetime.now(timezone.utc)),
                "status": "error",
                "error": str(e)[:200],
            }
        return None


def check_signal_result(signal):
    try:
        td = TDClient(apikey=TWELVE_DATA_API_KEY)
        ts = td.time_series(symbol=signal['symbol'], interval=INTERVAL, outputsize=10, timezone="UTC")
        df = ts.as_pandas()
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower).sort_index()

        expiry_time = signal['expiry_time']
        if expiry_time.tzinfo is None:
            expiry_time = expiry_time.replace(tzinfo=timezone.utc)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(timezone.utc)

        valid_candles = df[df.index <= expiry_time]
        if valid_candles.empty:
            return None

        exit_price = float(valid_candles.iloc[-1]['close'])
        entry_price = signal['entry_price']

        if exit_price == entry_price:
            return {'result': 'TIE', 'exit_price': exit_price, 'is_win': None}

        if signal['direction'] == 'CALL':
            is_win = exit_price > entry_price
        else:
            is_win = exit_price < entry_price

        return {'result': 'WIN' if is_win else 'LOSS', 'exit_price': exit_price, 'is_win': is_win}
    except Exception as e:
        print(f"check_result error: {e}")
        return None


def send_signal_result(signal):
    if signal['result'] == 'WIN':
        title = "TRADE WON"
    elif signal['result'] == 'LOSS':
        title = "TRADE LOST"
    else:
        title = "TRADE TIE"

    msg = (
        f"{title}\n====================\n"
        f"Symbol: {signal['symbol']}\nDirection: {signal['direction']}\n"
        f"Entry: {fmt_iran(signal['entry_time'])} IRAN\nEnd: {fmt_iran(signal['expiry_time'])} IRAN\n"
        f"Price In: {signal['entry_price']:.5f}\nPrice Out: {signal['exit_price']:.5f}\n"
    )

    total = len(completed_signals)
    wins = sum(1 for s in completed_signals if s.get('is_win') is True)
    losses = sum(1 for s in completed_signals if s.get('is_win') is False)
    ties = sum(1 for s in completed_signals if s.get('is_win') is None)

    if total > 0:
        decided = wins + losses
        wr = round(wins / decided * 100, 1) if decided > 0 else 0
        msg += f"\nSTATS:\nTotal: {total} | W: {wins} | L: {losses}"
        if ties > 0:
            msg += f" | T: {ties}"
        msg += f"\nWin Rate: {wr}%"

    send_telegram_with_keyboard(msg)


def result_checker_loop():
    global pending_signals, completed_signals
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            # ✅ با قفل برای جلوگیری از race
            with signal_lock:
                pending_copy = list(pending_signals)

            for signal in pending_copy:
                check_time = signal['expiry_time'] + timedelta(minutes=1)
                if now_utc >= check_time:
                    print(f"Checking result {signal['symbol']}")
                    result = check_signal_result(signal)
                    if result is None:
                        continue

                    with signal_lock:
                        if signal in pending_signals:
                            pending_signals.remove(signal)
                        signal['result'] = result['result']
                        signal['exit_price'] = result['exit_price']
                        signal['is_win'] = result['is_win']
                        completed_signals.append(signal)

                    send_signal_result(signal)
                    save_signals()
            time.sleep(30)
        except Exception as e:
            print(f"result_checker error: {e}")
            time.sleep(60)


# ================== SESSION FILTER ==================
def is_active_session():
    now_utc = datetime.now(timezone.utc)
    weekday = now_utc.weekday()
    hour = now_utc.hour
    if weekday >= 5:
        return False
    return 8 <= hour < 21


# ================== BIAS CHECK ==================
def check_direction_bias(symbol, direction):
    with streak_lock:
        if symbol not in direction_streak:
            direction_streak[symbol] = {"direction": direction, "count": 1}
            return {"streak": 1, "warn": False}
        info = direction_streak[symbol]
        if info["direction"] == direction:
            info["count"] += 1
        else:
            direction_streak[symbol] = {"direction": direction, "count": 1}
            info = direction_streak[symbol]
        warn = info["count"] >= MAX_SAME_DIRECTION_STREAK
        return {"streak": info["count"], "warn": warn}


def send_daily_summary():
    now_iran = to_iran(datetime.now(timezone.utc))
    today_start = now_iran.replace(hour=0, minute=0, second=0, microsecond=0)

    today_signals = []
    for s in completed_signals:
        try:
            if to_iran(s['entry_time']) >= today_start:
                today_signals.append(s)
        except:
            continue

    if not today_signals:
        send_telegram_with_keyboard(
            "خلاصه روز\n====================\n\n"
            f"تاریخ: {now_iran.strftime('%Y-%m-%d')}\n\nامروز هیچ سیگنالی صادر نشد"
        )
        return

    total = len(today_signals)
    wins = sum(1 for s in today_signals if s.get('is_win') is True)
    losses = sum(1 for s in today_signals if s.get('is_win') is False)
    ties = sum(1 for s in today_signals if s.get('is_win') is None)
    decided = wins + losses
    wr = round(wins / decided * 100, 1) if decided > 0 else 0

    pair_stats = {}
    for s in today_signals:
        sym = s['symbol']
        if sym not in pair_stats:
            pair_stats[sym] = {'total': 0, 'wins': 0, 'losses': 0}
        pair_stats[sym]['total'] += 1
        if s.get('is_win') is True:
            pair_stats[sym]['wins'] += 1
        elif s.get('is_win') is False:
            pair_stats[sym]['losses'] += 1

    msg = "خلاصه روز\n====================\n"
    msg += f"تاریخ: {now_iran.strftime('%Y-%m-%d')}\n\n"
    msg += f"کل سیگنال: {total}\nبرد: {wins} | باخت: {losses}"
    if ties > 0:
        msg += f" | مساوی: {ties}"
    msg += f"\nوین ریت: {wr}%\n\nبه تفکیک جفت ارز:\n"

    best_pair, best_wr = None, 0
    worst_pair, worst_wr = None, 100
    for sym, stats in pair_stats.items():
        d = stats['wins'] + stats['losses']
        pair_wr = round(stats['wins'] / d * 100, 1) if d > 0 else 0
        msg += f"  {sym}: {stats['total']} سیگنال ({pair_wr}%)\n"
        if stats['total'] >= 3:
            if pair_wr > best_wr:
                best_wr, best_pair = pair_wr, sym
            if pair_wr < worst_wr:
                worst_wr, worst_pair = pair_wr, sym

    if best_pair:
        msg += f"\nبهترین: {best_pair} ({best_wr}%)"
    if worst_pair and worst_pair != best_pair:
        msg += f"\nضعیف ترین: {worst_pair} ({worst_wr}%)"

    send_telegram_with_keyboard(msg)


def daily_summary_loop():
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            iran_time = to_iran(now_utc)
            if iran_time.hour == 0 and iran_time.minute == 30:
                today_key = iran_time.strftime("%Y-%m-%d")
                if today_key not in daily_summary_sent:
                    daily_summary_sent[today_key] = True
                    print(f"Sending daily summary for {today_key}")
                    send_daily_summary()
            time.sleep(60)
        except Exception as e:
            print(f"daily_summary error: {e}")
            time.sleep(60)


# ================== PROCESS SIGNAL (با قفل) ==================
def process_signal(symbol, result, now_utc, last_signal_time):
    """پردازش سیگنال — ✅ با قفل کامل"""
    global signal_id_counter
    try:
        bias_info = check_direction_bias(result['symbol'], result['direction'])
        sig_key = f"{result['symbol']}_{result['entry_time'].strftime('%Y%m%d%H%M')}"
        if sig_key == last_signal_time.get(result['symbol']):
            return
        last_signal_time[result['symbol']] = sig_key

        emoji = "CALL" if result['direction'] == "CALL" else "PUT"
        msg = (
            f"SIGNAL {emoji}\n====================\n"
            f"Symbol: {result['symbol']}\nEntry: {fmt_iran(result['entry_time'])} IRAN\n"
            f"Expiry: 30 min\nEnd: {fmt_iran(result['expiry_time'])} IRAN\n"
            f"Price: {result['entry_price']:.5f}\nConfidence: {result['confidence']}%"
        )
        if bias_info['warn']:
            msg += f"\n\nWARNING: {bias_info['streak']} same-direction signals in a row"

        # ✅ ارسال غیر بلاک
        send_telegram_async(msg, reply_markup=get_main_keyboard())
        print(f"Signal sent: {result['symbol']} {result['direction']}")

        # ✅ با قفل برای pending_signals + counter
        with signal_lock:
            signal_id_counter += 1
            result['id'] = signal_id_counter
            result['saved_at'] = now_utc
            pending_signals.append(result)

        save_signals()
    except Exception as e:
        print(f"process_signal error: {e}")


# ================== LIVE LOOP (موازی، rate-safe) ==================
def live_loop():
    global live_running
    last_signal_time = {}

    while live_running:
        try:
            now_utc = datetime.now(timezone.utc)
            minute = now_utc.minute
            second = now_utc.second

            if not is_active_session():
                time.sleep(60)
                continue

            # پنجره اسکن: 5-15 ثانیه بعد از بسته شدن کندل
            if minute % 15 == 0 and 5 <= second <= 15:
                slot_key = now_utc.strftime("%Y%m%d%H%M")
                if slot_key == last_signal_time.get("_slot"):
                    time.sleep(1)
                    continue
                last_signal_time["_slot"] = slot_key

                start_time = time.time()
                print(f"=== Scanning at {to_iran(now_utc).strftime('%H:%M')} Iran ===")

                # ✅ اجرای موازی با حداکثر 2 worker (rate-safe)
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                    futures = {
                        executor.submit(analyze_live, sym): (name, sym)
                        for name, sym in SYMBOLS.items()
                    }
                    for future in as_completed(futures):
                        name, sym = futures[future]
                        try:
                            result = future.result(timeout=30)
                            if result is not None:
                                process_signal(sym, result, now_utc, last_signal_time)
                        except Exception as e:
                            print(f"error {name}: {e}")

                elapsed = round(time.time() - start_time, 2)
                print(f"=== Scan completed in {elapsed}s ===")

            time.sleep(2)
        except Exception as e:
            print(f"live_loop error: {e}")
            time.sleep(10)


# ================== AUTO START ==================
def auto_start():
    global live_running, auto_start_done
    time.sleep(15)

    print("=" * 50)
    print("AUTO START")
    print("=" * 50)

    try:
        load_signals()
        send_telegram_with_keyboard("AUTO START\n\nPreparing bot...")

        if not trained_models:
            print("No models. Starting backtest...")
            run_backtest_background()
            while backtest_running:
                time.sleep(5)
            print(f"Backtest done. {len(trained_models)} models trained.")

        if trained_models and not live_running:
            live_running = True
            threading.Thread(target=live_loop, daemon=True).start()
            threading.Thread(target=result_checker_loop, daemon=True).start()
            threading.Thread(target=daily_summary_loop, daemon=True).start()
            threading.Thread(target=poll_telegram_commands, daemon=True).start()
            print("Live mode activated.")
            send_telegram_with_keyboard(
                "BOT READY!\n\n"
                f"{len(trained_models)} models trained\nExpiry: 30 min\n"
                f"Threshold: {THRESHOLD}%\n\n"
                "Signals will be sent automatically.\n"
                "Result of each signal reported 31 min later.\n\n"
                "از دکمه های پایین استفاده کن یا /help بزن"
            )
        auto_start_done = True
    except Exception as e:
        print(f"auto_start error: {e}")
        import traceback
        traceback.print_exc()
        send_telegram(f"Auto start error: {e}")


@app.route('/')
def health():
    return f"Signal Server | Models: {len(trained_models)} | Live: {live_running} | Pending: {len(pending_signals)} | Done: {len(completed_signals)} | Cache: {len(htf_cache)}"


@app.route('/backtest', methods=['GET'])
def backtest_route():
    threading.Thread(target=run_backtest_background, daemon=True).start()
    return jsonify({"status": "ok", "message": "Backtest started"}), 200


@app.route('/start_live', methods=['GET'])
def start_live_route():
    global live_running
    if live_running:
        return jsonify({"status": "already_running"}), 200
    if not trained_models:
        return jsonify({"status": "error", "message": "No models trained yet"}), 400
    live_running = True
    threading.Thread(target=live_loop, daemon=True).start()
    threading.Thread(target=result_checker_loop, daemon=True).start()
    threading.Thread(target=daily_summary_loop, daemon=True).start()
    send_telegram("Live mode activated.")
    return jsonify({"status": "started"}), 200


@app.route('/stop_live', methods=['GET'])
def stop_live_route():
    global live_running
    live_running = False
    send_telegram("Live mode stopped.")
    return jsonify({"status": "stopped"}), 200


@app.route('/status', methods=['GET'])
def status_route():
    return jsonify({
        "models": list(trained_models.keys()),
        "live_running": live_running,
        "backtest_running": backtest_running,
        "auto_start_done": auto_start_done,
        "pending": len(pending_signals),
        "completed": len(completed_signals),
        "direction_streak": direction_streak,
        "htf_cache_size": len(htf_cache),
    }), 200


@app.route('/analysis', methods=['GET'])
def analysis_route():
    now_utc = datetime.now(timezone.utc)
    result = {
        "current_time_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "current_time_iran": to_iran(now_utc).strftime("%Y-%m-%d %H:%M:%S"),
        "active_session": is_active_session(),
        "threshold": THRESHOLD,
        "symbols": {}
    }
    with analysis_lock:
        for name, sym in SYMBOLS.items():
            if sym in last_analysis:
                info = last_analysis[sym].copy()
                info["name"] = name
                result["symbols"][name] = info
            else:
                result["symbols"][name] = {"name": name, "status": "not_analyzed_yet", "message": "not analyzed yet"}
    return jsonify(result), 200


@app.route('/stats', methods=['GET'])
def stats_route():
    total = len(completed_signals)
    wins = sum(1 for s in completed_signals if s.get('is_win') is True)
    losses = sum(1 for s in completed_signals if s.get('is_win') is False)
    ties = sum(1 for s in completed_signals if s.get('is_win') is None)
    decided = wins + losses
    wr = round(wins / decided * 100, 2) if decided > 0 else 0
    recent_data = []
    for s in completed_signals[-10:]:
        recent_data.append({
            "symbol": s['symbol'], "direction": s['direction'],
            "result": s['result'], "time": fmt_iran(s['entry_time']),
        })
    return jsonify({
        "total": total, "wins": wins, "losses": losses, "ties": ties,
        "pending": len(pending_signals), "win_rate": wr, "recent": recent_data,
    }), 200


# ==================== AUTO START AT MODULE LEVEL ====================
_startup_thread = threading.Thread(target=auto_start, daemon=True)
_startup_thread.start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
