import os
import time
import requests
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ta
from twelvedata import TDClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

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
THRESHOLD = 70
N_FOLDS = 4
MIN_TRAIN_RATIO = 0.4


def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=20)
    except Exception as e:
        print(f"telegram error: {e}")


# ================== سناریوها ==================

FEATURES_CURRENT = [
    'rsi', 'bb_pos', 'macd', 'macd_signal', 'macd_diff',
    'adx', 'di_plus', 'di_minus',
    'dist_ema20', 'dist_ema50', 'dist_ema200',
    'candle_body', 'upper_wick', 'lower_wick',
    'price_change', 'volatility', 'rsi_change', 'macd_hist_change',
    'bull_engulf', 'bear_engulf', 'hammer', 'shooting',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'htf_trend', 'htf_rsi', 'htf_adx', 'htf_ema_dist',
]

FEATURES_PRUNED = [
    'rsi', 'bb_pos', 'macd', 'macd_diff',
    'adx', 'di_plus', 'di_minus',
    'dist_ema50', 'dist_ema200',
    'candle_body', 'upper_wick',
    'price_change', 'volatility',
    'htf_trend', 'htf_rsi', 'htf_adx', 'htf_ema_dist',
]

FEATURES_WITH_ATR = FEATURES_PRUNED + ['atr', 'bb_width']

FEATURES_EXTENDED = FEATURES_WITH_ATR + [
    'atr_ratio', 'stoch_k', 'cci', 'williams_r', 'body_atr',
    'dist_recent_high', 'dist_recent_low',
    'is_near_support', 'is_near_resistance',
    'morning_star', 'evening_star', 'piercing', 'dark_cloud',
]

FEATURES_FULL = FEATURES_EXTENDED + [
    'session', 'is_friday',
    'htf_trend_change', 'htf_rsi_change', 'htf_bb_pos',
]


# ================== ساخت همه فیچرها ==================

def build_all_features(df, df_htf):
    df = df.copy()

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_pos'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / (df['bb_mid'] + 1e-10)

    m = ta.trend.MACD(df['close'])
    df['macd'] = m.macd()
    df['macd_signal'] = m.macd_signal()
    df['macd_diff'] = m.macd_diff()

    df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 14, 3).stoch()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], 20).cci()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], 14).williams_r()

    a = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['adx'] = a.adx()
    df['di_plus'] = a.adx_pos()
    df['di_minus'] = a.adx_neg()

    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
    df['dist_ema20'] = (df['close'] - df['ema20']) / df['ema20']
    df['dist_ema50'] = (df['close'] - df['ema50']) / df['ema50']
    df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['atr_ma'] = df['atr'].rolling(50).mean()
    df['atr_ratio'] = df['atr'] / (df['atr_ma'] + 1e-10)

    df['candle_body'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_wick'] = (df['high'] - df[['close','open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_wick'] = (df[['close','open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['body_atr'] = (df['close'] - df['open']).abs() / (df['atr'] + 1e-10)

    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['price_change'].rolling(20).std()
    df['rsi_change'] = df['rsi'].diff()
    df['macd_hist_change'] = df['macd_diff'].diff()

    df['bull_engulf'] = 0
    df['bear_engulf'] = 0
    df['hammer'] = 0
    df['shooting'] = 0
    df['morning_star'] = 0
    df['evening_star'] = 0
    df['piercing'] = 0
    df['dark_cloud'] = 0
    df['doji'] = 0
    df['inside_bar'] = 0

    for i in range(3, len(df)):
        p, c = df.iloc[i-1], df.iloc[i]
        body_c = abs(c['close'] - c['open'])

        if p['close'] < p['open'] and c['close'] > c['open'] and c['close'] > p['open'] and c['open'] < p['close']:
            df.iloc[i, df.columns.get_loc('bull_engulf')] = 1
        if p['close'] > p['open'] and c['close'] < c['open'] and c['close'] < p['open'] and c['open'] > p['close']:
            df.iloc[i, df.columns.get_loc('bear_engulf')] = 1

        if body_c > 0:
            lw = min(c['close'], c['open']) - c['low']
            uw = c['high'] - max(c['close'], c['open'])
            if lw >= 2 * body_c and uw <= body_c * 0.8:
                df.iloc[i, df.columns.get_loc('hammer')] = 1
            if uw >= 2 * body_c and lw <= body_c * 0.8:
                df.iloc[i, df.columns.get_loc('shooting')] = 1

        if body_c / (c['high'] - c['low'] + 1e-10) < 0.1:
            df.iloc[i, df.columns.get_loc('doji')] = 1

        if c['high'] <= p['high'] and c['low'] >= p['low']:
            df.iloc[i, df.columns.get_loc('inside_bar')] = 1

        c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        b1 = abs(c1['close'] - c1['open'])
        b2 = abs(c2['close'] - c2['open'])
        b3 = abs(c3['close'] - c3['open'])
        if b1 > 0 and b2 > 0 and b3 > 0:
            if (c1['close'] < c1['open'] and b1 > b2 * 1.2 and
                c3['close'] > c3['open'] and b3 > b2 * 1.2 and
                c3['close'] > (c1['open'] + c1['close']) / 2):
                df.iloc[i, df.columns.get_loc('morning_star')] = 1
            if (c1['close'] > c1['open'] and b1 > b2 * 1.2 and
                c3['close'] < c3['open'] and b3 > b2 * 1.2 and
                c3['close'] < (c1['open'] + c1['close']) / 2):
                df.iloc[i, df.columns.get_loc('evening_star')] = 1

        if p['close'] < p['open'] and c['close'] > c['open']:
            mid = (p['open'] + p['close']) / 2
            if c['open'] < p['close'] and c['close'] > mid:
                df.iloc[i, df.columns.get_loc('piercing')] = 1
        if p['close'] > p['open'] and c['close'] < c['open']:
            mid = (p['open'] + p['close']) / 2
            if c['open'] > p['close'] and c['close'] < mid:
                df.iloc[i, df.columns.get_loc('dark_cloud')] = 1

    df['recent_high'] = df['high'].rolling(20).max()
    df['recent_low'] = df['low'].rolling(20).min()
    df['dist_recent_high'] = (df['recent_high'] - df['close']) / df['close']
    df['dist_recent_low'] = (df['close'] - df['recent_low']) / df['close']
    df['is_near_support'] = (df['dist_recent_low'] < 0.001).astype(int)
    df['is_near_resistance'] = (df['dist_recent_high'] < 0.001).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['is_friday'] = (df.index.dayofweek == 4).astype(int)

    def get_session(h):
        if 0 <= h < 8: return 1
        if 8 <= h < 16: return 2
        if 16 <= h < 21: return 3
        return 0
    df['session'] = df.index.hour.map(get_session)

    df_h = df_htf.copy()
    df_h['htf_ema50'] = ta.trend.EMAIndicator(df_h['close'], 50).ema_indicator()
    df_h['htf_trend'] = (df_h['close'] > df_h['htf_ema50']).astype(int)
    df_h['htf_rsi'] = ta.momentum.RSIIndicator(df_h['close'], 14).rsi()
    adx_h = ta.trend.ADXIndicator(df_h['high'], df_h['low'], df_h['close'], 14)
    df_h['htf_adx'] = adx_h.adx()
    df_h['htf_ema_dist'] = (df_h['close'] - df_h['htf_ema50']) / df_h['htf_ema50']
    bb_h = ta.volatility.BollingerBands(df_h['close'], 20, 2)
    df_h['htf_bb_pos'] = (df_h['close'] - bb_h.bollinger_lband()) / (bb_h.bollinger_hband() - bb_h.bollinger_lband() + 1e-10)
    df_h['htf_trend_change'] = df_h['htf_trend'].diff().fillna(0)
    df_h['htf_rsi_change'] = df_h['htf_rsi'].diff().fillna(0)
    df_h['close_time'] = df_h.index + pd.Timedelta(hours=1)

    cols = ['htf_trend', 'htf_rsi', 'htf_adx', 'htf_ema_dist', 'htf_bb_pos', 'htf_trend_change', 'htf_rsi_change', 'close_time']
    htf_feat = df_h[cols].set_index('close_time')

    left = df.reset_index()
    if 'time' not in left.columns:
        left['time'] = df.index
    right = htf_feat.reset_index().rename(columns={'close_time': 'time'})
    left = left.sort_values('time')
    right = right.sort_values('time')
    merged = pd.merge_asof(left, right, on='time', direction='backward', allow_exact_matches=True)
    merged = merged.set_index('time')

    return merged


def create_dataset(df, features, expiry):
    X, y = [], []
    for i in range(250, len(df) - expiry):
        row = df.iloc[i]
        if row[features].isna().any():
            continue
        entry = row['close']
        exit_p = df['close'].iloc[i + expiry]
        if exit_p == entry:
            continue
        label = 1 if exit_p > entry else 0
        X.append(row[features].values.astype(float))
        y.append(label)
    return np.array(X), np.array(y)


def purged_wf(X, y, expiry, n_folds):
    n = len(X)
    embargo = expiry
    remaining = n - int(n * MIN_TRAIN_RATIO)
    fold_size = remaining // n_folds
    oos_probs, oos_y = [], []
    fold_accs = []

    for fold in range(n_folds):
        train_end = int(n * MIN_TRAIN_RATIO) + fold * fold_size
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start or train_end < 200:
            continue
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        if len(X_te) < 20:
            continue
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=20, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        fold_accs.append(accuracy_score(y_te, model.predict(X_te)))
        oos_probs.extend(probs)
        oos_y.extend(y_te)

    return np.array(oos_probs), np.array(oos_y), fold_accs


def test_scenario(features, df, expiry):
    try:
        X, y = create_dataset(df, features, expiry)
        if len(X) < 400:
            return {"error": "not enough samples"}

        probs, y_test, folds = purged_wf(X, y, expiry, N_FOLDS)
        if len(probs) < 50:
            return {"error": "not enough OOS"}

        signals, wins = 0, 0
        for j, prob in enumerate(probs):
            if max(prob) * 100 < THRESHOLD:
                continue
            pred = 1 if prob[1] > prob[0] else 0
            signals += 1
            wins += int(pred == y_test[j])

        wr = (wins / signals * 100) if signals > 0 else 0
        return {
            "signals": signals,
            "wins": wins,
            "wr": round(wr, 2),
            "folds": [round(f*100, 2) for f in folds],
        }
    except Exception as e:
        return {"error": str(e)}


def run_feature_test():
    scenarios = [
        ("فعلی (30 فیچر)", FEATURES_CURRENT, "همه 30 فیچر فعلی ربات"),
        ("حذف تکراری ها (17)", FEATURES_PRUNED, "حذف فیچرهای تکراری و بی فایده"),
        ("با ATR و BB (19)", FEATURES_WITH_ATR, "اضافه کردن ATR و BB Width"),
        ("توسعه یافته (32)", FEATURES_EXTENDED, "اضافه کردن سطوح و الگوهای بیشتر"),
        ("کامل (37)", FEATURES_FULL, "اضافه کردن فیچرهای زمانی و HTF بیشتر"),
    ]

    send_telegram(
        "تست فیچرها شروع شد\n\n"
        "5 سناریو تست میشه:\n\n"
        "1. فعلی (30 فیچر)\n"
        "   همه فیچرهای فعلی ربات\n\n"
        "2. حذف تکراری ها (17)\n"
        "   حذف فیچرهای تکراری و بی فایده\n\n"
        "3. با ATR و BB (19)\n"
        "   اضافه کردن ATR و BB Width\n\n"
        "4. توسعه یافته (32)\n"
        "   اضافه کردن سطوح و الگوهای بیشتر\n\n"
        "5. کامل (37)\n"
        "   اضافه کردن فیچرهای زمانی و HTF بیشتر\n\n"
        "این تست 15 تا 20 دقیقه طول میکشه\n\n"
        "هدف: بفهمیم کدوم ترکیب فیچرها بهترین وین ریت رو میده"
    )

    time.sleep(2)

    totals = {name: {"signals": 0, "wins": 0} for name, _, _ in scenarios}

    for symbol, sym in SYMBOLS.items():
        print(f"Loading {symbol}...")
        send_telegram(f"در حال دانلود داده {symbol}...")

        td = TDClient(apikey=TWELVE_DATA_API_KEY)

        ts = td.time_series(symbol=sym, interval=INTERVAL, outputsize=OUTPUTSIZE, timezone="UTC")
        df = ts.as_pandas()
        time.sleep(8)

        ts_h = td.time_series(symbol=sym, interval=HTF_INTERVAL, outputsize=HTF_OUTPUTSIZE, timezone="UTC")
        df_h = ts_h.as_pandas()
        time.sleep(8)

        if df is None or df.empty or len(df) < 500:
            send_telegram(f"خطا در {symbol}: داده کافی نیست")
            continue

        df = df.rename(columns=str.lower).sort_index()
        df_h = df_h.rename(columns=str.lower).sort_index()

        try:
            df = build_all_features(df, df_h)
            df = df.dropna()
        except Exception as e:
            send_telegram(f"خطا در ساخت فیچرهای {symbol}: {str(e)[:150]}")
            continue

        if len(df) < 500:
            send_telegram(f"خطا در {symbol}: داده پس از فیلتر کم است")
            continue

        msg = f"نتایج {symbol}\n"
        msg += "====================\n\n"

        for name, features, desc in scenarios:
            result = test_scenario(features, df, EXPIRY_CANDLES)

            if "error" in result:
                msg += f"{name}\n"
                msg += f"خطا: {result['error']}\n\n"
                continue

            msg += f"{name}\n"
            msg += f"وین ریت: {result['wr']}%\n"
            msg += f"سیگنال: {result['signals']}\n"
            msg += f"برد: {result['wins']} | باخت: {result['signals'] - result['wins']}\n"
            msg += f"دقت فولدها: {result['folds']}\n\n"

            totals[name]["signals"] += result['signals']
            totals[name]["wins"] += result['wins']

        send_telegram(msg)

    # خلاصه نهایی
    final = "خلاصه نهایی\n"
    final += "====================\n\n"

    best_wr = 0
    best_name = ""

    for name, _, desc in scenarios:
        t = totals[name]
        wr = round(t['wins'] / t['signals'] * 100, 2) if t['signals'] > 0 else 0

        if wr > best_wr and t['signals'] > 100:
            best_wr = wr
            best_name = name

        final += f"{name}\n"
        final += f"وین ریت کل: {wr}%\n"
        final += f"سیگنال کل: {t['signals']}\n"
        final += f"برد: {t['wins']} | باخت: {t['signals'] - t['wins']}\n"
        final += f"توضیح: {desc}\n\n"

    final += "====================\n"
    if best_name:
        final += f"بهترین سناریو: {best_name}\n"
        final += f"با وین ریت {best_wr}%\n"
    final += "\nنتیجه:\n"
    final += "اگه یک سناریو وین ریت بالاتری داد،\n"
    final += "می تونیم فیچرهای ربات رو به اون تغییر بدیم"

    send_telegram(final)


if __name__ == "__main__":
    run_feature_test()
