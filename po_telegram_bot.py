import asyncio
import os
import threading
import pandas as pd
import ta
from flask import Flask
from telebot.async_telebot import AsyncTeleBot
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

# ================== تنظیمات ==================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
POCKET_OPTION_SSID = os.environ.get("POCKET_OPTION_SSID", "")
SYMBOL = "EURUSD_otc"
EXPIRY = 60
TRADE_AMOUNT = 1
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
CHECK_INTERVAL = 10

# ================== Flask (برای Render) ==================
app = Flask(__name__)

@app.route('/')
def health():
    return "Bot is running!"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# ================== متغیرهای جهانی ==================
auto_trade = False
chat_id = None
api = None
bot = AsyncTeleBot(BOT_TOKEN)

# ================== توابع ==================
def calculate_rsi(df, period=14):
    if len(df) < period + 1:
        return None
    rsi_series = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    return rsi_series.iloc[-1]

async def get_candles(symbol, timeframe=60, count=100):
    try:
        candles = await api.get_candles(symbol, timeframe, count)
        df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        print(f"❌ خطا در دریافت کندل: {e}")
        return None

async def execute_trade(direction):
    try:
        if direction == "call":
            result = await api.buy(SYMBOL, TRADE_AMOUNT, "call", EXPIRY)
            await bot.send_message(chat_id, f"✅ معامله CALL انجام شد. نتیجه: {result}")
        elif direction == "put":
            result = await api.buy(SYMBOL, TRADE_AMOUNT, "put", EXPIRY)
            await bot.send_message(chat_id, f"✅ معامله PUT انجام شد. نتیجه: {result}")
    except Exception as e:
        await bot.send_message(chat_id, f"❌ خطا در اجرای معامله: {e}")

async def trading_loop():
    global auto_trade, api
    while True:
        if not auto_trade or chat_id is None:
            await asyncio.sleep(2)
            continue
        try:
            df = await get_candles(SYMBOL, 60, 100)
            if df is None or len(df) < RSI_PERIOD + 1:
                await asyncio.sleep(CHECK_INTERVAL)
                continue
            rsi = calculate_rsi(df, RSI_PERIOD)
            if rsi is None:
                await asyncio.sleep(CHECK_INTERVAL)
                continue
            last_close = df['close'].iloc[-1]
            await bot.send_message(chat_id, f"📊 تحلیل: RSI = {rsi:.2f} | قیمت = {last_close}")
            if rsi < RSI_OVERSOLD:
                await bot.send_message(chat_id, "🔔 سیگنال خرید (CALL)!")
                await execute_trade("call")
            elif rsi > RSI_OVERBOUGHT:
                await bot.send_message(chat_id, "🔔 سیگنال فروش (PUT)!")
                await execute_trade("put")
            else:
                await bot.send_message(chat_id, "⏳ شرایط ورود فراهم نیست...")
        except Exception as e:
            try:
                await bot.send_message(chat_id, f"⚠️ خطا: {e}")
            except:
                pass
        await asyncio.sleep(CHECK_INTERVAL)

# ================== دستورات تلگرام ==================
@bot.message_handler(commands=['start'])
async def start(message):
    global chat_id
    chat_id = message.chat.id
    await bot.reply_to(message, "سلام! ربات آماده است.\n/auto_on روشن\n/auto_off خاموش\n/balance موجودی")

@bot.message_handler(commands=['auto_on'])
async def auto_on(message):
    global auto_trade, chat_id
    chat_id = message.chat.id
    auto_trade = True
    await bot.reply_to(message, "✅ معامله خودکار روشن شد.")

@bot.message_handler(commands=['auto_off'])
async def auto_off(message):
    global auto_trade
    auto_trade = False
    await bot.reply_to(message, "⛔ خاموش شد.")

@bot.message_handler(commands=['balance'])
async def balance(message):
    global chat_id
    chat_id = message.chat.id
    try:
        if api is None:
            await bot.reply_to(message, "❌ API متصل نیست. لطفاً منتظر بمانید یا لاگ‌ها را بررسی کنید.")
            return
        bal = await asyncio.wait_for(api.balance(), timeout=10.0)
        await bot.reply_to(message, f"💰 موجودی: {bal}$")
    except asyncio.TimeoutError:
        await bot.reply_to(message, "⏳ دریافت موجودی طول کشید. لطفاً دوباره تلاش کنید.")
    except Exception as e:
        await bot.reply_to(message, f"❌ خطا: {e}")

# ================== اجرا ==================
async def main():
    global api
    print("⏳ در حال اتصال به Pocket Option...")
    try:
        api = PocketOptionAsync(POCKET_OPTION_SSID)
        await api.connect()
        # تست اتصال با یک درخواست ساده
        bal = await asyncio.wait_for(api.balance(), timeout=15.0)
        print(f"✅ به Pocket Option متصل شد. موجودی: {bal}$")
    except asyncio.TimeoutError:
        print("❌ خطا: اتصال به Pocket Option timeout خورد.")
    except Exception as e:
        print(f"❌ خطا در اتصال به Pocket Option: {e}")
    
    print("🚀 ربات تلگرام در حال اجراست...")
    # پاک کردن وب‌هوک و پیام‌های قدیمی برای جلوگیری از تداخل
    await bot.delete_webhook(drop_pending_updates=True)
    await asyncio.gather(bot.polling(), trading_loop())

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(main())
