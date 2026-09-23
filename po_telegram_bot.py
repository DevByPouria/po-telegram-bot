import os
import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ================== تنظیمات از Environment Variables ==================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def send_telegram(message):
    """ارسال پیام به تلگرام"""
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ خطا: BOT_TOKEN یا CHAT_ID تنظیم نشده")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"✅ تلگرام: {response.status_code}")
    except Exception as e:
        print(f"❌ خطا در ارسال: {e}")

@app.route('/webhook', methods=['POST'])
def webhook():
    """دریافت سیگنال از TradingView"""
    try:
        raw_data = request.get_data(as_text=True)
        print(f"📩 دریافت شد: {raw_data}")
        
        try:
            signal = json.loads(raw_data)
        except:
            signal = {"raw": raw_data}
        
        action = signal.get("action", "UNKNOWN")
        symbol = signal.get("symbol", "N/A")
        price = signal.get("price", "N/A")
        time_str = signal.get("time", "N/A")
        
        if action == "BUY":
            emoji = "🟢"
            action_fa = "خرید (CALL)"
        elif action == "SELL":
            emoji = "🔴"
            action_fa = "فروش (PUT)"
        else:
            emoji = "⚪"
            action_fa = action
        
        message = f"""{emoji} <b>سیگنال جدید</b>

📊 نماد: <code>{symbol}</code>
💰 قیمت: <code>{price}</code>
🕐 زمان: {time_str}
📌 نوع: <b>{action_fa}</b>"""
        
        send_telegram(message)
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def health():
    return "Signal Server is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
