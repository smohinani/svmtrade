import time
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from email.message import EmailMessage
import smtplib

# Your modules
from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
from svm_predictor import extract_wave_features, train_calibrated_svm, predict_next_pivot

# === Email Config ===
EMAIL_ADDRESS = 'samuelmohinanihkg@gmail.com'
EMAIL_PASSWORD = 'jccn gigr yebd aycl'
RECIPIENTS = ['samuel.mohinani@gmail.com']


def send_email(subject, message, image_path=None, to_email=RECIPIENTS):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ', '.join(to_email)
    msg.set_content(message)

    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            msg.add_attachment(img_data, maintype='image', subtype='png',
                               filename=os.path.basename(image_path))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print("✅ Email sent with chart.")
    except Exception as e:
        print(f"❌ Email failed: {e}")


def _flatten_and_capitalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If MultiIndex, drop the top level; else capitalize column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).capitalize() for col in df.columns]
    return df

from config import DEFAULT_SYMBOL
def pivot_signal_email(symbol=DEFAULT_SYMBOL, interval="15m", period="5d", threshold=0.75):
    print(f"🔄 Checking {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

    try:
        # === Fetch & clean data ===
        df = fetch_market_data(symbol, interval, period)
        df = _flatten_and_capitalize_columns(df)
        if hasattr(df.index, 'tzinfo') and df.index.tzinfo is not None or getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_localize(None)

        # === Wave Detection & features ===
        waves = detect_waves(df, adaptive=True)
        waves = calculate_wave_metrics(waves, df)
        waves = calculate_pivot_confidence(waves, df)
        X, y = extract_wave_features(waves, lookback=5)
        if len(X) == 0:
            print("❌ Not enough data for SVM.")
            return

        # === Train & predict ===
        svm_model, svm_scaler = train_calibrated_svm(X, y)
        pivot_pred = predict_next_pivot(svm_model, svm_scaler, waves, lookback=5)

        direction = pivot_pred.get("predicted_type", "")
        confidence = pivot_pred.get("confidence", 0)
        if not direction or confidence < threshold:
            print("📉 No strong pivot signal.")
            return

        current_price = pivot_pred.get("current_price", df["Close"].iloc[-1])
        next_price = pivot_pred.get("estimated_value", None)
        offset = pivot_pred.get("estimated_index_offset", 0)

        # === Compute next_time from interval + offset ===
        unit = interval[-1]
        qty = int(interval[:-1])
        if unit == 'm':
            next_time = df.index[-1] + pd.Timedelta(minutes=qty * offset)
        elif unit == 'h':
            next_time = df.index[-1] + pd.Timedelta(hours=qty * offset)
        else:
            next_time = df.index[-1]

        # === Next pivot direction & certainty ===
        nd_raw = pivot_pred.get("next_direction", None)
        if isinstance(nd_raw, (tuple, list)) and len(nd_raw) == 2:
            next_direction, next_certainty = nd_raw
        else:
            next_direction = str(nd_raw) if nd_raw else "unknown"
            next_certainty = pivot_pred.get("next_certainty", 0)

        # === Create & save chart ===
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["Close"], label="Close Price")
        if waves.get("peaks_indices"):
            plt.scatter(df.index[waves["peaks_indices"]],
                        df["Close"].iloc[waves["peaks_indices"]],
                        marker="^", s=80, label="Peaks")
        if waves.get("troughs_indices"):
            plt.scatter(df.index[waves["troughs_indices"]],
                        df["Close"].iloc[waves["troughs_indices"]],
                        marker="v", s=80, label="Troughs")
        if next_price is not None:
            plt.scatter([next_time], [next_price], marker="o", s=120, label="Next Pivot")
            plt.annotate(
                f"{direction.capitalize()}\nConf: {confidence*100:.1f}%",
                xy=(next_time, next_price),
                xytext=(10, 0),
                textcoords="offset points",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
            )
        plt.title(f"{symbol} Pivot Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(alpha=0.3)

        os.makedirs("output", exist_ok=True)
        chart_path = f"output/svm_trading_bot_{symbol}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        # === Email content ===
        subject = f"🔔 {symbol} {direction.upper()} Detected - Pivot Alert"
        message = (
            f"🟢 Detected Pivot:\n"
            f"- Type: {direction.capitalize()}\n"
            f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- Price: ${current_price:.2f}\n"
            f"- Confidence: {confidence*100:.1f}%\n\n"
            f"🔮 Next Pivot Prediction:\n"
            f"- Type: {next_direction.capitalize()}\n"
            f"- Time: {next_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- Price: ${next_price:.2f}\n"
            f"- Confidence: {next_certainty*100:.1f}%\n"
            f"- Model Method: {pivot_pred.get('method', 'N/A')}"
        )

        send_email(subject, message, image_path=chart_path)

    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")


if __name__ == "__main__":
    print("📬 Pivot Notifier Running...\n")
    while True:
        now = datetime.now()
        # NYSE open hours are roughly 9:30–16:00 ET; adjust as needed
        if now.weekday() < 5 and 6 <= now.hour <= 13:
            pivot_signal_email()
        else:
            print("🌙 Market closed. Sleeping…")
        time.sleep(900)  # wait 15 minutes
