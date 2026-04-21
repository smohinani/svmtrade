import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from wave_detector import (
    fetch_market_data,
    detect_waves,
    calculate_wave_metrics,
    calculate_pivot_confidence,
    get_latest_pivot,
    predict_next_pivot_simple,
)

st.set_page_config(page_title="Wave Detector Tester", layout="wide")
st.title("📊 Wave Detection Tester")

# --- Input fields ---
symbol = st.text_input("Symbol", value="SPY")
interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)
period = st.selectbox("Period", ["7d", "14d", "30d", "90d"], index=2)

# --- Run on button press ---
if st.button("Run Wave Detection"):
    try:
        df = fetch_market_data(symbol, interval=interval, period=period)
        st.success(f"✅ Pulled {len(df)} rows for {symbol}")
        st.dataframe(df.tail())

        # --- Wave Detection ---
        waves = detect_waves(df)
        waves = calculate_wave_metrics(waves, df)
        waves = calculate_pivot_confidence(waves, df)

        st.markdown(f"**Detected**: {len(waves['peaks_indices'])} peaks and {len(waves['troughs_indices'])} troughs")

        # --- Plotting ---
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close", color="blue")

        peaks = waves['peaks_indices']
        troughs = waves['troughs_indices']
        ax.scatter(df.index[peaks], df['Close'].iloc[peaks], color='red', label='Peaks')
        ax.scatter(df.index[troughs], df['Close'].iloc[troughs], color='green', label='Troughs')

        ax.legend()
        ax.set_title(f"{symbol} - Detected Waves")
        st.pyplot(fig)

        # --- Latest Pivot Info ---
        pivot = get_latest_pivot(waves)
        if pivot:
            st.info(
                f"**Latest Pivot:** {pivot['type_name']} at {pivot['date']} "
                f"→ Value: {pivot['value']:.2f} | Confidence: {pivot['confidence']:.2f}"
            )

        # --- Simple Prediction ---
        prediction = predict_next_pivot_simple(waves)
        if prediction:
            st.markdown("### 🔮 Simple Prediction")
            st.write({
                "Next Pivot Type": prediction["predicted_type_name"],
                "Est. Value": round(prediction["estimated_value"], 2),
                "Est. Bars Until": round(prediction["estimated_index_offset"], 1),
                "Confidence": prediction["confidence"]
            })

    except Exception as e:
        st.error(f"❌ Error: {e}")
