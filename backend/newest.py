#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import core modules from your project
from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
from svm_predictor import extract_wave_features, train_calibrated_svm, predict_next_pivot
from intraday_predictor import extract_price_extreme_features, train_extreme_prediction_models, predict_next_day_extremes
from hourly_pivot_predictor import extract_hourly_features, train_hourly_pivot_model, predict_next_hourly_pivot

def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the DataFrame column names.
    If the columns are a MultiIndex, use the first level.
    Otherwise, capitalize each column name.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    else:
        data.columns = [col.capitalize() for col in data.columns]
    return data

from config import DEFAULT_SYMBOL
def main():
    parser = argparse.ArgumentParser(
        description="SVM Trading Bot: Predict next pivot, intraday high/low, and (optionally) hourly pivot using SVM"
    )
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL, help="Ticker symbol")
    parser.add_argument("--interval", type=str, default="1h", help="Market data interval")
    parser.add_argument("--period", type=str, default="60d", help="Market data period")
    parser.add_argument("--lookback", type=int, default=5, help="Lookback period for feature extraction")
    args = parser.parse_args()

    os.makedirs("./output", exist_ok=True)

    # --- Data fetching & pre-processing ---
    print(f"Fetching data for {args.symbol} ({args.interval}, {args.period})...")
    data = fetch_market_data(args.symbol, args.interval, args.period)
    print(f"Fetched {len(data)} data points")
    
    # Fix column names (handle MultiIndex or tuple headers)
    data = standardize_columns(data)
    # Ensure the DataFrame index is tz-naive
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # --- Wave Detection ---
    print("Detecting waves using adaptive parameters...")
    wave_data = detect_waves(data, adaptive=True)
    wave_data = calculate_wave_metrics(wave_data, data)
    wave_data = calculate_pivot_confidence(wave_data, data)

    # --- SVM Pivot Prediction ---
    print("Extracting features for pivot prediction...")
    X, y = extract_wave_features(wave_data, lookback=args.lookback)
    if len(X) == 0:
        print("Not enough pivot points to train SVM model.")
        return

    print("Training SVM model for pivot prediction...")
    svm_model, svm_scaler = train_calibrated_svm(X, y)
    pivot_pred = predict_next_pivot(svm_model, svm_scaler, wave_data, lookback=args.lookback)
    print("Next pivot prediction:")
    print(pivot_pred)

    # --- Intraday High/Low Prediction ---
    features = extract_price_extreme_features(data, wave_data)
    if features.empty:
        print("Not enough data for intraday high/low prediction.")
        intraday_pred = None
    else:
        print("Training intraday high/low prediction models...")
        intraday_models = train_extreme_prediction_models(features)
        intraday_pred = predict_next_day_extremes(intraday_models, data, wave_data)
        print("Next day intraday predictions:")
        print(intraday_pred)

    # --- Hourly Pivot Prediction (optional) ---
    '''X_hourly, y_hourly = extract_hourly_features(data, wave_data, lookback=args.lookback)
    if len(X_hourly) > 0:
        print("Training hourly pivot prediction model...")
        hourly_model, hourly_scaler = train_hourly_pivot_model(X_hourly, y_hourly)
        hourly_pred = predict_next_hourly_pivot(hourly_model, hourly_scaler, wave_data, data, lookback=args.lookback)
        print("Next hourly pivot prediction:")
        print(hourly_pred)
    else:
        hourly_pred = None
        print("Not enough data for hourly pivot prediction.")'''

    # --- Visualization (optional) ---
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")
    if len(wave_data["peaks_indices"]) > 0:
        plt.scatter(
            data.index[wave_data["peaks_indices"]],
            data["Close"].iloc[wave_data["peaks_indices"]],
            color="green", marker="^", s=80, label="Peaks"
        )
    if len(wave_data["troughs_indices"]) > 0:
        plt.scatter(
            data.index[wave_data["troughs_indices"]],
            data["Close"].iloc[wave_data["troughs_indices"]],
            color="red", marker="v", s=80, label="Troughs"
        )
    if pivot_pred.get("predicted_type") is not None:
        last_date = data.index[-1]
        offset = pivot_pred.get("estimated_index_offset", 10)
        next_date = last_date + pd.Timedelta(hours=offset)
        pred_value = pivot_pred.get("estimated_value", data["Close"].iloc[-1])
        plt.scatter([next_date], [pred_value], color="purple", marker="o", s=120, label="Next Pivot Prediction")
        plt.annotate(
            f"{pivot_pred.get('predicted_type_name')}\nConf: {pivot_pred.get('confidence', 0):.2f}",
            xy=(next_date, pred_value),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    plt.title(f"SVM Trading Bot Predictions for {args.symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_chart = f"./output/svm_trading_bot_{args.symbol}.png"
    plt.savefig(out_chart, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {out_chart}")
    plt.show()

    # --- Neatly Print Final Findings ---
    print("\n----------------------------------------------------")
    print("Final Findings:")
    print("----------------------------------------------------")
    # Pivot Prediction
    print("Pivot Prediction:")
    if pivot_pred:
        print(f"  Type: {pivot_pred.get('predicted_type_name')}")
        print(f"  Confidence: {pivot_pred.get('confidence'):.2f}")
        print(f"  Estimated Value: {pivot_pred.get('estimated_value'):.2f}")
        print(f"  Estimated Index Offset (hours): {pivot_pred.get('estimated_index_offset')}")
        print(f"  Method: {pivot_pred.get('method')}")
    else:
        print("  No pivot prediction available.")
    
    # Intraday Prediction
    '''if intraday_pred:
        print("\nIntraday High/Low Prediction:")
        print(f"  High Prediction: {intraday_pred.get('high_prediction'):.2f}")
        print(f"    Confidence Interval: ({intraday_pred.get('high_ci_lower'):.2f}, {intraday_pred.get('high_ci_upper'):.2f})")
        print(f"  Low Prediction: {intraday_pred.get('low_prediction'):.2f}")
        print(f"    Confidence Interval: ({intraday_pred.get('low_ci_lower'):.2f}, {intraday_pred.get('low_ci_upper'):.2f})")
        print(f"  Overall Confidence: {intraday_pred.get('confidence'):.2f}")
        print(f"  Similar Patterns Found: {intraday_pred.get('similar_patterns_count')}")
    else:
        print("\nNo intraday prediction available.")'''

    # Hourly Prediction
    '''if hourly_pred:
        print("\nHourly Pivot Prediction:")
        print(f"  Type: {hourly_pred.get('predicted_type_name')}")
        print(f"  Confidence: {hourly_pred.get('confidence'):.2f}")
        print(f"  Estimated Value: {hourly_pred.get('estimated_value'):.2f}")
        timing = hourly_pred.get('timing', {})
        if timing:
            est_dt = timing.get('estimated_datetime')
            est_hours = timing.get('estimated_hours')
            timing_conf = timing.get('confidence')
            print(f"  Estimated Timing: {est_dt} (approximately {est_hours:.1f} hours ahead)")
            print(f"  Timing Confidence: {timing_conf:.2f}")
        else:
            print("  No timing information available.")
        print(f"  Method: {hourly_pred.get('method')}")
    else:
        print("\nNo hourly pivot prediction available.")
    print("----------------------------------------------------")'''

if __name__ == "__main__":
    main()