import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import modules for data fetching, wave detection, SVM, intraday, hourly pivot, etc.
from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
from svm_predictor import extract_wave_features, train_calibrated_svm, predict_next_pivot
from intraday_predictor import extract_price_extreme_features, train_extreme_prediction_models, predict_next_day_extremes
from hourly_pivot_predictor import extract_hourly_features, train_hourly_pivot_model, predict_next_hourly_pivot

# Import the new visualization and backtest functions
from visualization import (
    clean_yf_data,
    plot_wave_analysis,
    plot_predictions,
    backtest_strategy,
    plot_backtest_results
)
from config import DEFAULT_SYMBOL
class EnhancedTradingSystem:
    def __init__(self, symbol=DEFAULT_SYMBOL, interval="1h", period="60d"):
        self.symbol = symbol
        self.interval = interval
        self.period = period
        # You can store objects like data, models, or wave_data here
        self.data = None
        self.wave_data = {}
        self.models = {}
        self.scalers = {}

    def run(self):
        """Full pipeline: fetch data, wave detection, model train, predict, visualize, and backtest."""
        
        # -------------------------
        # 1) Fetch & Clean Data
        # -------------------------
        raw_data = fetch_market_data(self.symbol, self.interval, self.period)
        data = clean_yf_data(raw_data)
        self.data = data
        
        # -------------------------
        # 2) Wave Detection
        # -------------------------
        wave_data = detect_waves(data)
        wave_data = calculate_wave_metrics(wave_data, data)
        wave_data = calculate_pivot_confidence(wave_data, data)
        self.wave_data = wave_data
        
        # -------------------------
        # 3) Feature Extraction & Model Training
        # -------------------------
        # Wave-based pivot prediction
        X, y = extract_wave_features(wave_data, lookback=5)
        if len(X) > 0:
            model, scaler = train_calibrated_svm(X, y)
            self.models["pivot_svm"] = model
            self.scalers["pivot_scaler"] = scaler
            
            # 3a) Pivot Prediction
            pivot_prediction = predict_next_pivot(model, scaler, wave_data)
        else:
            pivot_prediction = {}
        
        # Intraday (high/low) models
        features = extract_price_extreme_features(data, wave_data)
        if not features.empty:
            intraday_models = train_extreme_prediction_models(features)
            intraday_prediction = predict_next_day_extremes(intraday_models, data, wave_data)
        else:
            intraday_prediction = {}
        
        # Hourly pivot model
        X_hourly, y_hourly = extract_hourly_features(data, wave_data)
        if len(X_hourly) > 0:
            hourly_model, hourly_scaler = train_hourly_pivot_model(X_hourly, y_hourly)
            hourly_prediction = predict_next_hourly_pivot(hourly_model, hourly_scaler, wave_data, data)
        else:
            hourly_prediction = {}
        
        # -------------------------
        # 4) Visualization
        # -------------------------
        # (Optional) Create and save images
        # plot_wave_analysis(data, wave_data, save_path="wave_analysis.png")
        # plot_predictions(data, wave_data, pivot_prediction, intraday_prediction, hourly_prediction, save_path="predictions.png")
        
        # -------------------------
        # 5) Backtest
        # -------------------------
        if "pivot_svm" in self.models and "pivot_scaler" in self.scalers:
            backtest_results = backtest_strategy(
                data, wave_data,
                svm_model=self.models["pivot_svm"],
                svm_scaler=self.scalers["pivot_scaler"],
                lookback=5,
                confidence_threshold=0.7,
                stop_loss_pct=1.0,
                take_profit_pct=2.0
            )
            # (Optional) plot_backtest_results(backtest_results, save_path="backtest_results.png")
        else:
            backtest_results = {}
        
        # ------------------------------------------------
        # Return predictions & backtest in a result dict
        # ------------------------------------------------
        results = {
            "predictions": {
                "pivot": pivot_prediction,
                "intraday": intraday_prediction,
                "hourly":  hourly_prediction
            },
            "backtest": backtest_results
        }
        
        return results
