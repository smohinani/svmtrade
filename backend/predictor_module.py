# predictor_module.py
# Provides a callable function for pivot predictions using the 'newest.py' pipeline
# for integration with Streamlit applications.

import pandas as pd
import numpy as np
from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
from svm_predictor import extract_wave_features, train_calibrated_svm, predict_next_pivot


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names: flatten MultiIndex or capitalize columns.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    else:
        data.columns = [col.capitalize() for col in data.columns]
    return data


def filter_pivots(
    pivot_indices,
    close,
    atr,
    min_pct_move=0.01,
    min_atr_mult=0.3,
    cooldown_bars=2,
    min_abs_move=0.0
):
    filtered_pivots = []
    last_pivot_idx = None
    last_pivot_price = None
    for idx in pivot_indices:
        price = close.iloc[idx]
        current_atr = atr.iloc[idx]
        if not isinstance(price, (float, int)) or pd.isna(price):
            continue
        if not isinstance(current_atr, (float, int)) or pd.isna(current_atr):
            continue
        if last_pivot_idx is not None and last_pivot_price is not None:
            bars_since = idx - last_pivot_idx
            price_move = abs(price - last_pivot_price)
            min_move = max(min_pct_move * last_pivot_price, min_atr_mult * current_atr, min_abs_move)
            if bars_since < cooldown_bars:
                continue
            if price_move < min_move:
                continue
        filtered_pivots.append(idx)
        last_pivot_idx = idx
        last_pivot_price = price
    return filtered_pivots


def get_regime_from_predictions(pred_buffer, prev_regime=None, window=10, threshold=0.7):
    """
    Given a buffer of the last N predictions (strings: 'Peak', 'Trough', ...),
    return the regime: 'Peak' if >=threshold are 'Peak', 'Trough' if >=threshold are 'Trough', else prev_regime.
    """
    if len(pred_buffer) < window:
        # Not enough history, fallback to most common or previous
        if prev_regime is not None:
            return prev_regime
        if pred_buffer:
            return max(set(pred_buffer), key=pred_buffer.count)
        return None
    last_n = list(pred_buffer)[-window:]
    n_peak = sum(1 for p in last_n if p == 'Peak')
    n_trough = sum(1 for p in last_n if p == 'Trough')
    if n_peak >= threshold * window:
        return 'Peak'
    elif n_trough >= threshold * window:
        return 'Trough'
    else:
        return prev_regime


def run_pivot_prediction(
    symbol: str,
    interval: str,
    period: str = "60d",
    lookback: int = 5,
    adaptive: bool = True,
    base_order: int = 5,
    pred_buffer=None,
    prev_regime=None
) -> dict:
    # Fetch & prepare data
    data = fetch_market_data(symbol, interval, period)
    data = standardize_columns(data)
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Detect waves & compute metrics
    waves = detect_waves(data, adaptive=adaptive, base_order=base_order)
    waves = calculate_wave_metrics(waves, data)
    waves = calculate_pivot_confidence(waves, data)

    # Feature extraction & model training
    X, y = extract_wave_features(waves, lookback)
    if len(X) == 0:
        return {
            'predicted_type': None,
            'predicted_type_name': 'Unknown',
            'confidence': 0.0,
            'estimated_value': None,
            'estimated_index_offset': None,
            'regime': prev_regime or 'Unknown',
        }
    model, scaler = train_calibrated_svm(X, y)
    pred = predict_next_pivot(model, scaler, waves, lookback)

    # Regime smoothing
    if pred_buffer is not None:
        # Add the new prediction to the buffer
        pred_buffer.append(pred.get('predicted_type_name', 'Unknown'))
        # Keep only the last 10
        if len(pred_buffer) > 10:
            pred_buffer.pop(0)
        regime = get_regime_from_predictions(pred_buffer, prev_regime, window=10, threshold=0.7)
    else:
        regime = pred.get('predicted_type_name', 'Unknown')

    pred['regime'] = regime
    return pred

def get_entry_confirmation(df, signal, lookback=5):
    """
    Returns the confirmation price trigger level.
    """
    if signal == 'Trough':
        recent_low = df['Low'].rolling(lookback).min().iloc[-1]
        return round(recent_low + 0.3, 2)
    elif signal == 'Peak':
        recent_high = df['High'].rolling(lookback).max().iloc[-1]
        return round(recent_high - 0.3, 2)
    return None



def get_support_resistance(df, lookback=20):
    """
    Returns support and resistance levels from recent lows and highs.
    """
    support = df['Low'].rolling(window=lookback).min().iloc[-1]
    resistance = df['High'].rolling(window=lookback).max().iloc[-1]
    return round(support, 2), round(resistance, 2)
