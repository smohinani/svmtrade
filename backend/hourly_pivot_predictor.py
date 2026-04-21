"""
Hourly Pivot Prediction Module

This module provides specialized functionality for predicting pivot points on hourly charts.
It includes time-of-day patterns, pivot timing estimation, and confidence calculation
specifically optimized for hourly timeframes.

Dependencies: numpy, pandas, scikit-learn, statsmodels
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Tuple, List, Union, Optional
import datetime
import os
import joblib

# Import wave detection and SVM prediction modules
import sys
sys.path.append('.')
from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
from svm_predictor import extract_wave_features


def extract_hourly_features(data: pd.DataFrame, wave_data: Dict, lookback: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features specifically designed for hourly pivot prediction.
    
    Features include:
    - Standard wave features
    - Hour of day
    - Day of week
    - Time since last pivot
    - Hourly volatility patterns
    
    Args:
        data: DataFrame with market data
        wave_data: Dictionary with wave detection results
        lookback: Number of previous pivot points to include
        
    Returns:
        Tuple of (features, targets) arrays
    """
    # Get basic wave features
    X_basic, y_basic = extract_wave_features(wave_data, lookback)
    
    if len(X_basic) == 0:
        return np.array([]), np.array([])
    
    # Get pivot indices and dates
    all_pivot_indices = wave_data['all_pivot_indices']
    all_pivot_dates = pd.to_datetime(wave_data['all_pivot_dates'])
    pivot_types = wave_data['pivot_types']
    
    # Create enhanced features
    X_enhanced = []
    
    # For each sample in the basic features
    for i in range(len(X_basic)):
        # Get the corresponding pivot index
        pivot_idx = lookback + i
        
        # Get the pivot date
        pivot_date = all_pivot_dates[pivot_idx]
        
        # Extract time-based features
        hour_of_day = pivot_date.hour
        day_of_week = pivot_date.dayofweek
        
        # One-hot encode hour of day (trading hours: 9-16)
        hour_features = np.zeros(8)
        if 9 <= hour_of_day <= 16:
            hour_features[hour_of_day - 9] = 1
        
        # One-hot encode day of week
        day_features = np.zeros(5)
        if 0 <= day_of_week <= 4:  # Monday-Friday
            day_features[day_of_week] = 1
        
        # Calculate time since last pivot
        if pivot_idx > 0:
            prev_date = all_pivot_dates[pivot_idx - 1]
            time_diff = (pivot_date - prev_date).total_seconds() / 3600  # in hours
        else:
            time_diff = 0
        
        # Calculate average time between pivots
        if pivot_idx >= 3:
            date_diffs = []
            for j in range(pivot_idx - 3, pivot_idx):
                if j > 0:
                    diff = (all_pivot_dates[j] - all_pivot_dates[j-1]).total_seconds() / 3600
                    date_diffs.append(diff)
            avg_time_between_pivots = np.mean(date_diffs) if date_diffs else 0
        else:
            avg_time_between_pivots = 0
        
        # Calculate hourly volatility
        if 'Close' in data.columns:
            # Get hourly returns
            hourly_returns = data['Close'].pct_change()
            
            # Get volatility for the same hour of day
            same_hour_mask = [d.hour == hour_of_day for d in data.index]
            same_hour_returns = hourly_returns[same_hour_mask]
            hour_volatility = same_hour_returns.std() if len(same_hour_returns) > 0 else 0
            
            # Get volatility for the same day of week
            same_day_mask = [d.dayofweek == day_of_week for d in data.index]
            same_day_returns = hourly_returns[same_day_mask]
            day_volatility = same_day_returns.std() if len(same_day_returns) > 0 else 0
        else:
            hour_volatility = 0
            day_volatility = 0
        
        # Combine all features
        enhanced_features = np.concatenate([
            X_basic[i],
            hour_features,
            day_features,
            [time_diff, avg_time_between_pivots, hour_volatility, day_volatility]
        ])
        
        X_enhanced.append(enhanced_features)
    
    return np.array(X_enhanced), y_basic


def train_hourly_pivot_model(X: np.ndarray, y: np.ndarray) -> Tuple[SVC, StandardScaler]:
    """
    Train a specialized SVM model for hourly pivot prediction.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuple of (model, scaler)
    """
    # Create scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create time series split for financial data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'poly']
    }
    
    # Create SVM model
    svm = SVC(probability=True)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_scaled, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    return best_model, scaler


def estimate_pivot_timing(wave_data: Dict, data: pd.DataFrame) -> Dict:
    """
    Estimate when the next pivot is likely to occur based on historical patterns.
    
    Args:
        wave_data: Dictionary with wave detection results
        data: DataFrame with market data
        
    Returns:
        Dictionary with timing estimation
    """
    # Get pivot indices and dates
    all_pivot_indices = wave_data['all_pivot_indices']
    all_pivot_dates = pd.to_datetime(wave_data['all_pivot_dates'])
    
    if len(all_pivot_dates) < 3:
        return {
            'estimated_hours': 4,  # Default estimate
            'confidence': 0.5,
            'estimated_datetime': None
        }
    
    # Calculate time differences between consecutive pivots
    time_diffs = []
    for i in range(1, len(all_pivot_dates)):
        diff = (all_pivot_dates[i] - all_pivot_dates[i-1]).total_seconds() / 3600  # in hours
        time_diffs.append(diff)
    
    # Calculate statistics
    mean_diff = np.mean(time_diffs)
    median_diff = np.median(time_diffs)
    std_diff = np.std(time_diffs)
    
    # Use more recent pivots for better estimation
    recent_count = min(5, len(time_diffs))
    recent_diffs = time_diffs[-recent_count:]
    recent_mean = np.mean(recent_diffs)
    
    # Check if there's a pattern by hour of day
    hour_patterns = {}
    for i in range(1, len(all_pivot_dates)):
        hour = all_pivot_dates[i-1].hour
        if hour not in hour_patterns:
            hour_patterns[hour] = []
        hour_patterns[hour].append(time_diffs[i-1])
    
    # Find the hour with the most consistent pattern
    hour_std = {h: np.std(diffs) for h, diffs in hour_patterns.items() if len(diffs) >= 3}
    
    # Get the last pivot date
    last_pivot_date = all_pivot_dates[-1]
    last_hour = last_pivot_date.hour
    
    # Estimate next pivot timing
    if last_hour in hour_patterns and len(hour_patterns[last_hour]) >= 3:
        # Use hour-specific pattern
        estimated_hours = np.mean(hour_patterns[last_hour])
        confidence = 1.0 - (np.std(hour_patterns[last_hour]) / estimated_hours)
    else:
        # Use recent average
        estimated_hours = recent_mean
        confidence = 1.0 - (std_diff / mean_diff)
    
    # Ensure confidence is between 0 and 1
    confidence = max(0.0, min(1.0, confidence))
    
    # Calculate estimated datetime
    estimated_datetime = last_pivot_date + pd.Timedelta(hours=estimated_hours)
    
    return {
        'estimated_hours': estimated_hours,
        'confidence': confidence,
        'estimated_datetime': estimated_datetime
    }


def predict_next_hourly_pivot(model, scaler: StandardScaler, wave_data: Dict, data: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Predict the next pivot point on hourly charts.
    
    Args:
        model: Trained SVM model
        scaler: Feature scaler
        wave_data: Dictionary with wave detection results
        data: DataFrame with market data
        lookback: Number of previous pivot points to include
        
    Returns:
        Dictionary with prediction results
    """
    # Extract hourly features for the current state
    all_pivot_indices = wave_data['all_pivot_indices']
    all_pivot_dates = pd.to_datetime(wave_data['all_pivot_dates'])
    pivot_types = wave_data['pivot_types']
    wave_heights = wave_data['wave_heights']
    
    # Get additional metrics if available
    wave_durations = wave_data.get('wave_durations', np.zeros_like(wave_heights))
    wave_slopes = wave_data.get('wave_slopes', np.zeros_like(wave_heights))
    confidence_scores = wave_data.get('confidence_scores', np.zeros_like(wave_heights))
    
    # Check if we have enough data
    if len(all_pivot_indices) < lookback:
        return {
            'predicted_type': None,
            'predicted_type_name': None,
            'confidence': 0.0,
            'method': 'hourly_insufficient_data'
        }
    
    # Extract basic wave features
    feature_vector = []
    
    # Calculate mean height and duration for normalization
    if len(wave_heights) > 1:
        mean_height = np.mean(wave_heights[1:])
    else:
        mean_height = 1.0
        
    if len(wave_durations) > 1:
        mean_duration = np.mean(wave_durations[1:])
    else:
        mean_duration = 1.0
    
    # Add features for each wave in the lookback window
    for j in range(-lookback, 1):  # Include current pivot
        idx = len(all_pivot_indices) + j - 1
        if idx < 0:
            # Not enough history, use zeros as padding
            feature_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # height, type, duration, slope, confidence
        else:
            # Add normalized wave height
            if mean_height > 0:
                norm_height = wave_heights[idx] / mean_height
            else:
                norm_height = wave_heights[idx]
            feature_vector.append(float(norm_height))
            
            # Add wave type (1 for peak, -1 for trough)
            feature_vector.append(float(pivot_types[idx]))
            
            # Add normalized wave duration
            if idx > 0 and mean_duration > 0:
                norm_duration = wave_durations[idx] / mean_duration
            else:
                norm_duration = 0.0
            feature_vector.append(float(norm_duration))
            
            # Add wave slope
            feature_vector.append(float(wave_slopes[idx]) if idx > 0 else 0.0)
            
            # Add confidence score
            feature_vector.append(float(confidence_scores[idx]))
    
    # Get the last pivot date
    last_pivot_date = all_pivot_dates[-1]
    
    # Extract time-based features
    hour_of_day = last_pivot_date.hour
    day_of_week = last_pivot_date.dayofweek
    
    # One-hot encode hour of day (trading hours: 9-16)
    hour_features = np.zeros(8)
    if 9 <= hour_of_day <= 16:
        hour_features[hour_of_day - 9] = 1
    
    # One-hot encode day of week
    day_features = np.zeros(5)
    if 0 <= day_of_week <= 4:  # Monday-Friday
        day_features[day_of_week] = 1
    
    # Calculate time since last pivot
    if len(all_pivot_dates) > 1:
        prev_date = all_pivot_dates[-2]
        time_diff = (last_pivot_date - prev_date).total_seconds() / 3600  # in hours
    else:
        time_diff = 0
    
    # Calculate average time between pivots
    if len(all_pivot_dates) >= 4:
        date_diffs = []
        for j in range(len(all_pivot_dates) - 4, len(all_pivot_dates) - 1):
            if j > 0:
                diff = (all_pivot_dates[j] - all_pivot_dates[j-1]).total_seconds() / 3600
                date_diffs.append(diff)
        avg_time_between_pivots = np.mean(date_diffs) if date_diffs else 0
    else:
        avg_time_between_pivots = 0
    
    # Calculate hourly volatility
    if 'Close' in data.columns:
        # Get hourly returns
        hourly_returns = data['Close'].pct_change()
        
        # Get volatility for the same hour of day
        same_hour_mask = [d.hour == hour_of_day for d in data.index]
        same_hour_returns = hourly_returns[same_hour_mask]
        hour_volatility = same_hour_returns.std() if len(same_hour_returns) > 0 else 0
        
        # Get volatility for the same day of week
        same_day_mask = [d.dayofweek == day_of_week for d in data.index]
        same_day_returns = hourly_returns[same_day_mask]
        day_volatility = same_day_returns.std() if len(same_day_returns) > 0 else 0
    else:
        hour_volatility = 0
        day_volatility = 0
    
    # Combine all features
    enhanced_features = np.concatenate([
        feature_vector,
        hour_features,
        day_features,
        [time_diff, avg_time_between_pivots, hour_volatility, day_volatility]
    ])
    
    # Scale features
    X = np.array([enhanced_features])
    
    # Ensure feature dimensions match what the model expects
    expected_features = X.shape[1]
    model_features = getattr(model, 'n_features_in_', None)
    
    if model_features is not None and expected_features != model_features:
        print(f"WARNING: Feature dimensions mismatch. Got {expected_features}, expected {model_features}.")
        # Adjust feature vector to match model expectations
        if expected_features < model_features:
            # Pad with zeros if we have fewer features than expected
            padding = np.zeros((X.shape[0], model_features - expected_features))
            X = np.hstack((X, padding))
        else:
            # Truncate if we have more features than expected
            X = X[:, :model_features]
    
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    # Get probability
    probabilities = model.predict_proba(X_scaled)[0]
    probability = np.max(probabilities)
    
    # Get prediction type name
    pred_type_name = "Peak" if prediction == 1 else "Trough"
    
    # Estimate next pivot value and timing
    # Get the last pivot
    last_pivot_value = wave_data['all_pivot_values'][-1]
    
    # Calculate average wave metrics for estimation
    if len(wave_heights) > 1:
        # Use recent waves for better estimation (last 5 or fewer)
        recent_count = min(5, len(wave_heights) - 1)
        avg_height = np.mean(wave_heights[-recent_count:])
    else:
        avg_height = 0.01 * last_pivot_value  # Default 1% move
    
    # Estimate next pivot value
    if prediction == 1:  # Peak
        estimated_value = last_pivot_value + avg_height
    else:  # Trough
        estimated_value = last_pivot_value - avg_height
    
    # Estimate pivot timing
    timing = estimate_pivot_timing(wave_data, data)
    
    return {
        'predicted_type': prediction,
        'predicted_type_name': pred_type_name,
        'confidence': probability,
        'estimated_value': estimated_value,
        'timing': timing,
        'method': 'hourly_svm_prediction'
    }


def save_hourly_model(model, scaler: StandardScaler, model_dir: str = './models'):
    """
    Save trained hourly pivot model to disk.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        model_dir: Directory to save model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = os.path.join(model_dir, f'hourly_model_{timestamp}.joblib')
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(model_dir, f'hourly_scaler_{timestamp}.joblib')
    joblib.dump(scaler, scaler_path)
    
    print(f"Hourly model saved to {model_path}")
    print(f"Hourly scaler saved to {scaler_path}")
    
    return model_path, scaler_path


def load_hourly_model(model_path: str, scaler_path: str) -> Tuple:
    """
    Load trained hourly pivot model from disk.
    
    Args:
        model_path: Path to saved model
        scaler_path: Path to saved scaler
        
    Returns:
        Tuple of (model, scaler)
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    return model, scaler


# Example usage
if __name__ == "__main__":
    # Import wave detection module
    from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
    
    # Fetch data
    from config import DEFAULT_SYMBOL as symbol
    data = fetch_market_data(symbol, interval='1h', period='60d')
    
    # Detect waves
    wave_data = detect_waves(data)
    wave_data = calculate_wave_metrics(wave_data, data)
    wave_data = calculate_pivot_confidence(wave_data, data)
    
    # Extract hourly features
    X, y = extract_hourly_features(data, wave_data, lookback=5)
    
    if len(X) > 0:
        print(f"Extracted {len(X)} samples with {X.shape[1]} features")
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model, scaler = train_hourly_pivot_model(X_train, y_train)
        
        # Evaluate model
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Predict next hourly pivot
        prediction = predict_next_hourly_pivot(model, scaler, wave_data, data)
        print(f"Next hourly pivot prediction: {prediction}")
        
        # Save model
        save_hourly_model(model, scaler)
    else:
        print("Not enough data to extract features")
