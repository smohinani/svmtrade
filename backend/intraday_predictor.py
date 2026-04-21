"""
Intraday High/Low Prediction Module

This module provides functionality for predicting next day's intraday high and low prices
using wave patterns and SVM models. It includes specialized feature engineering,
prediction confidence intervals, and historical pattern matching.

Dependencies: numpy, pandas, scikit-learn, scipy, statsmodels
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import statsmodels.api as sm
from typing import Dict, Tuple, List, Union, Optional
import datetime
import os
import joblib

# Import wave detection and SVM prediction modules
import sys
sys.path.append('.')
from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics
from svm_predictor import extract_wave_features


def extract_price_extreme_features(data: pd.DataFrame, wave_data: Dict, lookback_days: int = 10) -> pd.DataFrame:
    """
    Extract features specifically designed for predicting price extremes.
    
    Features include:
    - Recent high/low values
    - Daily ranges
    - Wave metrics
    - Day of week
    - Volatility measures
    - Price momentum
    
    Args:
        data: DataFrame with market data
        wave_data: Dictionary with wave detection results
        lookback_days: Number of previous days to include in features
        
    Returns:
        DataFrame with extracted features
    """
    # Resample to daily data if not already daily
    if 'D' not in str(data.index.freq):
        daily_data = data.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    else:
        daily_data = data.copy()
    
    # Calculate daily features
    features = pd.DataFrame(index=daily_data.index)
    
    # Basic price features
    features['DailyRange'] = daily_data['High'] - daily_data['Low']
    features['DailyReturn'] = daily_data['Close'].pct_change()
    features['GapUp'] = (daily_data['Open'] - daily_data['Close'].shift(1)) / daily_data['Close'].shift(1)
    
    # Volatility features
    features['VolatilityHigh'] = daily_data['High'].rolling(window=lookback_days).std()
    features['VolatilityLow'] = daily_data['Low'].rolling(window=lookback_days).std()
    features['VolatilityClose'] = daily_data['Close'].rolling(window=lookback_days).std()
    
    # Range features
    features['AvgRange'] = features['DailyRange'].rolling(window=lookback_days).mean()
    features['RangeRatio'] = features['DailyRange'] / features['AvgRange']
    
    # Momentum features
    features['HighMomentum'] = daily_data['High'].pct_change(periods=lookback_days)
    features['LowMomentum'] = daily_data['Low'].pct_change(periods=lookback_days)
    
    # Previous day's high/low relative to close
    features['PrevHighToClose'] = (daily_data['High'].shift(1) - daily_data['Close'].shift(1)) / daily_data['Close'].shift(1)
    features['PrevLowToClose'] = (daily_data['Low'].shift(1) - daily_data['Close'].shift(1)) / daily_data['Close'].shift(1)
    
    # Time-based features
    features['DayOfWeek'] = daily_data.index.dayofweek
    features['DayOfMonth'] = daily_data.index.day
    features['MonthOfYear'] = daily_data.index.month
    
    # One-hot encode day of week
    for i in range(5):  # 0-4 for Monday-Friday
        features[f'DayOfWeek_{i}'] = (features['DayOfWeek'] == i).astype(int)
    
    # Wave-based features
    if wave_data and len(wave_data['all_pivot_indices']) > 0:
        # Count recent peaks and troughs
        recent_pivots = pd.Series(
            wave_data['pivot_types'],
            index=pd.to_datetime(wave_data['all_pivot_dates'])
        )
        
        # For each day, count peaks and troughs in the previous N days
        for day_idx in range(len(daily_data)):
            day = daily_data.index[day_idx]
            start_date = day - pd.Timedelta(days=lookback_days)
            
            # Filter pivots in the lookback period
            period_pivots = recent_pivots[(recent_pivots.index >= start_date) & (recent_pivots.index < day)]
            
            if len(period_pivots) > 0:
                features.loc[day, 'RecentPeakCount'] = sum(period_pivots == 1)
                features.loc[day, 'RecentTroughCount'] = sum(period_pivots == -1)
                
                # Last pivot type and confidence
                last_pivot_idx = period_pivots.index[-1]
                last_pivot_type = period_pivots.iloc[-1]
                
                features.loc[day, 'LastPivotType'] = last_pivot_type
                
                # Find confidence score for this pivot
                pivot_date_idx = np.where(wave_data['all_pivot_dates'] == np.datetime64(last_pivot_idx))[0]
                if len(pivot_date_idx) > 0 and 'confidence_scores' in wave_data:
                    features.loc[day, 'LastPivotConfidence'] = wave_data['confidence_scores'][pivot_date_idx[0]]
                else:
                    features.loc[day, 'LastPivotConfidence'] = 0.5
            else:
                features.loc[day, 'RecentPeakCount'] = 0
                features.loc[day, 'RecentTroughCount'] = 0
                features.loc[day, 'LastPivotType'] = 0
                features.loc[day, 'LastPivotConfidence'] = 0.5
    
    # Fill NaN values
    features = features.fillna(0)
    
    # Add target variables (next day's high and low)
    features['NextDayHigh'] = daily_data['High'].shift(-1)
    features['NextDayLow'] = daily_data['Low'].shift(-1)
    
    # Remove rows with NaN targets (last day)
    features = features.dropna(subset=['NextDayHigh', 'NextDayLow'])
    
    return features


def find_similar_patterns(features: pd.DataFrame, current_features: pd.Series, 
                         n_matches: int = 5, lookback_window: int = 60) -> pd.DataFrame:
    """
    Find historical patterns similar to the current market conditions.
    
    Args:
        features: DataFrame with historical features
        current_features: Series with current feature values
        n_matches: Number of similar patterns to find
        lookback_window: How far back to look for patterns
        
    Returns:
        DataFrame with similar patterns
    """
    # Select features to use for similarity matching
    similarity_features = [
        'DailyRange', 'DailyReturn', 'GapUp', 
        'VolatilityClose', 'AvgRange', 'RangeRatio',
        'HighMomentum', 'LowMomentum',
        'PrevHighToClose', 'PrevLowToClose',
        'LastPivotType', 'LastPivotConfidence'
    ]
    
    # Ensure all required features exist
    valid_features = [f for f in similarity_features if f in features.columns and f in current_features.index]
    
    if not valid_features:
        return pd.DataFrame()
    
    # Normalize features for distance calculation
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features[valid_features]),
        index=features.index,
        columns=valid_features
    )
    
    # Scale current features
    current_scaled = pd.Series(
        scaler.transform(current_features[valid_features].values.reshape(1, -1))[0],
        index=valid_features
    )
    
    # Calculate Euclidean distance for each historical point
    distances = pd.Series(index=features_scaled.index)
    for idx in features_scaled.index:
        distances[idx] = np.sqrt(((features_scaled.loc[idx] - current_scaled) ** 2).sum())
    
    # Get indices of n_matches most similar patterns
    similar_indices = distances.sort_values().head(n_matches).index
    
    # Return similar patterns with their targets
    similar_patterns = features.loc[similar_indices, valid_features + ['NextDayHigh', 'NextDayLow']]
    similar_patterns['Distance'] = distances[similar_indices]
    
    return similar_patterns


def train_extreme_prediction_models(features: pd.DataFrame) -> Dict:
    """
    Train specialized SVR models for predicting next day's high and low.
    
    Args:
        features: DataFrame with extracted features
        
    Returns:
        Dictionary with trained models and scalers
    """
    # Define feature columns to use
    feature_cols = [col for col in features.columns if col not in ['NextDayHigh', 'NextDayLow']]
    
    # Prepare data
    X = features[feature_cols].values
    y_high = features['NextDayHigh'].values
    y_low = features['NextDayLow'].values
    
    # Create scalers
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    
    y_high_scaler = StandardScaler()
    y_high_scaled = y_high_scaler.fit_transform(y_high.reshape(-1, 1)).flatten()
    
    y_low_scaler = StandardScaler()
    y_low_scaled = y_low_scaler.fit_transform(y_low.reshape(-1, 1)).flatten()
    
    # Create time series split for financial data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
        'epsilon': [0.01, 0.05, 0.1, 0.2]
    }
    
    # Train high model with grid search
    svr_high = SVR(kernel='rbf')
    grid_search_high = GridSearchCV(
        estimator=svr_high,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search_high.fit(X_scaled, y_high_scaled)
    
    # Train low model with grid search
    svr_low = SVR(kernel='rbf')
    grid_search_low = GridSearchCV(
        estimator=svr_low,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search_low.fit(X_scaled, y_low_scaled)
    
    # Get best models
    best_high_model = grid_search_high.best_estimator_
    best_low_model = grid_search_low.best_estimator_
    
    print(f"Best high model parameters: {grid_search_high.best_params_}")
    print(f"Best low model parameters: {grid_search_low.best_params_}")
    
    return {
        'high_model': best_high_model,
        'low_model': best_low_model,
        'X_scaler': X_scaler,
        'y_high_scaler': y_high_scaler,
        'y_low_scaler': y_low_scaler,
        'feature_cols': feature_cols
    }


def evaluate_extreme_prediction_models(models: Dict, features: pd.DataFrame) -> Dict:
    """
    Evaluate performance of high/low prediction models.
    
    Args:
        models: Dictionary with trained models and scalers
        features: DataFrame with features and targets
        
    Returns:
        Dictionary with performance metrics
    """
    # Extract models and scalers
    high_model = models['high_model']
    low_model = models['low_model']
    X_scaler = models['X_scaler']
    y_high_scaler = models['y_high_scaler']
    y_low_scaler = models['y_low_scaler']
    feature_cols = models['feature_cols']
    
    # Prepare data
    X = features[feature_cols].values
    y_high = features['NextDayHigh'].values
    y_low = features['NextDayLow'].values
    
    # Scale data
    X_scaled = X_scaler.transform(X)
    
    # Make predictions
    y_high_pred_scaled = high_model.predict(X_scaled)
    y_low_pred_scaled = low_model.predict(X_scaled)
    
    # Inverse transform predictions
    y_high_pred = y_high_scaler.inverse_transform(y_high_pred_scaled.reshape(-1, 1)).flatten()
    y_low_pred = y_low_scaler.inverse_transform(y_low_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    high_mse = mean_squared_error(y_high, y_high_pred)
    high_mae = mean_absolute_error(y_high, y_high_pred)
    high_r2 = r2_score(y_high, y_high_pred)
    
    low_mse = mean_squared_error(y_low, y_low_pred)
    low_mae = mean_absolute_error(y_low, y_low_pred)
    low_r2 = r2_score(y_low, y_low_pred)
    
    # Calculate percentage errors
    high_mape = np.mean(np.abs((y_high - y_high_pred) / y_high)) * 100
    low_mape = np.mean(np.abs((y_low - y_low_pred) / y_low)) * 100
    
    return {
        'high_mse': high_mse,
        'high_mae': high_mae,
        'high_r2': high_r2,
        'high_mape': high_mape,
        'low_mse': low_mse,
        'low_mae': low_mae,
        'low_r2': low_r2,
        'low_mape': low_mape
    }


def predict_next_day_extremes(models: Dict, data: pd.DataFrame, wave_data: Dict) -> Dict:
    from numpy import percentile

    # Extract models and scalers
    high_model = models['high_model']
    low_model = models['low_model']
    X_scaler = models['X_scaler']
    y_high_scaler = models['y_high_scaler']
    y_low_scaler = models['y_low_scaler']
    feature_cols = models['feature_cols']
    
    features = extract_price_extreme_features(data, wave_data)
    if features.empty:
        return {
            'high_prediction': None,
            'low_prediction': None,
            'confidence': 0.0
        }
    
    current_features = features.iloc[-1]
    for col in feature_cols:
        if col not in current_features:
            current_features[col] = 0

    X = current_features[feature_cols].values.reshape(1, -1)
    X_scaled = X_scaler.transform(X)

    y_high_pred_scaled = high_model.predict(X_scaled)
    y_low_pred_scaled = low_model.predict(X_scaled)
    y_high_pred = y_high_scaler.inverse_transform(y_high_pred_scaled.reshape(-1, 1)).flatten()[0]
    y_low_pred = y_low_scaler.inverse_transform(y_low_pred_scaled.reshape(-1, 1)).flatten()[0]

    similar_patterns = find_similar_patterns(features[:-1], current_features)

    if not similar_patterns.empty:
        high_resid = similar_patterns['NextDayHigh'] - y_high_pred
        low_resid = similar_patterns['NextDayLow'] - y_low_pred

        # Percentile-based confidence intervals (tighter than std dev)
        high_ci_lower = y_high_pred + percentile(high_resid, 10)
        high_ci_upper = y_high_pred + percentile(high_resid, 90)
        low_ci_lower = y_low_pred + percentile(low_resid, 10)
        low_ci_upper = y_low_pred + percentile(low_resid, 90)

        weights = 1 / (similar_patterns['Distance'] + 0.001)
        weights /= weights.sum()
        weighted_similar_high = (similar_patterns['NextDayHigh'] * weights).sum()
        weighted_similar_low = (similar_patterns['NextDayLow'] * weights).sum()

        blended_high_pred = 0.7 * y_high_pred + 0.3 * weighted_similar_high
        blended_low_pred = 0.7 * y_low_pred + 0.3 * weighted_similar_low

        high_iqr = percentile(high_resid, 90) - percentile(high_resid, 10)
        low_iqr = percentile(low_resid, 90) - percentile(low_resid, 10)

        high_conf = 1.0 - (high_iqr / blended_high_pred)
        low_conf = 1.0 - (low_iqr / blended_low_pred)
        confidence = max(0.0, min(1.0, (high_conf + low_conf) / 2))

    else:
        # Conservative fallback
        high_ci_lower = y_high_pred * 0.995
        high_ci_upper = y_high_pred * 1.005
        low_ci_lower = y_low_pred * 0.995
        low_ci_upper = y_low_pred * 1.005
        blended_high_pred = y_high_pred
        blended_low_pred = y_low_pred
        confidence = 0.6

    return {
        'high_prediction': blended_high_pred,
        'high_ci_lower': high_ci_lower,
        'high_ci_upper': high_ci_upper,
        'low_prediction': blended_low_pred,
        'low_ci_lower': low_ci_lower,
        'low_ci_upper': low_ci_upper,
        'confidence': confidence,
        'similar_patterns_count': len(similar_patterns)
    }



def save_extreme_models(models: Dict, model_dir: str = './models'):
    """
    Save trained high/low prediction models to disk.
    
    Args:
        models: Dictionary with trained models and scalers
        model_dir: Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save models and scalers
    high_model_path = os.path.join(model_dir, f'high_model_{timestamp}.joblib')
    low_model_path = os.path.join(model_dir, f'low_model_{timestamp}.joblib')
    X_scaler_path = os.path.join(model_dir, f'X_scaler_{timestamp}.joblib')
    y_high_scaler_path = os.path.join(model_dir, f'y_high_scaler_{timestamp}.joblib')
    y_low_scaler_path = os.path.join(model_dir, f'y_low_scaler_{timestamp}.joblib')
    
    joblib.dump(models['high_model'], high_model_path)
    joblib.dump(models['low_model'], low_model_path)
    joblib.dump(models['X_scaler'], X_scaler_path)
    joblib.dump(models['y_high_scaler'], y_high_scaler_path)
    joblib.dump(models['y_low_scaler'], y_low_scaler_path)
    
    # Save feature columns
    feature_cols_path = os.path.join(model_dir, f'feature_cols_{timestamp}.joblib')
    joblib.dump(models['feature_cols'], feature_cols_path)
    
    print(f"High model saved to {high_model_path}")
    print(f"Low model saved to {low_model_path}")
    print(f"Feature columns saved to {feature_cols_path}")
    
    return {
        'high_model_path': high_model_path,
        'low_model_path': low_model_path,
        'X_scaler_path': X_scaler_path,
        'y_high_scaler_path': y_high_scaler_path,
        'y_low_scaler_path': y_low_scaler_path,
        'feature_cols_path': feature_cols_path
    }


def load_extreme_models(model_paths: Dict) -> Dict:
    """
    Load trained high/low prediction models from disk.
    
    Args:
        model_paths: Dictionary with paths to saved models and scalers
        
    Returns:
        Dictionary with loaded models and scalers
    """
    high_model = joblib.load(model_paths['high_model_path'])
    low_model = joblib.load(model_paths['low_model_path'])
    X_scaler = joblib.load(model_paths['X_scaler_path'])
    y_high_scaler = joblib.load(model_paths['y_high_scaler_path'])
    y_low_scaler = joblib.load(model_paths['y_low_scaler_path'])
    feature_cols = joblib.load(model_paths['feature_cols_path'])
    
    return {
        'high_model': high_model,
        'low_model': low_model,
        'X_scaler': X_scaler,
        'y_high_scaler': y_high_scaler,
        'y_low_scaler': y_low_scaler,
        'feature_cols': feature_cols
    }


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
    
    # Extract features for high/low prediction
    features = extract_price_extreme_features(data, wave_data)
    
    if not features.empty:
        print(f"Extracted {len(features)} samples with {len(features.columns)} features")
        
        # Train models
        models = train_extreme_prediction_models(features)
        
        # Evaluate models
        metrics = evaluate_extreme_prediction_models(models, features)
        print(f"Model performance: {metrics}")
        
        # Predict next day's high and low
        predictions = predict_next_day_extremes(models, data, wave_data)
        print(f"Next day predictions: {predictions}")
        
        # Save models
        save_extreme_models(models)
    else:
        print("Not enough data to extract features")
