"""
Enhanced SVM Prediction Module

This module provides advanced functionality for predicting market movements
using Support Vector Machines (SVM) trained on wave pattern features.
It includes hyperparameter optimization, probability calibration,
ensemble approaches, and cross-validation.

Dependencies: numpy, pandas, scikit-learn, joblib
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from typing import Dict, Tuple, List, Union, Optional
import os
import datetime

# Import wave detection module
import sys
sys.path.append('.')
from wave_detector import detect_waves, calculate_wave_metrics, calculate_pivot_confidence


def extract_wave_features(wave_data: Dict, lookback: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from wave data for SVM training.
    
    Features include:
    - Wave heights
    - Wave types (peak/trough)
    - Wave durations
    - Wave slopes
    - Confidence scores
    
    Args:
        wave_data: Dictionary with wave detection results
        lookback: Number of previous pivot points to include in features
        
    Returns:
        Tuple of (features, targets) arrays
    """
    all_pivot_indices = wave_data['all_pivot_indices']
    pivot_types = wave_data['pivot_types']
    wave_heights = wave_data['wave_heights']
    
    # Get additional metrics if available
    wave_durations = wave_data.get('wave_durations', np.zeros_like(wave_heights))
    wave_slopes = wave_data.get('wave_slopes', np.zeros_like(wave_heights))
    confidence_scores = wave_data.get('confidence_scores', np.zeros_like(wave_heights))
    
    # We need at least lookback+1 pivot points to create a sample
    if len(all_pivot_indices) <= lookback:
        return np.array([]), np.array([])
    
    # Create features and target lists
    features = []
    targets = []
    
    # For each pivot point after the lookback period
    for i in range(lookback, len(all_pivot_indices) - 1):
        # Extract features from previous waves
        feature_vector = []
        
        # Add features for each previous wave in the lookback window
        for j in range(i - lookback, i + 1):  # Include current pivot
            # Add wave height (normalized by the mean height)
            mean_height = np.mean(wave_heights[1:i+1]) if i > 0 else 1.0
            if mean_height > 0:
                norm_height = wave_heights[j] / mean_height
            else:
                norm_height = wave_heights[j]
            feature_vector.append(float(norm_height))
            
            # Add wave type (1 for peak, -1 for trough)
            feature_vector.append(float(pivot_types[j]))
            
            # Add wave duration (if available)
            if j > 0:
                # Normalize duration by mean duration
                mean_duration = np.mean(wave_durations[1:i+1]) if i > 0 else 1.0
                if mean_duration > 0:
                    norm_duration = wave_durations[j] / mean_duration
                else:
                    norm_duration = wave_durations[j]
                feature_vector.append(float(norm_duration))
            else:
                feature_vector.append(0.0)
            
            # Add wave slope (if available)
            if j > 0:
                feature_vector.append(float(wave_slopes[j]))
            else:
                feature_vector.append(0.0)
            
            # Add confidence score (if available)
            feature_vector.append(float(confidence_scores[j]))
        
        # Target is the next pivot type (1 for peak, -1 for trough)
        target = float(pivot_types[i+1])
        
        features.append(feature_vector)
        targets.append(target)
    
    return np.array(features), np.array(targets)


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
    """
    Find optimal SVM hyperparameters using grid search.
    
    Args:
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with best parameters
    """
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Create time series split for financial data
    tscv = TimeSeriesSplit(n_splits=cv)
    
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
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


def train_calibrated_svm(X, y, random_state=42):
    """
    Train a calibrated SVM model for pivot prediction.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random state for reproducibility
        
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
    svm = SVC(probability=True, random_state=random_state)
    
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


def train_ensemble_svm(X: np.ndarray, y: np.ndarray) -> Tuple[VotingClassifier, StandardScaler]:
    """
    Train an ensemble of SVM models with different kernels.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuple of (ensemble model, scaler)
    """
    # Create scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create SVM models with different kernels
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm_poly = SVC(kernel='poly', C=1.0, gamma='scale', probability=True)
    svm_sigmoid = SVC(kernel='sigmoid', C=1.0, gamma='scale', probability=True)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rbf', svm_rbf),
            ('poly', svm_poly),
            ('sigmoid', svm_sigmoid)
        ],
        voting='soft'  # Use probability estimates
    )
    
    # Train ensemble
    ensemble.fit(X_scaled, y)
    
    return ensemble, scaler


def evaluate_model(model, X: np.ndarray, y: np.ndarray, scaler: StandardScaler) -> Dict:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X: Test feature matrix
        y: Test target vector
        scaler: Feature scaler
        
    Returns:
        Dictionary with performance metrics
    """
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    # Calculate confidence
    confidence = np.max(y_prob, axis=1).mean()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confidence': confidence
    }


def predict_next_pivot(model, scaler: StandardScaler, wave_data: Dict, lookback: int = 5) -> Dict:
    """
    Predict the next pivot point using the trained SVM model.
    
    Args:
        model: Trained SVM model
        scaler: Feature scaler
        wave_data: Dictionary with wave detection results
        lookback: Number of previous pivot points to include
        
    Returns:
        Dictionary with prediction results
    """
    all_pivot_indices = wave_data['all_pivot_indices']
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
            'method': 'svm_insufficient_data'
        }
    
    # Extract features from the most recent waves
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
    
    # Scale features
    X = np.array([feature_vector])
    
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
    last_pivot_idx = -1
    last_pivot_value = wave_data['all_pivot_values'][last_pivot_idx]
    last_pivot_index = wave_data['all_pivot_indices'][last_pivot_idx]
    
    # Calculate average wave metrics for estimation
    if len(wave_heights) > 1 and len(wave_durations) > 1:
        # Use recent waves for better estimation (last 5 or fewer)
        recent_count = min(5, len(wave_heights) - 1)
        avg_height = np.mean(wave_heights[-recent_count:])
        avg_duration = np.mean(wave_durations[-recent_count:])
    else:
        avg_height = 0.01 * last_pivot_value  # Default 1% move
        avg_duration = 5  # Default if not enough data
    
    # Estimate next pivot value
    if prediction == 1:  # Peak
        estimated_value = last_pivot_value + avg_height
    else:  # Trough
        estimated_value = last_pivot_value - avg_height
    
    # Estimate next pivot timing
    estimated_index = last_pivot_index + avg_duration
    
    return {
        'predicted_type': prediction,
        'predicted_type_name': pred_type_name,
        'confidence': probability,
        'estimated_value': estimated_value,
        'estimated_index_offset': avg_duration,
        'method': 'svm_prediction'
    }


def save_model(model, scaler: StandardScaler, model_dir: str = './models'):
    """
    Save trained model and scaler to disk.
    
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
    model_path = os.path.join(model_dir, f'svm_model_{timestamp}.joblib')
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(model_dir, f'scaler_{timestamp}.joblib')
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    return model_path, scaler_path


def load_model(model_path: str, scaler_path: str) -> Tuple:
    """
    Load trained model and scaler from disk.
    
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
    
    # Extract features
    X, y = extract_wave_features(wave_data, lookback=5)
    
    if len(X) > 0:
        print(f"Extracted {len(X)} samples with {X.shape[1]} features")
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model, scaler = train_calibrated_svm(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, scaler)
        print(f"Model performance: {metrics}")
        
        # Predict next pivot
        prediction = predict_next_pivot(model, scaler, wave_data)
        print(f"Next pivot prediction: {prediction}")
        
        # Save model
        save_model(model, scaler)
    else:
        print("Not enough data to extract features")
