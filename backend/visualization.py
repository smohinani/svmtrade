"""
Visualization and Testing Framework

This module provides comprehensive visualization and backtesting functionality
for the enhanced SVM-based trading system. It includes wave analysis visualization,
prediction visualization, performance metrics, and interactive charts.

Dependencies: numpy, pandas, matplotlib, seaborn, plotly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Tuple, List, Union, Optional
import os
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import system modules
import sys
sys.path.append('.')
from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
from svm_predictor import extract_wave_features, predict_next_pivot
from intraday_predictor import predict_next_day_extremes
from hourly_pivot_predictor import predict_next_hourly_pivot


def clean_yf_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans Yahoo Finance data by flattening MultiIndex, capitalizing columns,
    validating required OHLCV fields, and ensuring the datetime index is tz–naive.
    """
    if raw_data.empty:
        raise ValueError("Downloaded data is empty.")
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.swaplevel(0, 1)
        raw_data = raw_data.droplevel(0, axis=1)

    raw_data.columns = [col.capitalize() for col in raw_data.columns]

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in raw_data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Ensure the datetime index is timezone-naive
    if raw_data.index.tz is not None:
        raw_data.index = raw_data.index.tz_localize(None)
    
    return raw_data


def plot_wave_analysis(data: pd.DataFrame, wave_data: Dict, title: str = "Wave Analysis", 
                      save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
    """
    Create a visualization of detected waves with peaks and troughs.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot close price
    ax.plot(data.index, data['Close'], label='Close', color='blue', alpha=0.7)
    
    # Plot smoothed price if available
    if 'smooth_values' in wave_data:
        ax.plot(data.index, wave_data['smooth_values'], label='Smoothed', color='blue', linestyle='--')
    
    # Plot peaks
    peaks_indices = wave_data['peaks_indices']
    if len(peaks_indices) > 0:
        ax.scatter(
            data.index[peaks_indices], 
            data['Close'].iloc[peaks_indices], 
            color='green', 
            marker='^', 
            s=100, 
            label='Peaks'
        )
    
    # Plot troughs
    troughs_indices = wave_data['troughs_indices']
    if len(troughs_indices) > 0:
        ax.scatter(
            data.index[troughs_indices], 
            data['Close'].iloc[troughs_indices], 
            color='red', 
            marker='v', 
            s=100, 
            label='Troughs'
        )
    
    # Add confidence scores if available
    if 'confidence_scores' in wave_data and len(wave_data['all_pivot_indices']) > 0:
        all_pivot_indices = wave_data['all_pivot_indices']
        confidence_scores = wave_data['confidence_scores']
        pivot_types = wave_data['pivot_types']
        
        for i, idx in enumerate(all_pivot_indices):
            if idx < len(data):
                confidence = confidence_scores[i]
                pivot_type = "Peak" if pivot_types[i] == 1 else "Trough"
                ax.annotate(
                    f"{confidence:.2f}",
                    xy=(data.index[idx], data['Close'].iloc[idx]),
                    xytext=(0, 10 if pivot_types[i] == 1 else -20),
                    textcoords="offset points",
                    ha='center',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                )
    
    # Format plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # Add order parameter used
    if 'order_used' in wave_data:
        ax.text(
            0.02, 0.02, 
            f"Order parameter: {wave_data['order_used']}", 
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_predictions(data: pd.DataFrame, wave_data: Dict, pivot_prediction: Dict,
                    intraday_prediction: Dict = None, hourly_prediction: Dict = None,
                    title: str = "Market Predictions", save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (14, 10)):
    """
    Create a visualization of market predictions including next pivot, intraday high/low,
    and hourly pivot predictions.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    
    # Wave analysis subplot
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot close price
    ax1.plot(data.index, data['Close'], label='Close', color='blue', alpha=0.7)
    
    # Plot peaks and troughs
    peaks_indices = wave_data['peaks_indices']
    troughs_indices = wave_data['troughs_indices']
    
    if len(peaks_indices) > 0:
        ax1.scatter(
            data.index[peaks_indices], 
            data['Close'].iloc[peaks_indices], 
            color='green', 
            marker='^', 
            s=80, 
            label='Peaks'
        )
    
    if len(troughs_indices) > 0:
        ax1.scatter(
            data.index[troughs_indices], 
            data['Close'].iloc[troughs_indices], 
            color='red', 
            marker='v', 
            s=80, 
            label='Troughs'
        )
    
    # Add next pivot prediction
    if pivot_prediction and pivot_prediction.get('predicted_type') is not None:
        last_date = data.index[-1]
        if 'estimated_index_offset' in pivot_prediction:
            offset_hours = pivot_prediction['estimated_index_offset']
            next_date = last_date + pd.Timedelta(hours=offset_hours)
        else:
            next_date = last_date + pd.Timedelta(days=1)
        
        pred_value = pivot_prediction.get('estimated_value', data['Close'].iloc[-1])
        pred_type = pivot_prediction.get('predicted_type')
        pred_type_name = pivot_prediction.get('predicted_type_name', 'Unknown')
        confidence = pivot_prediction.get('confidence', 0.5)
        
        marker = '^' if pred_type == 1 else 'v'
        color = 'green' if pred_type == 1 else 'red'
        
        ax1.scatter(
            [next_date], 
            [pred_value], 
            color=color, 
            marker=marker, 
            s=120, 
            alpha=0.7,
            edgecolor='black',
            label=f'Predicted {pred_type_name}'
        )
        
        ax1.annotate(
            f"{pred_type_name}\nConf: {confidence:.2f}",
            xy=(next_date, pred_value),
            xytext=(10, 0),
            textcoords="offset points",
            ha='left',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    ax1.set_title("Wave Analysis with Next Pivot Prediction", fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Intraday high/low prediction subplot
    ax2 = fig.add_subplot(gs[1, 0])
    recent_days = min(30, len(data))
    ax2.plot(data.index[-recent_days:], data['Close'].iloc[-recent_days:], label='Close', color='blue', alpha=0.7)
    
    for i in range(1, recent_days):
        idx = -recent_days + i
        ax2.plot(
            [data.index[idx], data.index[idx]],
            [data['Low'].iloc[idx], data['High'].iloc[idx]],
            color='gray',
            alpha=0.5,
            linewidth=3
        )
    
    if intraday_prediction and intraday_prediction.get('high_prediction') is not None:
        last_date = data.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        high_pred = intraday_prediction.get('high_prediction')
        low_pred = intraday_prediction.get('low_prediction')
        confidence = intraday_prediction.get('confidence', 0.5)
        
        ax2.plot(
            [next_date, next_date],
            [low_pred, high_pred],
            color='orange',
            alpha=0.8,
            linewidth=5,
            label='Predicted Range'
        )
        
        if 'high_ci_upper' in intraday_prediction and 'low_ci_lower' in intraday_prediction:
            high_ci_upper = intraday_prediction['high_ci_upper']
            high_ci_lower = intraday_prediction['high_ci_lower']
            low_ci_upper = intraday_prediction['low_ci_upper']
            low_ci_lower = intraday_prediction['low_ci_lower']
            ax2.fill_betweenx(
                [low_ci_lower, high_ci_upper],
                next_date - pd.Timedelta(hours=6),
                next_date + pd.Timedelta(hours=6),
                alpha=0.2,
                color='orange'
            )
        
        ax2.annotate(
            f"High: {high_pred:.2f}\nLow: {low_pred:.2f}\nConf: {confidence:.2f}",
            xy=(next_date, (high_pred + low_pred) / 2),
            xytext=(10, 0),
            textcoords="offset points",
            ha='left',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    ax2.set_title("Next Day High/Low Prediction", fontsize=14)
    ax2.set_xlabel('')
    ax2.set_ylabel('Price', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Hourly pivot prediction subplot
    ax3 = fig.add_subplot(gs[1, 1])
    recent_hours = min(100, len(data))
    ax3.plot(data.index[-recent_hours:], data['Close'].iloc[-recent_hours:], label='Close', color='blue', alpha=0.7)
    
    recent_peaks = [i for i in peaks_indices if i >= len(data) - recent_hours]
    recent_troughs = [i for i in troughs_indices if i >= len(data) - recent_hours]
    
    if recent_peaks:
        ax3.scatter(
            data.index[recent_peaks], 
            data['Close'].iloc[recent_peaks], 
            color='green', 
            marker='^', 
            s=80, 
            label='Peaks'
        )
    
    if recent_troughs:
        ax3.scatter(
            data.index[recent_troughs], 
            data['Close'].iloc[recent_troughs], 
            color='red', 
            marker='v', 
            s=80, 
            label='Troughs'
        )
    
    if hourly_prediction and hourly_prediction.get('predicted_type') is not None:
        timing = hourly_prediction.get('timing', {})
        estimated_datetime = timing.get('estimated_datetime')
        if estimated_datetime is None:
            last_date = data.index[-1]
            estimated_hours = timing.get('estimated_hours', 4)
            estimated_datetime = last_date + pd.Timedelta(hours=estimated_hours)
        
        pred_value = hourly_prediction.get('estimated_value', data['Close'].iloc[-1])
        pred_type = hourly_prediction.get('predicted_type')
        pred_type_name = hourly_prediction.get('predicted_type_name', 'Unknown')
        confidence = hourly_prediction.get('confidence', 0.5)
        
        marker = '^' if pred_type == 1 else 'v'
        color = 'green' if pred_type == 1 else 'red'
        
        ax3.scatter(
            [estimated_datetime], 
            [pred_value], 
            color=color, 
            marker=marker, 
            s=120, 
            alpha=0.7,
            edgecolor='black',
            label=f'Predicted {pred_type_name}'
        )
        
        ax3.annotate(
            f"{pred_type_name}\nTime: {estimated_datetime.strftime('%H:%M')}\nConf: {confidence:.2f}",
            xy=(estimated_datetime, pred_value),
            xytext=(10, 0),
            textcoords="offset points",
            ha='left',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    ax3.set_title("Hourly Pivot Prediction", fontsize=14)
    ax3.set_xlabel('')
    ax3.set_ylabel('Price', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # Trading strategy subplot
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(data.index[-recent_days:], data['Close'].iloc[-recent_days:], label='Close', color='blue', alpha=0.7)
    
    signal = 0
    entry_price = data['Close'].iloc[-1]
    stop_loss = entry_price
    take_profit = entry_price
    
    if pivot_prediction and pivot_prediction.get('predicted_type') is not None:
        pred_type = pivot_prediction.get('predicted_type')
        confidence = pivot_prediction.get('confidence', 0.5)
        if confidence >= 0.7:
            signal = -1 if pred_type == 1 else 1
    if intraday_prediction and signal != 0:
        high_pred = intraday_prediction.get('high_prediction')
        low_pred = intraday_prediction.get('low_prediction')
        if high_pred is not None and low_pred is not None:
            if signal == 1:
                stop_loss = low_pred * 0.99
                take_profit = high_pred * 0.99
            else:
                stop_loss = high_pred * 1.01
                take_profit = low_pred * 1.01
    
    last_date = data.index[-1]
    next_date = last_date + pd.Timedelta(days=1)
    
    if signal != 0:
        marker = '^' if signal == 1 else 'v'
        color = 'green' if signal == 1 else 'red'
        signal_name = 'BUY' if signal == 1 else 'SELL'
        
        ax4.scatter(
            [next_date], 
            [entry_price], 
            color=color, 
            marker=marker, 
            s=120, 
            alpha=0.7,
            edgecolor='black',
            label=f'{signal_name} Signal'
        )
        
        ax4.axhline(y=stop_loss, color='red', linestyle='--', alpha=0.7, label='Stop Loss')
        ax4.axhline(y=take_profit, color='green', linestyle='--', alpha=0.7, label='Take Profit')
        
        ax4.annotate(
            f"Signal: {signal_name}\nEntry: {entry_price:.2f}\nStop: {stop_loss:.2f}\nTarget: {take_profit:.2f}",
            xy=(next_date, entry_price),
            xytext=(10, 0),
            textcoords="offset points",
            ha='left',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    else:
        ax4.text(
            0.5, 0.5, 
            "No Trading Signal", 
            transform=ax4.transAxes,
            ha='center',
            va='center',
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    ax4.set_title("Trading Strategy", fontsize=14)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Price', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    # Performance metrics subplot
    ax5 = fig.add_subplot(gs[2, 1])
    metrics = [
        ('Win Rate', '0.0%'),
        ('Avg Win', '$0.00'),
        ('Avg Loss', '$0.00'),
        ('Profit Factor', '0.00'),
        ('Total Return', '0.0%')
    ]
    ax5.axis('off')
    table = ax5.table(
        cellText=[[m[0], m[1]] for m in metrics],
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    ax5.set_title("Performance Metrics (Backtest Required)", fontsize=14)
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def backtest_strategy(data: pd.DataFrame, wave_data: Dict, svm_model, svm_scaler,
                      lookback: int = 5, confidence_threshold: float = 0.7,
                      stop_loss_pct: float = 1.0, take_profit_pct: float = 2.0) -> Dict:
    from svm_predictor import predict_next_pivot
    
    trades = []
    all_pivot_indices = wave_data['all_pivot_indices']
    all_pivot_dates = pd.to_datetime(wave_data['all_pivot_dates'])
    pivot_types = wave_data['pivot_types']
    
    if len(all_pivot_indices) <= lookback:
        return {
            'trades': [],
            'backtest_data': data,
            'metrics': {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_return': 0.0
            }
        }
    
    backtest_data = data.copy()
    backtest_data['Position'] = 0
    backtest_data['Entry'] = 0.0
    backtest_data['StopLoss'] = 0.0
    backtest_data['TakeProfit'] = 0.0
    backtest_data['Signal'] = 0

    i = lookback
    n_pivots = len(all_pivot_indices)
    
    while i < n_pivots - 1:
        pivot_idx = all_pivot_indices[i]
        pivot_date = all_pivot_dates[i]
        
        if pivot_idx >= len(backtest_data):
            i += 1
            continue
        
        subset_wave_data = {
            'all_pivot_indices': all_pivot_indices[:i+1],
            'all_pivot_dates': all_pivot_dates[:i+1],
            'pivot_types': pivot_types[:i+1],
            'all_pivot_values': wave_data.get('all_pivot_values', data['Close'].iloc[all_pivot_indices].tolist())[:i+1],
            'wave_heights': wave_data['wave_heights'][:i+1],
            'wave_durations': wave_data.get('wave_durations', np.zeros_like(wave_data['wave_heights']))[:i+1],
            'wave_slopes': wave_data.get('wave_slopes', np.zeros_like(wave_data['wave_heights']))[:i+1],
            'confidence_scores': wave_data.get('confidence_scores', np.zeros_like(wave_data['wave_heights']))[:i+1]
        }

        
        prediction = predict_next_pivot(svm_model, svm_scaler, subset_wave_data, lookback)
        
        if prediction and prediction.get('predicted_type') is not None:
            pred_type = prediction['predicted_type']
            confidence = prediction.get('confidence', 0.0)
            
            if confidence >= confidence_threshold:
                signal = -1 if pred_type == 1 else 1
                entry_price = backtest_data.loc[pivot_date, 'Close']
                if signal == 1:
                    stop_loss = entry_price * (1 - stop_loss_pct / 100)
                    take_profit = entry_price * (1 + take_profit_pct / 100)
                else:
                    stop_loss = entry_price * (1 + stop_loss_pct / 100)
                    take_profit = entry_price * (1 - take_profit_pct / 100)
                
                trade_exit_date = None
                trade_exit_price = None
                trade_rows = backtest_data.loc[pivot_date:].iloc[1:]
                for exit_date, row in trade_rows.iterrows():
                    if signal == 1:
                        if row['Low'] <= stop_loss:
                            trade_exit_date = exit_date
                            trade_exit_price = stop_loss
                            break
                        elif row['High'] >= take_profit:
                            trade_exit_date = exit_date
                            trade_exit_price = take_profit
                            break
                    else:
                        if row['High'] >= stop_loss:
                            trade_exit_date = exit_date
                            trade_exit_price = stop_loss
                            break
                        elif row['Low'] <= take_profit:
                            trade_exit_date = exit_date
                            trade_exit_price = take_profit
                            break
                if trade_exit_date is None:
                    trade_exit_date = backtest_data.index[-1]
                    trade_exit_price = backtest_data.iloc[-1]['Close']
                
                backtest_data.loc[pivot_date:trade_exit_date, 'Position'] = signal
                backtest_data.loc[pivot_date:trade_exit_date, 'Entry'] = entry_price
                backtest_data.loc[pivot_date:trade_exit_date, 'StopLoss'] = stop_loss
                backtest_data.loc[pivot_date:trade_exit_date, 'TakeProfit'] = take_profit
                
                if signal == 1:
                    profit_pct = (trade_exit_price - entry_price) / entry_price * 100
                else:
                    profit_pct = (entry_price - trade_exit_price) / entry_price * 100
                
                trades.append({
                    'entry_date': pivot_date,
                    'exit_date': trade_exit_date,
                    'entry_price': entry_price,
                    'exit_price': trade_exit_price,
                    'position': signal,
                    'profit_pct': profit_pct
                })
                
                while i < n_pivots and all_pivot_dates[i] <= trade_exit_date:
                    i += 1
                continue
        i += 1

    if trades:
        wins = [t for t in trades if t['profit_pct'] > 0]
        losses = [t for t in trades if t['profit_pct'] <= 0]
        win_rate = (len(wins) / len(trades)) * 100
        avg_win = np.mean([t['profit_pct'] for t in wins]) if wins else 0.0
        avg_loss = np.mean([t['profit_pct'] for t in losses]) if losses else 0.0
        total_return = np.sum([t['profit_pct'] for t in trades])
        sum_wins = np.sum([t['profit_pct'] for t in wins])
        sum_losses = np.sum([-t['profit_pct'] for t in losses])
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else np.inf
    else:
        win_rate = avg_win = avg_loss = total_return = profit_factor = 0.0

    return {
        'trades': trades,
        'backtest_data': backtest_data,
        'metrics': {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return
        }
    }


def plot_backtest_results(backtest_results: Dict, save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (14, 10)):
    """
    Plot backtest results including trades and performance metrics.
    """
    backtest_data = backtest_results['backtest_data']
    trades = backtest_results['trades']
    metrics = backtest_results['metrics']
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(backtest_data.index, backtest_data['Close'], label='Close', color='blue', alpha=0.7)
    
    for trade in trades:
        if 'entry_date' in trade and 'exit_date' in trade:
            entry_date = trade['entry_date']
            entry_price = trade['entry_price']
            exit_date = trade['exit_date']
            exit_price = trade['exit_price']
            position = trade['position']
            profit_pct = trade.get('profit_pct', 0.0)
            
            entry_color = 'green' if position == 1 else 'red'
            exit_color = 'green' if profit_pct > 0 else 'red'
            
            ax1.scatter(
                entry_date, 
                entry_price, 
                color=entry_color, 
                marker='^' if position == 1 else 'v', 
                s=100, 
                label='_nolegend_'
            )
            
            ax1.scatter(
                exit_date, 
                exit_price, 
                color=exit_color, 
                marker='o', 
                s=100, 
                label='_nolegend_'
            )
            
            ax1.plot(
                [entry_date, exit_date], 
                [entry_price, exit_price], 
                color=exit_color, 
                alpha=0.5, 
                linestyle='--', 
                label='_nolegend_'
            )
    
    ax1.set_title("Backtest Results: Price Chart with Trades", fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(
        backtest_data.index, 
        backtest_data['Position'], 
        0, 
        where=backtest_data['Position'] > 0, 
        color='green', 
        alpha=0.3, 
        label='Long'
    )
    ax2.fill_between(
        backtest_data.index, 
        backtest_data['Position'], 
        0, 
        where=backtest_data['Position'] < 0, 
        color='red', 
        alpha=0.3, 
        label='Short'
    )
    ax2.set_title("Position", fontsize=14)
    ax2.set_xlabel('')
    ax2.set_ylabel('Position', fontsize=12)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Flat', 'Long'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    ax3 = fig.add_subplot(gs[1, 1])
    equity = pd.Series(index=backtest_data.index, data=0.0)
    
    for trade in trades:
        if 'entry_date' in trade and 'exit_date' in trade:
            exit_date = trade['exit_date']
            profit_pct = trade.get('profit_pct', 0.0)
            equity.loc[exit_date:] += profit_pct
    
    ax3.plot(equity.index, equity.cumsum(), label='Equity Curve', color='blue')
    ax3.set_title("Equity Curve", fontsize=14)
    ax3.set_xlabel('')
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    ax4 = fig.add_subplot(gs[2, :])
    metrics_data = [
        ['Win Rate', f"{metrics['win_rate']:.2f}%"],
        ['Average Win', f"{metrics['avg_win']:.2f}%"],
        ['Average Loss', f"{metrics['avg_loss']:.2f}%"],
        ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
        ['Total Return', f"{metrics['total_return']:.2f}%"]
    ]
    metrics_data.append(['Total Trades', str(len(trades))])
    winning_trades = [t for t in trades if 'profit_pct' in t and t['profit_pct'] > 0]
    losing_trades = [t for t in trades if 'profit_pct' in t and t['profit_pct'] <= 0]
    metrics_data.append(['Winning Trades', str(len(winning_trades))])
    metrics_data.append(['Losing Trades', str(len(losing_trades))])
    
    ax4.axis('off')
    table = ax4.table(
        cellText=metrics_data,
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax4.set_title("Performance Metrics", fontsize=14)
    
    plt.tight_layout()
    fig.suptitle("Trading Strategy Backtest Results", fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage
if __name__ == "__main__":
    from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
    from svm_predictor import extract_wave_features, train_calibrated_svm, predict_next_pivot
    from intraday_predictor import extract_price_extreme_features, train_extreme_prediction_models, predict_next_day_extremes
    from hourly_pivot_predictor import extract_hourly_features, train_hourly_pivot_model, predict_next_hourly_pivot
    
    from config import DEFAULT_SYMBOL as symbol
    data = fetch_market_data(symbol, interval='1h', period='60d')
    data = clean_yf_data(data)
    
    wave_data = detect_waves(data)
    wave_data = calculate_wave_metrics(wave_data, data)
    wave_data = calculate_pivot_confidence(wave_data, data)
    
    X, y = extract_wave_features(wave_data, lookback=5)
    
    if len(X) > 0:
        model, scaler = train_calibrated_svm(X, y)
        pivot_prediction = predict_next_pivot(model, scaler, wave_data)
        
        features = extract_price_extreme_features(data, wave_data)
        if not features.empty:
            intraday_models = train_extreme_prediction_models(features)
            intraday_prediction = predict_next_day_extremes(intraday_models, data, wave_data)
        else:
            intraday_prediction = None
        
        X_hourly, y_hourly = extract_hourly_features(data, wave_data)
        if len(X_hourly) > 0:
            hourly_model, hourly_scaler = train_hourly_pivot_model(X_hourly, y_hourly)
            hourly_prediction = predict_next_hourly_pivot(hourly_model, hourly_scaler, wave_data, data)
        else:
            hourly_prediction = None
        
        fig1 = plot_wave_analysis(data, wave_data, save_path='./wave_analysis.png')
        fig2 = plot_predictions(data, wave_data, pivot_prediction, intraday_prediction, hourly_prediction, save_path='./predictions.png')
        # The interactive chart creation is assumed to be defined similarly,
        # but is omitted here for brevity.
        backtest_results = backtest_strategy(data, wave_data, model, scaler)
        fig4 = plot_backtest_results(backtest_results, save_path='./backtest_results.png')
        
        print("Visualizations created successfully!")
    else:
        print("Not enough data to extract features")
