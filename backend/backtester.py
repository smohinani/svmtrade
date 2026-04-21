"""
Walk-forward backtester for the MURLI pivot prediction system.

How it works
------------
For each confirmed pivot in history (starting after min_train_pivots), train the SVM
on all PRIOR pivots only, generate a prediction, then scan forward to see if TP or SL
was hit. This preserves temporal ordering so the model never sees future data.

A pivot at bar `i` is only "confirmed" once `order` bars have passed (required by
argrelextrema). Entry is placed at the open of the bar immediately after confirmation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
from svm_predictor import extract_wave_features, train_calibrated_svm, predict_next_pivot
from predictor_module import standardize_columns, _compute_atr


CONFIDENCE_THRESHOLD = 0.60
MIN_RR_RATIO = 1.5          # Only trade setups with reward >= 1.5x risk


@dataclass
class Trade:
    pivot_k: int             # Which pivot index triggered this trade
    confirm_bar: int         # Bar where pivot was confirmed (pivot_bar + order)
    signal_type: str         # 'Peak' (long) or 'Trough' (short)
    confidence: float
    entry_bar: int
    entry_price: float
    tp: float
    sl: float
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'tp' | 'sl' | 'timeout'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


def _build_wave_subset(all_pivot_indices, pivot_types, wave_heights,
                       wave_durations, wave_slopes, confidence_scores, k: int) -> dict:
    """Slice the pre-computed wave arrays to include only pivots 0..k."""
    return {
        'all_pivot_indices': all_pivot_indices[:k + 1],
        'all_pivot_values': None,  # not needed for feature extraction
        'pivot_types': pivot_types[:k + 1],
        'wave_heights': wave_heights[:k + 1],
        'wave_durations': wave_durations[:k + 1],
        'wave_slopes': wave_slopes[:k + 1],
        'confidence_scores': confidence_scores[:k + 1],
    }


def run_backtest(
    symbol: str,
    interval: str,
    period: str = '365d',
    lookback: int = 5,
    min_train_pivots: int = 20,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    max_holding_bars: int = 60,
) -> dict:
    """
    Full walk-forward backtest.

    Parameters
    ----------
    sl_atr_mult : float
        ATR multiplier for stop-loss distance.
    tp_atr_mult : float
        ATR multiplier for take-profit distance.
        Default 2.5 gives R:R = 2.5/1.5 ≈ 1.67 which exceeds MIN_RR_RATIO.
    max_holding_bars : int
        Force-exit after this many bars if neither TP nor SL was hit.
    """
    data = fetch_market_data(symbol, interval, period)
    data = standardize_columns(data)
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    if len(data) < 100:
        return {'error': 'Insufficient data for backtest', 'bars': len(data)}

    # Detect all pivots on the full dataset once.
    # argrelextrema is causal to within `order` bars, so no true look-ahead here.
    waves = detect_waves(data, adaptive=True)
    waves = calculate_wave_metrics(waves, data)
    waves = calculate_pivot_confidence(waves, data)

    order = int(waves.get('order_used', 5))
    all_pivot_indices = waves['all_pivot_indices']
    pivot_types = waves['pivot_types']
    wave_heights = waves['wave_heights']
    wave_durations = waves['wave_durations']
    wave_slopes = waves['wave_slopes']
    confidence_scores = waves['confidence_scores']

    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values

    n_pivots = len(all_pivot_indices)
    trades: list[Trade] = []
    skipped_low_confidence = 0
    skipped_low_rr = 0
    skipped_no_data = 0

    # Walk forward: for pivot k, train on pivots 0..k-1, predict pivot k+1 direction
    for k in range(min_train_pivots + lookback, n_pivots - 1):
        confirm_bar = int(all_pivot_indices[k]) + order

        # Need at least 2 bars after confirmation (entry + 1 check bar)
        if confirm_bar + 2 >= len(data):
            skipped_no_data += 1
            break

        # Build training set: pivots 0..k (inclusive) — we predict what comes AFTER k
        sub_waves = _build_wave_subset(
            all_pivot_indices, pivot_types, wave_heights,
            wave_durations, wave_slopes, confidence_scores, k
        )
        # ATR at confirmation bar
        atr = _compute_atr(data.iloc[:confirm_bar + 1])
        sub_waves['atr'] = atr
        # all_pivot_values needed by predict_next_pivot for estimated_value
        sub_waves['all_pivot_values'] = waves['all_pivot_values'][:k + 1]

        X, y = extract_wave_features(sub_waves, lookback)
        if len(X) < max(5, min_train_pivots - lookback):
            skipped_no_data += 1
            continue

        model, scaler = train_calibrated_svm(X, y)
        pred = predict_next_pivot(model, scaler, sub_waves, lookback)

        confidence = pred.get('confidence', 0.0)
        if confidence < CONFIDENCE_THRESHOLD:
            skipped_low_confidence += 1
            continue

        signal = pred.get('predicted_type_name')
        if signal not in ('Peak', 'Trough'):
            skipped_low_confidence += 1
            continue

        # Entry at open of bar immediately after confirmation
        entry_bar = confirm_bar + 1
        entry_price = float(close[entry_bar])
        atr_val = atr or entry_price * 0.005

        # TP / SL levels
        if signal == 'Peak':   # Long
            tp = entry_price + tp_atr_mult * atr_val
            sl = entry_price - sl_atr_mult * atr_val
        else:                  # Short
            tp = entry_price - tp_atr_mult * atr_val
            sl = entry_price + sl_atr_mult * atr_val

        rr = tp_atr_mult / sl_atr_mult
        if rr < MIN_RR_RATIO:
            skipped_low_rr += 1
            continue

        # Scan forward for exit
        exit_bar = exit_price = exit_reason = None
        for b in range(entry_bar + 1, min(entry_bar + max_holding_bars + 1, len(data))):
            bh, bl = high[b], low[b]
            if signal == 'Peak':
                if bl <= sl:
                    exit_bar, exit_price, exit_reason = b, sl, 'sl'
                    break
                if bh >= tp:
                    exit_bar, exit_price, exit_reason = b, tp, 'tp'
                    break
            else:
                if bh >= sl:
                    exit_bar, exit_price, exit_reason = b, sl, 'sl'
                    break
                if bl <= tp:
                    exit_bar, exit_price, exit_reason = b, tp, 'tp'
                    break

        if exit_bar is None:
            exit_bar = min(entry_bar + max_holding_bars, len(data) - 1)
            exit_price = float(close[exit_bar])
            exit_reason = 'timeout'

        raw_pnl = (exit_price - entry_price) if signal == 'Peak' else (entry_price - exit_price)
        pnl_pct = raw_pnl / entry_price * 100

        trades.append(Trade(
            pivot_k=k,
            confirm_bar=confirm_bar,
            signal_type=signal,
            confidence=confidence,
            entry_bar=entry_bar,
            entry_price=entry_price,
            tp=tp,
            sl=sl,
            exit_bar=exit_bar,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=raw_pnl,
            pnl_pct=pnl_pct,
        ))

    metrics = _compute_metrics(trades)

    return {
        'symbol': symbol,
        'interval': interval,
        'period': period,
        'total_bars': len(data),
        'total_pivots': n_pivots,
        'skipped_low_confidence': skipped_low_confidence,
        'skipped_low_rr': skipped_low_rr,
        'skipped_no_data': skipped_no_data,
        'params': {
            'sl_atr_mult': sl_atr_mult,
            'tp_atr_mult': tp_atr_mult,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'min_rr': MIN_RR_RATIO,
        },
        'metrics': metrics,
        'trades': [_trade_to_dict(t, data) for t in trades],
    }


def _compute_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {'total_trades': 0, 'note': 'No trades met filters'}

    pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]

    gross_win = sum(wins)
    gross_loss = abs(sum(losses)) if losses else 0

    win_rate = len(wins) / len(pnl_pcts) if pnl_pcts else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy = float(np.mean(pnl_pcts)) if pnl_pcts else 0

    # Equity curve (cumulative % return assuming equal-weight trades)
    equity = np.cumsum(pnl_pcts)
    max_dd = 0.0
    peak = equity[0]
    for v in equity:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    by_reason = {r: sum(1 for t in trades if t.exit_reason == r)
                 for r in ('tp', 'sl', 'timeout')}

    avg_confidence = float(np.mean([t.confidence for t in trades]))

    return {
        'total_trades': len(trades),
        'win_rate': round(win_rate, 3),
        'avg_win_pct': round(float(np.mean(wins)), 3) if wins else 0,
        'avg_loss_pct': round(float(np.mean(losses)), 3) if losses else 0,
        'profit_factor': round(profit_factor, 3),
        'expectancy_pct': round(expectancy, 3),
        'total_return_pct': round(float(sum(pnl_pcts)), 3),
        'max_drawdown_pct': round(max_dd, 3),
        'avg_confidence': round(avg_confidence, 3),
        'by_exit_reason': by_reason,
    }


def _trade_to_dict(t: Trade, data: pd.DataFrame) -> dict:
    entry_time = str(data.index[t.entry_bar]) if t.entry_bar < len(data) else None
    exit_time = str(data.index[t.exit_bar]) if t.exit_bar and t.exit_bar < len(data) else None
    return {
        'signal_type': t.signal_type,
        'confidence': round(t.confidence, 3),
        'entry_time': entry_time,
        'entry_price': round(t.entry_price, 2),
        'tp': round(t.tp, 2),
        'sl': round(t.sl, 2),
        'exit_time': exit_time,
        'exit_price': round(t.exit_price, 2) if t.exit_price else None,
        'exit_reason': t.exit_reason,
        'pnl_pct': round(t.pnl_pct, 3) if t.pnl_pct is not None else None,
    }
