from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predictor_module import run_pivot_prediction, get_entry_confirmation, get_support_resistance
from backtester import run_backtest
import numpy as np
import pandas as pd
import pytz
import math
from collections import defaultdict, deque

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTION_BUFFERS: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=10))
PREV_REGIMES: dict[tuple, str | None] = defaultdict(lambda: None)


class PredictRequest(BaseModel):
    symbol: str
    intervals: list[str] = ["1h", "4h"]
    period_map: dict = {"1h": "15d", "4h": "30d"}


def safe_float(val):
    try:
        if isinstance(val, (np.floating, float, int)):
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)
        return val
    except Exception:
        return None


def compute_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


@app.post("/predict")
def predict(req: PredictRequest):
    consensus = {}
    predictions = {}
    stored_data = {}
    stored_waves = {}
    interval_results = {}

    for ivl in req.intervals:
        period = req.period_map.get(ivl, "30d")
        buf_key = (req.symbol, ivl)
        pred_buffer = PREDICTION_BUFFERS[buf_key]
        prev_regime = PREV_REGIMES[buf_key]

        pred, data, waves = run_pivot_prediction(
            req.symbol, ivl, period,
            pred_buffer=pred_buffer,
            prev_regime=prev_regime,
        )
        predictions[ivl] = pred
        PREV_REGIMES[buf_key] = pred.get('regime', prev_regime)
        name = pred.get("regime", "Unknown")
        consensus[ivl] = name

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        stored_data[ivl] = data
        stored_waves[ivl] = waves

        # ATR
        if all(col in data.columns for col in ["High", "Low", "Close"]) and not data.empty:
            high, low, close = data["High"], data["Low"], data["Close"]
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean()
            last_atr = float(atr.iloc[-1]) if not atr.empty else None
            last_close = float(close.iloc[-1]) if not close.empty else None
            atr_ratio = last_atr / last_close if last_atr and last_close else None
        else:
            last_atr = atr_ratio = None

        low_vol_thresholds = {"1h": 2.0, "4h": 5.0}
        low_volatility = last_atr is not None and last_atr < low_vol_thresholds.get(ivl, 2.0)

        # Entry / exit / support / resistance
        entry = get_entry_confirmation(data, pred.get("predicted_type_name")) if not data.empty else None
        exit_target = pred.get("estimated_value", 0) or 0
        support, resistance = get_support_resistance(data, waves=waves) if not data.empty else (0, 0)

        pred_type = pred.get("predicted_type_name")
        is_valid = False
        if pred_type in ("Peak", "Trough") and entry is not None:
            if pred_type == "Trough" and exit_target <= entry and support <= entry <= resistance:
                is_valid = True
            elif pred_type == "Peak" and exit_target >= entry and support <= entry <= resistance:
                is_valid = True

        if entry is not None and is_valid:
            risk = (entry - support) if pred_type == "Peak" else (resistance - entry)
            reward = abs(exit_target - entry)
            rr = reward / risk if risk and reward else None
        else:
            rr = None

        # Date window for charting
        if not data.empty:
            idx_max = data.index.max()
            if isinstance(idx_max, pd.Timestamp) and not pd.isna(idx_max):
                min_date = idx_max - pd.Timedelta(days=30)
            else:
                min_date = data.index[0]
            data_window = data[data.index >= min_date]
        else:
            data_window = data

        start_idx = len(data) - len(data_window)
        peaks_all = waves.get("peaks_indices", np.array([])).tolist() if waves else []
        troughs_all = waves.get("troughs_indices", np.array([])).tolist() if waves else []
        peaks = [i - start_idx for i in peaks_all if start_idx <= i < len(data)]
        troughs = [i - start_idx for i in troughs_all if start_idx <= i < len(data)]

        # Projected time for next pivot
        bars_ahead = int(round(pred.get('estimated_index_offset', 0) or 0))
        projected_time_et = None
        if not data_window.empty and bars_ahead:
            last_time = data_window.index[-1]
            if isinstance(last_time, pd.Timestamp) and not pd.isna(last_time):
                if ivl.endswith('m'):
                    delta = pd.Timedelta(minutes=bars_ahead * int(ivl[:-1]))
                else:
                    delta = pd.Timedelta(hours=bars_ahead * int(ivl[:-1]))
                eastern = pytz.timezone("US/Eastern")
                if last_time.tzinfo is None:
                    last_time = last_time.tz_localize('UTC')
                next_time = last_time + delta
                if isinstance(next_time, pd.Timestamp) and not pd.isna(next_time):
                    projected_time_et = next_time.astimezone(eastern).strftime("%H:%M")

        data_window = data_window.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        ohlcv = data_window.reset_index().to_dict(orient="records")

        # MACD confirmation
        macd_tick = False
        if not data.empty and 'Close' in data.columns and len(data['Close']) >= 26:
            macd, macd_signal = compute_macd(data['Close'])
            macd_val = float(macd.iloc[-1])
            signal_line_val = float(macd_signal.iloc[-1])
            regime = pred.get('regime')
            if regime == 'Peak' and macd_val > signal_line_val:
                macd_tick = True
            elif regime == 'Trough' and macd_val < signal_line_val:
                macd_tick = True

        close_series = data_window.get("Close") if hasattr(data_window, 'get') else data_window["Close"] if "Close" in data_window.columns else None
        latest_price = safe_float(close_series.iloc[-1]) if close_series is not None and not close_series.empty else None
        latest_timestamp = str(data_window.index[-1]) if not data_window.empty else None

        interval_results[ivl] = {
            "prediction": pred,
            "entry": safe_float(entry),
            "exit_target": safe_float(exit_target),
            "support": safe_float(support),
            "resistance": safe_float(resistance),
            "risk_reward": safe_float(rr),
            "is_valid": is_valid,
            "ohlcv": ohlcv,
            "peaks": peaks,
            "troughs": troughs,
            "latest_price": latest_price,
            "latest_timestamp": latest_timestamp,
            "projected_time_et": projected_time_et,
            "atr": safe_float(last_atr),
            "atr_ratio": safe_float(atr_ratio),
            "low_volatility": low_volatility,
            "macd_tick": macd_tick,
        }

    # Consensus — requires >= 2 of N timeframes agreeing on same direction
    valid_types = [
        consensus[ivl] for ivl in req.intervals
        if consensus.get(ivl) in ("Peak", "Trough") and interval_results.get(ivl, {}).get("is_valid")
    ]
    peak_count = valid_types.count("Peak")
    trough_count = valid_types.count("Trough")
    total = len(req.intervals)

    entry_levels = [
        interval_results[ivl]["entry"] for ivl in req.intervals
        if interval_results[ivl].get("is_valid") and interval_results[ivl]["entry"] is not None
    ]
    exit_levels = [
        interval_results[ivl]["exit_target"] for ivl in req.intervals
        if interval_results[ivl].get("is_valid")
    ]

    # Pick a reference data source for S/R (prefer first valid interval)
    ref_ivl = next((ivl for ivl in req.intervals if not stored_data.get(ivl, pd.DataFrame()).empty), None)

    if (peak_count >= 2 or trough_count >= 2) and entry_levels and exit_levels:
        signal_type = "CALL" if peak_count >= trough_count else "PUT"
        avg_entry = float(np.mean(entry_levels))
        avg_exit = float(np.mean(exit_levels))
        strike = int(round(avg_entry))

        if ref_ivl:
            ref_support, ref_resistance = get_support_resistance(stored_data[ref_ivl], waves=stored_waves.get(ref_ivl))
        else:
            ref_support = avg_entry * 0.995
            ref_resistance = avg_entry * 1.005

        if signal_type == "CALL":
            sl = round(ref_support, 2)
            tp = round(avg_exit, 2)
            risk = max(avg_entry - sl, 1e-6)
            reward = max(tp - avg_entry, 1e-6)
        else:
            sl = round(ref_resistance, 2)
            tp = round(avg_exit, 2)
            risk = max(sl - avg_entry, 1e-6)
            reward = max(avg_entry - tp, 1e-6)

        rr_consensus = reward / risk
        consensus_msg = {
            "signal": signal_type,
            "strike": strike,
            "avg_entry": avg_entry,
            "sl": sl,
            "tp": tp,
            "risk_reward": safe_float(rr_consensus),
            "agreeing": f"{max(peak_count, trough_count)}/{total}",
            "is_trade": True,
        }
    else:
        reason = "Mixed signals across timeframes" if valid_types else "No timeframe reached confidence threshold"
        consensus_msg = {
            "signal": "NO_TRADE",
            "reason": reason,
            "is_trade": False,
        }

    return {"intervals": interval_results, "consensus": consensus_msg}


class BacktestRequest(BaseModel):
    symbol: str
    interval: str = "1h"
    period: str = "365d"
    lookback: int = 5
    min_train_pivots: int = 20
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5
    max_holding_bars: int = 60


@app.post("/backtest")
def backtest(req: BacktestRequest):
    return run_backtest(
        symbol=req.symbol,
        interval=req.interval,
        period=req.period,
        lookback=req.lookback,
        min_train_pivots=req.min_train_pivots,
        sl_atr_mult=req.sl_atr_mult,
        tp_atr_mult=req.tp_atr_mult,
        max_holding_bars=req.max_holding_bars,
    )
