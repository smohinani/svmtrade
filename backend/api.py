from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predictor_module import run_pivot_prediction, get_entry_confirmation, get_support_resistance
from wave_detector import fetch_market_data, detect_waves, calculate_wave_metrics, calculate_pivot_confidence
import numpy as np
import pandas as pd
import pytz
import math
from collections import defaultdict, deque

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add global buffers for regime smoothing
# For each interval, keep a buffer of last 10 predictions and the previous regime
PREDICTION_BUFFERS = defaultdict(lambda: deque(maxlen=10))
PREV_REGIMES = defaultdict(lambda: None)

class PredictRequest(BaseModel):
    symbol: str
    intervals: list[str] = ["5m", "15m", "1h"]
    period_map: dict = {"5m": "30d", "15m": "30d", "1h": "60d"}

# Helper to sanitize floats for JSON
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
    price_targets = []
    stored_data = {}
    stored_waves = {}
    interval_results = {}
    atrs = {}
    atr_ratios = {}

    # Run predictions & store data for each interval
    for ivl in req.intervals:
        period = req.period_map.get(ivl, "30d")
        # Use global buffer and regime for this interval
        pred_buffer = PREDICTION_BUFFERS[ivl]
        prev_regime = PREV_REGIMES[ivl]
        pred = run_pivot_prediction(req.symbol, ivl, period, pred_buffer=pred_buffer, prev_regime=prev_regime)
        predictions[ivl] = pred
        # Update previous regime for this interval
        PREV_REGIMES[ivl] = pred.get('regime', prev_regime)
        # Use regime for consensus display
        name = pred.get("regime", "Unknown")
        consensus[ivl] = name
        if name in ["Trough", "Peak"]:
            price_targets.append(float(pred.get("estimated_value", 0)))

        # fetch data & waves
        data = fetch_market_data(req.symbol, ivl, period)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        else:
            data.columns = [col.capitalize() for col in data.columns]
        # Ensure data.index is a DatetimeIndex for timezone and subtraction operations
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        # Remove timezone if present
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        # data["Close"] = data["Close"].rolling(window=3, min_periods=1).mean()  # REMOVE this line to use raw close price
        waves = detect_waves(data, adaptive=True)
        waves = calculate_wave_metrics(waves, data)
        waves = calculate_pivot_confidence(waves, data)
        stored_data[ivl] = data
        stored_waves[ivl] = waves

        # ATR calculation (14-period by default)
        if all(col in data.columns for col in ["High", "Low", "Close"]):
            high = data["High"]
            low = data["Low"]
            close = data["Close"]
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean()
            last_atr = float(atr.iloc[-1]) if isinstance(atr, pd.Series) and not atr.empty else None
            last_close = float(close.iloc[-1]) if isinstance(close, pd.Series) and not close.empty else None
            atr_ratio = last_atr / last_close if last_atr is not None and last_close else None
        else:
            last_atr = None
            atr_ratio = None
        atrs[ivl] = last_atr
        atr_ratios[ivl] = atr_ratio

        # Low volatility thresholds by interval
        if ivl == "5m":
            low_volatility = last_atr is not None and last_atr < 0.6
        elif ivl == "15m":
            low_volatility = last_atr is not None and last_atr < 1.0
        elif ivl == "1h":
            low_volatility = last_atr is not None and last_atr < 2.0
        else:
            low_volatility = False

        # SPY–QQQ Convergence/Divergence logic (only for 15m interval)
        spy_qqq_info = {}
        if ivl == "15m":
            try:
                spy_data = fetch_market_data("SPY", ivl, period)
                qqq_data = fetch_market_data("QQQ", ivl, period)
                # Use last 3 bars for % change
                n_bars = 3
                spy_close = spy_data["Close"]
                qqq_close = qqq_data["Close"]
                if len(spy_close) > n_bars and len(qqq_close) > n_bars:
                    spy_pct = (spy_close.iloc[-1] - spy_close.iloc[-n_bars-1]) / spy_close.iloc[-n_bars-1]
                    qqq_pct = (qqq_close.iloc[-1] - qqq_close.iloc[-n_bars-1]) / qqq_close.iloc[-n_bars-1]
                    # Direction
                    spy_dir = "up" if spy_pct > 0 else "down" if spy_pct < 0 else "flat"
                    qqq_dir = "up" if qqq_pct > 0 else "down" if qqq_pct < 0 else "flat"
                    if spy_dir == qqq_dir and spy_dir != "flat":
                        convergence_status = "convergent"
                        leader = None
                        leader_direction = None
                        signal_agrees_with_leader = None
                    elif spy_dir != qqq_dir and spy_dir != "flat" and qqq_dir != "flat":
                        convergence_status = "divergent"
                        # Leader: greater abs pct change
                        if abs(spy_pct) > abs(qqq_pct):
                            leader = "SPY"
                            leader_direction = spy_dir
                        else:
                            leader = "QQQ"
                            leader_direction = qqq_dir
                        # Does bot signal agree with leader?
                        bot_signal = pred.get("predicted_type_name", None)
                        if bot_signal == "Peak":
                            bot_dir = "up"
                        elif bot_signal == "Trough":
                            bot_dir = "down"
                        else:
                            bot_dir = None
                        signal_agrees_with_leader = (bot_dir == leader_direction) if bot_dir else None
                    else:
                        convergence_status = "neutral"
                        leader = None
                        leader_direction = None
                        signal_agrees_with_leader = None
                    spy_qqq_info = {
                        "spy_pct_change": float(spy_pct),
                        "qqq_pct_change": float(qqq_pct),
                        "convergence_status": convergence_status,
                        "leader": leader,
                        "leader_direction": leader_direction,
                        "signal_agrees_with_leader": signal_agrees_with_leader
                    }
            except Exception as e:
                spy_qqq_info = {"error": str(e)}

        # Entry/exit/support/resistance/risk-reward
        entry = get_entry_confirmation(data, pred.get("predicted_type_name"))
        exit_target = pred.get("estimated_value", 0)
        support, resistance = get_support_resistance(data)
        pred_type = pred.get("predicted_type_name")
        is_valid = True
        if pred_type == "Trough":
            if exit_target > entry or not (support <= entry <= resistance):
                is_valid = False
        elif pred_type == "Peak":
            if exit_target < entry or not (support <= entry <= resistance):
                is_valid = False
        # Risk/reward
        if entry is not None:
            if pred_type == "Peak":
                risk = entry - support
            else:
                risk = resistance - entry
            reward = abs(exit_target - entry)
            rr = reward / risk if risk and reward else None
        else:
            rr = None
        # Use a dynamic date window for each interval, matching Streamlit
        idx_max = data.index.max()
        if isinstance(idx_max, pd.DatetimeIndex):
            idx_max = idx_max[-1] if len(idx_max) > 0 else None
        if isinstance(idx_max, pd.Timestamp) and not pd.isna(idx_max):
            if ivl in ["5m", "15m"]:
                min_date = idx_max - pd.Timedelta(days=20)
            elif ivl == "1h":
                min_date = idx_max - pd.Timedelta(days=60)
            else:
                min_date = data.index[0]
        else:
            min_date = data.index[0]
        data_window = data[data.index >= min_date]
        start_idx = len(data) - len(data_window)
        peaks_all = waves.get("peaks_indices", np.array([]))
        troughs_all = waves.get("troughs_indices", np.array([]))
        if isinstance(peaks_all, np.ndarray):
            peaks_all = peaks_all.tolist()
        if isinstance(troughs_all, np.ndarray):
            troughs_all = troughs_all.tolist()
        # Only include peaks/troughs within the window, and re-index them
        peaks = [i - start_idx for i in peaks_all if start_idx <= i < len(data)]
        troughs = [i - start_idx for i in troughs_all if start_idx <= i < len(data)]
        # Calculate projected time for the next pivot (like Streamlit)
        bars_ahead = int(round(pred.get('estimated_index_offset', 0)))
        num, unit = int(ivl[:-1]), ivl[-1]
        # For projected_time, ensure last_time is a Timestamp and not NaT
        if not data_window.empty:
            last_time = data_window.index[-1]
            if isinstance(last_time, pd.DatetimeIndex):
                last_time = last_time[-1] if len(last_time) > 0 else None
            # Only convert to datetime if it's a string
            if not isinstance(last_time, pd.Timestamp):
                if isinstance(last_time, str):
                    last_time = pd.to_datetime(last_time)
                else:
                    last_time = None
            if last_time is None or pd.isna(last_time):
                projected_time = None
                projected_time_et = None
            else:
                if unit == 'm':
                    delta = pd.Timedelta(minutes=bars_ahead * num)
                else:
                    delta = pd.Timedelta(hours=bars_ahead * num)
                # Convert to US/Eastern time
                eastern = pytz.timezone("US/Eastern")
                if isinstance(last_time, pd.Timestamp) and last_time.tzinfo is None:
                    last_time = last_time.tz_localize('UTC')
                if isinstance(last_time, pd.Timestamp) and not pd.isna(last_time):
                    next_time = last_time + delta
                    if isinstance(next_time, pd.Timestamp) and not pd.isna(next_time):
                        projected_time = next_time.astimezone(eastern).strftime("%Y-%m-%d %H:%M %Z")
                        projected_time_et = next_time.astimezone(eastern).strftime("%H:%M")
                    else:
                        projected_time = None
                        projected_time_et = None
                else:
                    projected_time = None
                    projected_time_et = None
        else:
            projected_time = None
            projected_time_et = None
        # Sanitize data_window for JSON
        data_window = data_window.replace([np.inf, -np.inf], np.nan)
        data_window = data_window.ffill().bfill()
        ohlcv_df = data_window.reset_index()
        for col in ohlcv_df.select_dtypes(include=[np.number]).columns:
            ohlcv_df[col] = ohlcv_df[col].astype(float)
        ohlcv = ohlcv_df.to_dict(orient="records")
        # MACD calculation for tick
        macd_tick = False
        if 'Close' in data.columns and len(data['Close']) >= 26:
            macd, macd_signal = compute_macd(data['Close'])
            macd_val = macd.iloc[-1]
            regime = pred.get('regime', None)
            if regime == 'Peak' and macd_val > 0:
                macd_tick = True
            elif regime == 'Trough' and macd_val < 0:
                macd_tick = True
        # Compose interval result
        close_series = data_window["Close"] if "Close" in data_window.columns else None
        latest_price = safe_float(close_series.iloc[-1]) if close_series is not None and isinstance(close_series, pd.Series) and not close_series.empty else None
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
            "projected_time": projected_time,
            "projected_time_et": projected_time_et,
            "atr": safe_float(last_atr),
            "atr_ratio": safe_float(atr_ratio),
            "low_volatility": low_volatility,
            "macd_tick": macd_tick,
            **(spy_qqq_info if ivl == "15m" else {}),
        }

    # Consensus logic
    types = [t for t in consensus.values() if t in ["Trough", "Peak"]]
    entry_levels = [interval_results[ivl]["entry"] for ivl in req.intervals if interval_results[ivl]["is_valid"]]
    exit_levels = [interval_results[ivl]["exit_target"] for ivl in req.intervals if interval_results[ivl]["is_valid"]]
    consensus_msg = ""
    rr = None
    # ATR/Close < 0.01 check is now informational only, not for invalidation
    # low_vol_intervals = [ivl for ivl in req.intervals if atr_ratios.get(ivl) is not None and atr_ratios[ivl] < 0.01]
    if all(t == "Trough" for t in types):
        consensus_type = "High Conviction Short (All Troughs)"
        if entry_levels and exit_levels:
            avg_entry = float(np.mean(entry_levels))
            avg_exit = float(np.mean(exit_levels))
            support, resistance = get_support_resistance(stored_data["1h"])
            if np.isfinite(support) and np.isfinite(resistance) and support <= avg_entry <= resistance:
                risk = max(resistance - avg_entry, 1e-6)
                reward = max(avg_entry - avg_exit, 1e-6)
                rr = reward / risk
            consensus_msg = {
                "type": consensus_type,
                "avg_entry": avg_entry,
                "avg_exit": avg_exit,
                "risk_reward": rr,
                "is_valid": True
            }
    elif all(t == "Peak" for t in types):
        consensus_type = "High Conviction Long (All Peaks)"
        if entry_levels and exit_levels:
            avg_entry = float(np.mean(entry_levels))
            avg_exit = float(np.mean(exit_levels))
            support, resistance = get_support_resistance(stored_data["1h"])
            if support <= avg_entry <= resistance:
                risk = max(avg_entry - support, 1e-6)
                reward = max(avg_exit - avg_entry, 1e-6)
                rr = reward / risk
            consensus_msg = {
                "type": consensus_type,
                "avg_entry": avg_entry,
                "avg_exit": avg_exit,
                "risk_reward": rr,
                "is_valid": True
            }
    else:
        consensus_msg = {"type": "Mixed Signals — No Trade", "is_valid": False}

    return {
        "intervals": interval_results,
        "consensus": consensus_msg
    } 