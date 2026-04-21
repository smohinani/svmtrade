import pandas as pd
import numpy as np
from wave_detector import fetch_market_data, detect_waves
from itertools import product

SYMBOL = 'SPY'
INTERVALS = ['5m', '15m', '1h']
PERIOD = '30d'

# Wide parameter grid for Goldilocks search
MIN_PCT_MOVE_GRID = [0.1/100, 0.3/100, 0.5/100, 0.7/100, 1.0/100, 1.5/100, 2.0/100]
MIN_ATR_MULT_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
COOLDOWN_BARS_GRID = [2, 5, 8, 10, 15, 20]
MIN_ABS_MOVE_GRID = [0, 0.5, 1.0, 2.0]

results = {ivl: [] for ivl in INTERVALS}

for INTERVAL in INTERVALS:
    print(f"\n=== {SYMBOL} {INTERVAL} ===")
    data = fetch_market_data(SYMBOL, INTERVAL, PERIOD)
    close = pd.Series(data['Close'])
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean()
    atr = pd.Series(atr)
    waves = detect_waves(data)
    pivot_indices = np.sort(np.concatenate([waves['peaks_indices'], waves['troughs_indices']]))
    pivot_indices = [int(idx) for idx in pivot_indices]

    for MIN_PCT_MOVE, MIN_ATR_MULT, COOLDOWN_BARS, MIN_ABS_MOVE in product(
        MIN_PCT_MOVE_GRID, MIN_ATR_MULT_GRID, COOLDOWN_BARS_GRID, MIN_ABS_MOVE_GRID):
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
                min_move = max(MIN_PCT_MOVE * last_pivot_price, MIN_ATR_MULT * current_atr, MIN_ABS_MOVE)
                if bars_since < COOLDOWN_BARS:
                    continue
                if price_move < min_move:
                    continue
            filtered_pivots.append(idx)
            last_pivot_idx = idx
            last_pivot_price = price
        avg_time = None
        if len(filtered_pivots) > 1:
            times = data.index[filtered_pivots]
            avg_time = (times[1:] - times[:-1]).mean()
        # Score: prefer pivots in Goldilocks range (e.g., 5-20), maximize avg_time
        score = 0
        if 5 <= len(filtered_pivots) <= 20:
            score = (avg_time.total_seconds() if avg_time is not None else 0) / len(filtered_pivots)
        results[INTERVAL].append({
            'min_pct_move': MIN_PCT_MOVE,
            'min_atr_mult': MIN_ATR_MULT,
            'cooldown_bars': COOLDOWN_BARS,
            'min_abs_move': MIN_ABS_MOVE,
            'original': len(pivot_indices),
            'filtered': len(filtered_pivots),
            'avg_time': avg_time,
            'score': score
        })

# Print top results for each interval
for INTERVAL in INTERVALS:
    print(f"\n=== Top Results for {SYMBOL} {INTERVAL} ===")
    sorted_results = sorted(results[INTERVAL], key=lambda x: x['score'], reverse=True)
    print(f"{'Min%':<6} {'ATRm':<6} {'CD':<3} {'Abs$':<6} {'Orig':<5} {'Filt':<5} {'AvgTime':<20} {'Score':<10}")
    for row in sorted_results[:15]:
        avg_time_str = str(row['avg_time'])[:19] if row['avg_time'] is not None else '-'
        print(f"{row['min_pct_move']*100:<6.2f} {row['min_atr_mult']:<6.2f} {row['cooldown_bars']:<3} {row['min_abs_move']:<6.2f} {row['original']:<5} {row['filtered']:<5} {avg_time_str:<20} {row['score']:<10.2f}") 