import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wave_detector import fetch_market_data, detect_waves

# --- Parameters to tune ---
SYMBOL = 'SPY'
INTERVALS = ['5m', '15m', '1h']
PERIOD = '30d'
MIN_PCT_MOVE = 0.2 / 100  # 0.2%
MIN_ATR_MULT = 0.5        # 0.5x ATR
COOLDOWN_BARS = 3         # bars to wait after a pivot

summary = []

for INTERVAL in INTERVALS:
    print(f"\n=== {SYMBOL} {INTERVAL} ===")
    # --- Fetch data ---
    data = fetch_market_data(SYMBOL, INTERVAL, PERIOD)
    close = pd.Series(data['Close'])
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])

    # --- ATR Calculation ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean()
    atr = pd.Series(atr)

    # --- Run basic wave detection ---
    waves = detect_waves(data)
    pivot_indices = np.sort(np.concatenate([waves['peaks_indices'], waves['troughs_indices']]))
    pivot_indices = [int(idx) for idx in pivot_indices]

    # --- Filter pivots by min move and cooldown ---
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
            min_move = max(MIN_PCT_MOVE * last_pivot_price, MIN_ATR_MULT * current_atr)
            if bars_since < COOLDOWN_BARS:
                continue
            if price_move < min_move:
                continue
        filtered_pivots.append(idx)
        last_pivot_idx = idx
        last_pivot_price = price

    # --- Print stats ---
    print(f"Min % Move: {MIN_PCT_MOVE*100:.2f}%, Min ATR Mult: {MIN_ATR_MULT}, Cooldown Bars: {COOLDOWN_BARS}")
    print(f"Original pivots: {len(pivot_indices)} | Filtered pivots: {len(filtered_pivots)}")
    avg_time = None
    if len(filtered_pivots) > 1:
        times = data.index[filtered_pivots]
        avg_time = (times[1:] - times[:-1]).mean()
        print(f"Avg time between pivots: {avg_time}")
    summary.append({
        'interval': INTERVAL,
        'original': len(pivot_indices),
        'filtered': len(filtered_pivots),
        'avg_time': avg_time
    })

    # --- Plot ---
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, close, label='Close')
    plt.scatter(data.index[filtered_pivots], close.iloc[filtered_pivots], color='orange', label='Filtered Pivots')
    plt.title(f"{SYMBOL} {INTERVAL} - Filtered Pivots")
    plt.legend()
    plt.show()

# --- Summary Table ---
print("\n=== Summary ===")
print(f"{'Interval':<8} {'Original':<10} {'Filtered':<10} {'Avg Time Between Pivots'}")
for row in summary:
    avg_time_str = str(row['avg_time']) if row['avg_time'] is not None else '-'
    print(f"{row['interval']:<8} {row['original']:<10} {row['filtered']:<10} {avg_time_str}") 