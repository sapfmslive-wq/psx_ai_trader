"""Technical indicators (Phase 11, Section 18 of the project spec).

Every function here is causal by construction: it uses pandas rolling/EWM
operations, which at index i only ever read data at indices <= i. None of
these functions look ahead — see tests/test_indicators_no_lookahead.py for
the automated check Section 58 requires (mutate a future value, recompute,
assert every earlier value is unchanged).

Modules:
    trend.py       - SMA, EMA, MACD, ADX
    momentum.py    - RSI, ROC, Momentum, Stochastic
    volatility.py  - ATR, Bollinger Bands, rolling volatility
    volume.py      - OBV, VWAP, volume moving average, distance from VWAP
    engine.py       - computes every configured indicator from a candle
                       DataFrame and (optionally) stores results as Feature
                       rows in the database

All functions take and return `pandas.Series` (or a `DataFrame` for
multi-output indicators like MACD/Bollinger/Stochastic), indexed however
the caller's input was indexed (typically candle open_time). Leading values
that don't yet have a full window are `NaN` — never a fabricated number.
"""
