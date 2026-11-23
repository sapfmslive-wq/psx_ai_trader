"""Feature engineering (Phase 12, Sections 19/20 of the project spec).

Modules:
    market_regime.py - KSE-100/KMI-30 regime features (returns, volatility,
                        trend, position vs moving averages)
    stock_features.py - per-stock features NOT already covered by the
                         Phase 11 indicators (log returns, volume change/
                         acceleration, MA distance, breakouts, intraday
                         range, gap, relative volume, relative performance)
    engine.py          - aligns stock and index candles, computes both
                         groups, stores results as Feature rows

Deliberately NOT implemented, per Section 19's own "if valid data is
available" condition: **market breadth** and **sector performance**. Both
need per-constituent price data (which stocks are up/down, which sector
each belongs to), and this project only records the index level and the
individually-configured stocks — not every KSE-100/KMI-30 constituent. A
feature computed from a handful of watchlist stocks and presented as
"market breadth" would be misleading, so it is left out rather than
approximated.

Same no-look-ahead discipline as the indicators: every function here is
causal (rolling/shift(positive) only), and tests/test_features_no_lookahead.py
enforces it with Section 58's mutate-the-future check.
"""
