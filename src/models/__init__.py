"""Models (Phase 14+).

    baseline.py        - the six baseline direction predictors (Section 23)
    evaluation.py       - classification metrics, chronological split, and an
                          honest "is this distinguishable from chance?" check
    baseline_runner.py  - builds a full comparison report and records each
                          baseline as a `model_runs` row (Section 100)

Baselines exist to be beaten. They are prediction-quality benchmarks scored
against the Phase 13 direction labels — NOT trading strategies. No costs or
slippage are applied here (that is the Phase 16 backtester), so nothing in
this package can say a baseline is profitable.
"""
