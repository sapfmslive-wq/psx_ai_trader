"""Label engine (Phase 13): computes every configured prediction target for
a symbol's candles and stores them in the `labels` table.

Labels for the last `horizon` bars of the candle range read are NaN (their
future is not in the data) and are never stored. To label the final bars of
a range, extend `end` past them by at least the largest horizon — those
later candles are used only to compute the earlier bars' labels.

`LABEL_VERSION` is independent of the feature versions.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.features import labels
from src.indicators.engine import candles_to_dataframe

LABEL_VERSION = "labels_v1"


def compute_labels(df: pd.DataFrame, config) -> pd.DataFrame:  # noqa: ANN001 — LabelsConfig
    """All configured targets from a candle DataFrame, one column each:
    label_future_return_{h}bar, label_direction_{h}bar, label_3class_{h}bar,
    label_tp_before_sl_{h}bar, for every configured horizon h."""
    if df.empty:
        return pd.DataFrame(index=df.index)

    close, high, low = df["close"], df["high"], df["low"]
    columns: dict[str, pd.Series] = {}
    for h in config.horizons:
        future_ret = labels.future_return(close, h)
        columns[f"label_future_return_{h}bar"] = future_ret
        columns[f"label_direction_{h}bar"] = labels.direction_label(future_ret)
        columns[f"label_3class_{h}bar"] = labels.three_class_label(future_ret, config.neutral_threshold_pct)
        columns[f"label_tp_before_sl_{h}bar"] = labels.tp_before_sl_label(
            high, low, close, h, config.take_profit_pct, config.stop_loss_pct
        )
    return pd.DataFrame(columns, index=df.index)


def compute_and_store_labels(
    config,  # noqa: ANN001 — AppConfig
    session_factory,  # noqa: ANN001
    symbol: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
) -> dict[str, int]:
    """Read candles in [start, end), compute labels, store non-NaN values
    in the `labels` table. Returns {"written": N, "duplicate": M}."""
    from src.database.repository import get_candles, insert_label

    with session_factory() as session:
        candles = get_candles(session, symbol, timeframe, start, end)

    labels_df = compute_labels(candles_to_dataframe(candles), config.labels)

    written = duplicate = 0
    with session_factory() as session:
        for column in labels_df.columns:
            for timestamp, value in labels_df[column].items():
                if pd.isna(value):
                    continue
                inserted = insert_label(
                    session,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp.to_pydatetime(),
                    label_name=column,
                    label_value=float(value),
                    label_version=LABEL_VERSION,
                )
                if inserted:
                    written += 1
                else:
                    duplicate += 1
    return {"written": written, "duplicate": duplicate}
