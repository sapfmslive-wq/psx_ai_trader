# PSX AI Trader — Local Market Data, Research & Paper-Trading Platform

Status: **PHASE 15 — ML Models** (of 26 planned phases)
Mode: **RESEARCH** (no modeling or trading logic exists yet)

---

## 0. Read This First

This project is a long-horizon engineering build, not a one-shot script. It is being
built **in phases**, each one complete, tested, and verified before the next begins.

At this stage the project **cannot**:
- connect to *real, live* PSX data (see the Phase 7 write-up below for
  exactly why, and exactly what's needed to enable it — this is a licensing
  reality, not a missing-code problem)
- connect to a broker (Phase 26)
- use a trained model for anything yet: models are trained, calibrated and
  compared against the baselines, but nothing consumes their predictions —
  no signals (Phase 18), no backtest (Phase 16), no walk-forward
  validation (Phase 17); a `predict` command and a return-magnitude
  regressor are deferred (see the Phase 15 write-up)
- say whether anything is *profitable*: no fees, tax, slippage or spread
  are modeled anywhere yet (Phase 16's backtester), so every result so far
  is about prediction quality, not money
- compute market breadth or sector performance (Section 19 allows these
  only "if valid data is available" — this project records the index level
  and the configured stocks, not every constituent, so there is nothing
  valid to compute them from)
- generate BUY/SELL signals of any kind (Phase 18)
- backtest, paper-trade, or live-trade anything (Phase 16, 20, 26)

What it **can** do right now:
- create and verify its own directory structure
- run a structured logging system (5 separate log streams)
- load, fully validate, and report on `config/settings.yaml` and
  `config/symbols.yaml` (all-or-nothing: any invalid value blocks startup)
- enforce the LIVE-trading safety switch at config-load time, not just at
  order-placement time
- create and query a real SQLite database with all 15 tables the whole
  project will eventually use, with working duplicate-prevention and
  foreign-key enforcement
- generate synthetic index/stock observations through a real
  `DataProvider` interface, clearly labeled SYNTHETIC end to end, with no
  external dependency at all
- import a real historical CSV file into the database, with automatic
  column detection, an explicit-mapping override, and every row —
  including invalid ones — classified and stored, never silently dropped
- run a real VALID/SUSPECT/INVALID/DUPLICATE data-quality engine covering
  bad prices, impossible OHLC, negative volume, unknown symbols,
  out-of-order/duplicate/far-future timestamps, extreme % moves, stale
  prices, and unrecognized market sessions — all flagged, nothing deleted
- connect to a real, licensed PSX data feed the moment you configure one —
  the adapter is fully built and tested end-to-end (including against a
  real local HTTP server), it just correctly refuses to run without your
  own license, because PSX's own terms require one
- answer `is_market_open()` / `current_session()` / `next_open()` /
  `next_close()` / `is_holiday()` correctly against PSX's actual published
  2026 trading hours and holiday calendar — including Friday's two-session
  split around the prayer break, weekends, holidays, and date-ranged
  special-hours overrides (e.g. Ramadan) — entirely from a YAML file, no
  PSX-specific date or time hardcoded in Python
- record one real (or synthetic) observation for KSE-100, KMI-30, and every
  configured stock, correctly refusing to do so when the market isn't open
  unless explicitly forced, with every row flagged by the quality engine
  and written through the same duplicate-safe database layer as everything
  else
- run a continuous, unattended collector loop with real exponential
  backoff on connection failure, graceful Ctrl+C/SIGTERM shutdown,
  automatic outage recording (`connection_lost`/`connection_restored`
  events with duration), and a configurable retry-limit that logs CRITICAL
  but never gives up on its own — it survives one bad record or one bad
  connection exactly as Section 15 requires, rather than crashing
- build real 1m/5m/15m/30m/60m/daily OHLCV candles from recorded ticks —
  correctly excluding pre-open/post-close observations and anything
  flagged INVALID, correctly handling Friday's two-session day in the
  daily candle, and correctly distinguishing cumulative from per-tick
  volume conventions rather than silently summing whichever one it is
  handed
- compute all fourteen indicators Section 18 asks for (SMA, EMA, MACD, ADX,
  RSI, ROC, Momentum, Stochastic, ATR, Bollinger Bands, rolling volatility,
  OBV, VWAP, distance from VWAP) from real candle data, with every period
  configurable, verified causal — no look-ahead — by an automated test per
  indicator that mutates only future values and asserts every past value
  is untouched, and stores results as versioned Feature rows
- build market-regime features for KSE-100 and KMI-30 (multi-horizon
  returns, realized volatility, a trend score, position vs each moving
  average) and stock features (log returns, volume change/acceleration,
  distance from moving averages, high/low breakouts, bar range, gap,
  relative volume) — including each stock's return *relative to* each
  index, the Section 20 cross-feature — all causal, all verified so by
  the same mutate-the-future test as the indicators
- build prediction labels for every configured horizon (future return,
  UP/DOWN, UP/NEUTRAL/DOWN, and take-profit-before-stop-loss), stored in
  their own `labels` table kept physically apart from features, with
  automated tests that a label depends only on bars after T while every
  feature depends only on bars at or before T
- score six baseline direction predictors (buy-and-hold, previous
  direction, moving average, RSI, random, majority class) against the
  labels on one shared chronological test segment, with a purge between
  train and test, abstention-aware coverage, the accuracy a constant guess
  would have scored on the same bars, and a conservative plain-language
  verdict — each run recorded in `model_runs` with its seed and package
  versions
- train Logistic Regression, Random Forest, Gradient Boosting and
  HistGradientBoosting classifiers on scale-free features with a purged
  chronological train/validation/test split, choose hyperparameters on
  validation only, calibrate probabilities (Platt), and score all of them on
  an untouched test segment against the baselines — with a significance bar
  tightened for the number of models tried, permutation importance, and
  saved, versioned, reproducible model files
- run a CLI with `init`, `health`, `validate-config`, `init-db`,
  `provider-demo`, `import-csv`, `market-status`, `record-indices`,
  `record-stocks`, `run-collector`, `build-candles`, `build-indicators`,
  `build-features`, `build-labels`, `evaluate-baselines`, and `train`
  commands, plus
  `scripts/collect_market_data.py` and `run_collector.bat` for
  Windows-native startup
- pass its own test suite (323 tests total: 8 Phase 1, 36 Phase 2, 9 Phase 3,
  11 Phase 4, 19 Phase 5, 18 Phase 6, 12 Phase 7, 25 Phase 8, 6 Phase 9,
  15 Phase 10, 34 Phase 11, 30 Phase 12, 19 Phase 13, 36 Phase 14, 45 Phase 15;
  16 of these — test_database.py and test_csv_importer.py — need SQLAlchemy and could
  not be run in the build sandbox)

Anyone reading this repo at Phase 1 should see an empty, well-organized skeleton —
not a fake trading bot. No BUY/SELL output, no synthetic "predictions", and no
placeholder functions pretending to be real logic exist in this codebase.

---

## 1. Architecture Overview

The system is organized as a strict pipeline with hard separation between stages.
Nothing downstream is allowed to reach back and mutate an upstream stage's data,
and no stage is allowed to depend on a concrete implementation of another —
everything talks through interfaces defined in this skeleton and built out phase
by phase.

```
PSX / permitted data source
        |
        v
 DATA PROVIDER ABSTRACTION   (Phase 4/7)  -> MockProvider / CSVProvider / PSXProvider
        |
        v
 COLLECTOR + VALIDATOR       (Phase 5/6)  -> raw/ (immutable) -> cleaned/
        |
        v
 CANDLE ENGINE                (Phase 10)  -> 1m/5m/15m/30m/60m/daily OHLCV
        |
        v
 INDICATOR + FEATURE ENGINE   (Phase 11/12) -> technical + KSE100/KMI30 regime features
        |
        v
 LABELING (no look-ahead)     (Phase 13)  -> shifted future targets, leakage-tested
        |
        v
 BASELINE MODELS              (Phase 14)  -> buy&hold, MA, RSI, prev-direction, random
        |
        v
 ML MODELS                    (Phase 15)  -> LogReg / RandomForest / GradientBoosting
        |
        v
 SIGNAL ENGINE                (Phase 18)  -> BUY/HOLD/SELL, always with reasons+warnings
        |
        v
 RISK MANAGER                 (Phase 19)  -> position sizing, exposure/drawdown limits
        |
        v
 ORDER MANAGER
        |
   +----+----------------------+
   v                           v
 PAPER BROKER (Phase 20)   REAL BROKER (Phase 26, gated by LIVE_TRADING_ENABLED)
        |
        v
 PORTFOLIO / REPORTS / DASHBOARD (Phase 21-23)
```

Design rules that hold for the whole project, not just Phase 1:

- **DATA, ANALYSIS, PREDICTION, SIGNAL, BACKTEST, PAPER TRADE, and LIVE TRADE are
  separate concerns.** They are never merged into one function or one class.
- **Raw data is never overwritten.** Cleaned/derived data always lives in a
  separate directory tree from raw data.
- **No component may use future information.** Every feature/label boundary is
  unit-tested for leakage once those components exist (Phase 12/13/58/101).
- **Live trading is off by default and stays off** until `LIVE_TRADING_ENABLED=true`
  *and* every other safety check in Section 66 of the working spec passes. This
  switch is declared in `.env.example` starting now, in Phase 1, even though
  nothing reads it yet — so it is never introduced as an afterthought.

---

## 2. Full Target Project Structure

This is the **eventual** full structure (all 26 phases). Phase 1 creates the
directory skeleton and the files marked `[P1]`. Everything else is an empty,
version-controlled placeholder directory (via `.gitkeep`) or does not exist yet
— it will be created, with real working code, in the phase that owns it.

```
psx_ai_trader/
    README.md                      [P1]
    requirements.txt                [P1]
    .env.example                    [P1]
    .gitignore                      [P1]

    config/
        settings.yaml               [P2]
        symbols.yaml                [P2]
        market_calendar.yaml        [P8: sessions/holidays/special sessions, all data-driven]
        costs.yaml                  (Phase 29)

    data/
        raw/                        [P1 dir only]
        cleaned/                    [P1 dir only]
        candles/                    [P1 dir only]
        features/                   [P1 dir only]
        predictions/                [P1 dir only]
        reports/                    [P1 dir only]

    database/                       [P3: schema + engine + repository]

    logs/                           [P1 dir + live logger]

    models/
        trained/                    [P1 dir only]
        scalers/                    [P1 dir only]
        metadata/                   [P1 dir only]

    src/
        main.py                     [P1-P4: init/health/validate-config/init-db/provider-demo]
        config/loader.py            [P2: full validation]
        database/
            models.py                [P3: 14 tables; P13 added `labels` = 15]
            engine.py                 [P3: engine/session factory]
            repository.py              [P3: duplicate-safe inserts]
        data/
            base_provider.py          [P4: DataProvider interface]
            mock_provider.py           [P4: synthetic random-walk provider]
            csv_mapping.py              [P5: pure header detection/parsing]
            csv_importer.py              [P5: CSV -> raw + cleaned tables]
            historical_provider.py                          (Phase 5, deferred — see note)
            psx_provider.py            [P7: licensed-feed adapter, credential-gated]
            validator.py               [P6: VALID/SUSPECT/INVALID/DUPLICATE quality engine]
            collector.py               [P8/P9: collect_indices_once/collect_stocks_once/collect_once]
            live_collector.py           [P9: reconnect/backoff/heartbeat/graceful-shutdown loop]
            candle_engine.py             [P10: pure OHLCV bucketing + DB orchestration]
            normalizer.py                                       (only if a later phase needs it)
        market/
            trading_calendar.py       [P8: sessions/holidays/next_open/next_close/etc.]
            sessions.py / symbols.py                        (only if a later phase needs them)
        indicators/
            trend.py                  [P11: SMA, EMA, MACD, ADX]
            momentum.py                [P11: RSI, ROC, Momentum, Stochastic]
            volatility.py               [P11: ATR, Bollinger Bands, rolling volatility]
            volume.py                    [P11: OBV, VWAP, volume MA, distance from VWAP]
            engine.py                     [P11: candles -> indicators -> Feature rows]
        features/
            market_regime.py          [P12: index returns/volatility/trend/position vs MA]
            stock_features.py          [P12: log returns, volume dynamics, breakouts, gap, relative perf]
            engine.py                   [P12: align stock+index candles -> Feature rows]
            labels.py                   [P13: future return, direction, 3-class, TP-before-SL]
            label_engine.py              [P13: candles -> labels table]
        models/
            baseline.py                [P14: six baseline direction predictors]
            evaluation.py               [P14: metrics, chronological split, verdicts]
            baseline_runner.py           [P14: comparison report + model_runs recording]
            dataset.py                  [P15: scale-free feature matrix + purged chronological split]
            ml_models.py                 [P15: LR / RF / GB / HGB, small grids, Platt calibrator]
            ml_metrics.py                 [P15: AUC, PR-AUC, Brier, log loss, calibration table, ECE]
            model_manager.py               [P15: model IDs, joblib + metadata JSON, version checks]
            trainer.py                      [P15: train/compare/store; baseline comparison]
        backtest/...                                     (Phase 16/17)
        signals/...                                       (Phase 18)
        paper/...                                          (Phase 20)
        risk/...                                           (Phase 19)
        reports/...                                         (Phase 23)
        dashboard/app.py                                    (Phase 21)
        utils/
            logger.py               [P1: real, 5 streams]
            file_utils.py            [P1: real]
            time_utils.py                             (Phase 8, tz work)

    tests/
        test_project_structure.py   [P1]
        test_logger.py               [P1]
        test_config_loader.py         [P2]
        test_database.py               [P3]
        test_mock_provider.py           [P4]
        test_csv_mapping.py              [P5]
        test_csv_importer.py              [P5/P6: rewired to use the quality engine]
        test_quality_engine.py             [P6]
        test_psx_provider.py                [P7]
        test_trading_calendar.py             [P8]
        test_collector.py                     [P8/P9: indices+stocks+collect_once]
        test_live_collector.py                 [P9]
        test_candle_engine.py                    [P10: pure bucketing/aggregation logic]
        test_candle_engine_db.py                  [P10: DB orchestration via fake repository]
        test_indicators_no_lookahead.py             [P11: Section 58's mandatory leakage test, per indicator]
        test_indicators_values.py                    [P11: correctness vs hand-calculated values]
        test_indicators_engine_db.py                   [P11: DB orchestration via fake repository]
        test_features_no_lookahead.py                    [P12: mutate-the-future test per feature]
        test_features_values.py                           [P12: correctness vs hand-calculated values]
        test_features_engine_db.py                         [P12: DB orchestration via fake repository]
        test_labels.py                                      [P13: label values + leakage tests]
        test_label_engine_db.py                              [P13: labels table orchestration via fake repository]
        test_baselines.py                                     [P14: predictors + causality checks]
        test_evaluation.py                                     [P14: metrics, split/purge, verdicts]
        test_baseline_runner.py                                 [P14: report, no-leakage, false-positive guard, model_runs]
        test_ml_dataset.py                                       [P15: scale-free proof, no-lookahead, purged split]
        test_ml_metrics_and_models.py                             [P15: metrics, models, calibration, selection]
        test_trainer.py                                            [P15: leakage controls, noise guard, signal/regime checks]
        test_model_manager.py                                       [P15: IDs, save/load round-trip, train_and_store]
        (all other test_*.py files added as their phase lands)

    scripts/
        collect_market_data.py       [P9: thin wrapper for run-collector]

    install.bat                     [P1]
    run_collector.bat              [P9]
    run_dashboard.bat / etc.        (added as their phase lands)
```

---

## 3. Technology Decisions (and why)

| Choice | Reason |
|---|---|
| Python 3.10+ | Broad library support, type hints, still light on an old Windows box |
| SQLite (Phase 3) | Zero-install, file-based, fine for the target 2-core/6GB machine; migration path to Postgres/Parquet documented if data outgrows it (Section 69/70) |
| Pandas / NumPy (Phase 10+) | Standard, well-optimized vectorized operations; avoids hand-rolled loops on a slow CPU |
| Scikit-learn (Phase 15) | CPU-only, no CUDA requirement, covers LogReg/RandomForest/HistGradientBoosting without heavy install |
| PyYAML (Phase 2) | Human-editable configuration without touching source code, per the symbol/cost/settings requirements |
| FastAPI or Flask (Phase 21) | Lightweight local dashboard, binds to 127.0.0.1 only by default |
| pytest (from Phase 1) | Standard, fast, supports the leakage/backtest/risk test suite required throughout |
| stdlib `logging` (from Phase 1) | No extra dependency for something this fundamental; rotating file handlers are sufficient for a single-machine deployment |

No dependency is added before the phase that needs it. Phase 1's
`requirements.txt` intentionally contains only `pytest` — everything else is
added, with justification, as its owning phase lands.

---

## 4. Development Roadmap (26 Phases)

1. **Project skeleton** — done
2. **Configuration** (`settings.yaml`, `symbols.yaml`, loader + validation) — done
3. **Database** (SQLite schema: ticks, candles, features, predictions, signals, orders, positions, portfolio snapshots, model runs, backtest runs, system events) — done
4. **Mock data provider** (synthetic, clearly labeled `SYNTHETIC`) — done
5. **Historical CSV importer** — done
6. **Data validation / quality engine** (VALID/SUSPECT/INVALID/DUPLICATE) — done
7. **PSX provider abstraction** (researched against current official PSX data-access options — no undocumented endpoints, no scraping around access controls) — done
8. **KSE-100 / KMI-30 recording + trading calendar** — done
9. **Stock recording** (+ the robust continuous collector loop) — done
10. **Candle engine** (1m/5m/15m/30m/60m/daily) — done
11. **Technical indicators** (SMA/EMA/RSI/MACD/ATR/ADX/Bollinger/etc., configurable periods) — done
12. **Feature engine** (stock + KSE100/KMI30 regime features) — done
13. **Labels** (shifted-forward targets; leakage tests) — done
14. **Baseline models** (buy&hold, MA, RSI, previous-direction, random, majority) — done
15. **ML models** (LogReg, RandomForest, GradientBoosting/HistGB) — done
16. Backtest engine (event-driven, costs, slippage, no look-ahead) ← *you are here*
17. Walk-forward validation (chronological splits only)
18. Signal engine (BUY/HOLD/SELL with explicit reasons + warnings, never a bare label)
19. Risk manager (position sizing, exposure/drawdown limits)
20. Paper broker (shares the exact same signal→risk→order pipeline live trading will use)
21. Dashboard (localhost only)
22. Alerts (desktop/terminal/log first; email/Telegram optional later)
23. Daily reports (HTML/CSV)
24. Model monitoring (drift, rolling accuracy/profit factor, calibration)
25. Advanced/optional models (ensembles, regime-specific models — never required)
26. Optional broker integration (only via an officially documented API, gated by `LIVE_TRADING_ENABLED`)

Each phase is delivered complete with code, tests, run instructions, expected
output, and a stated list of limitations before moving to the next. Say
**PROCEED** when ready for Phase 2.

---

## 5. Phase 1 — What Was Actually Built

- Full directory tree (`data/`, `database/`, `logs/`, `models/`, `src/`, `tests/`, `scripts/`, `config/`)
- `src/utils/logger.py` — real logging setup producing 5 independent log files
  (`logs/system.log`, `data.log`, `model.log`, `trading.log`, `error.log`) plus
  console output, with rotation so logs don't grow unbounded on a small disk.
- `src/utils/file_utils.py` — project-root discovery and idempotent directory
  creation helpers used by every later phase (so "where do files live" is
  answered once, here, not re-implemented per module).
- `src/main.py` — CLI with two real commands:
  - `init` — creates the full directory tree (idempotent) and confirms `.env`
    exists (copies from `.env.example` if missing), then logs and prints a
    result summary.
  - `health` — checks Python version, directory presence/writability, `.env`
    presence, and that `pytest` is installed. This is intentionally limited:
    it cannot yet check the database, data provider, or model files, because
    those don't exist until later phases. It says so explicitly rather than
    faking a green check.
- `.env.example` — declares `LIVE_TRADING_ENABLED=false` from day one, per the
  project's non-negotiable safety rule, even though nothing reads it yet.
- Test suite (`tests/test_project_structure.py`, `tests/test_logger.py`)
  covering directory creation and log-file creation/levels.

No configuration loader, no database, no data provider, no indicators, no
models, and no signals exist yet — building any of those now would violate the
phase-by-phase rule and risk exactly the kind of impressive-looking-but-hollow
system this project is explicitly trying to avoid.

---

## 5b. Phase 2 — What Was Actually Built

- `config/settings.yaml` — the single editable file for market timezone,
  data-provider selection, collector reconnect/stale-timeout behavior,
  prediction horizons, risk limits, trading mode, dashboard binding, and log
  level. Every key is commented with what it does and which phase actually
  consumes it.
- `config/symbols.yaml` — the single editable file for the index/stock/
  watchlist universe. No symbol is hardcoded anywhere in `src/`.
- `src/config/loader.py` — `load_config()` reads both YAML files and `.env`,
  validates every field, and returns an immutable `AppConfig`. Validation is
  **exhaustive, not fail-fast**: every problem in the file is collected and
  reported together via `ConfigValidationError`, so fixing config is one pass,
  not trial-and-error.
- **Safety enforcement now, not deferred to Phase 26**: if
  `trading.mode: LIVE` is set in `settings.yaml` but `LIVE_TRADING_ENABLED`
  is not `true` in `.env`, `load_config()` raises immediately. The two files
  must agree before the app will even finish loading its configuration —
  long before any order-placement code exists.
- Dashboard binding is validated against the localhost-only rule (Section 68
  of the working spec): a non-localhost `host` without `dashboard.allow_external: true`
  is rejected at load time, not silently allowed.
- `src/main.py` gained a `validate-config` command that loads config and
  prints every resolved value (there are no secrets in these two YAML files,
  so nothing is masked), and the `health` command now includes a
  "Configuration valid" check.
- `requirements.txt` grew by exactly two entries — `PyYAML` and
  `python-dotenv` — both justified by this phase's actual need.
- 12 new tests in `tests/test_config_loader.py` cover: valid load, missing
  file, invalid timezone, invalid provider, invalid horizon format, risk
  bounds, LIVE mode without the env flag (rejected), LIVE mode with the flag
  (accepted), dashboard host/allow_external interaction, duplicate symbols,
  unknown watchlist entries, and multi-error aggregation.

Still nothing that touches a database, a data provider, an indicator, a
model, or a signal — that discipline continues through Phase 2 exactly as it
did in Phase 1.

---

## 5c. Phase 3 — What Was Actually Built

- `src/database/models.py` — SQLAlchemy ORM definitions for all 14 tables
  from the project spec (`market_ticks`, `index_ticks`, `stock_ticks`,
  `candles`, `features`, `predictions`, `signals`, `orders`, `fills`,
  `positions`, `portfolio_snapshots`, `model_runs`, `backtest_runs`,
  `system_events`), matching the DATA / ANALYSIS / PREDICTION / SIGNAL /
  ORDER / RESEARCH / OPERATIONS grouping the project keeps everywhere else.
- **Raw vs. cleaned split is enforced by the schema, not just by convention**:
  `market_ticks` is an append-only raw log (never deduplicated, never
  edited); `index_ticks` / `stock_ticks` are the cleaned, typed tables a
  future Phase 6 validator will populate, and *those* carry the
  `UniqueConstraint("provider", "symbol", "timestamp")` that makes replayed
  data a safe no-op instead of a duplicate row.
- `src/database/engine.py` — the only module that calls
  `sqlalchemy.create_engine`; every other module goes through
  `get_session_factory()`. Enables `PRAGMA foreign_keys=ON` on every
  connection, because SQLite silently ignores foreign keys otherwise.
- `src/database/repository.py` — duplicate-safe insert helpers for
  `index_ticks`, `stock_ticks`, and `candles` using SQLite's
  `INSERT ... ON CONFLICT DO NOTHING`, each returning whether a row was
  actually inserted (so a future collector can count/log duplicates instead
  of pretending they didn't happen). Plain append-only inserts for
  `market_ticks` and `system_events`.
- `src/main.py` gained an `init-db` command (creates the `.db` file and all
  tables, logs a `system_events` row recording the initialization) and the
  `health` command now opens the database, confirms every expected table
  exists, and reports the current `system_events` count.
- `config/settings.yaml` / `src/config/loader.py` gained a `database:`
  section (`path`, `echo`), validated the same way every other section is
  (non-empty `.db` path, boolean `echo`).
- `requirements.txt` grew by exactly one entry: `SQLAlchemy`.
- 9 new tests in `tests/test_database.py`: every table gets created,
  duplicate inserts are correctly ignored for `index_ticks`/`stock_ticks`/
  `candles`, `market_ticks`/`system_events` are correctly *not* deduplicated,
  a foreign-key violation on `fills.order_id` is correctly rejected, a valid
  fill against a real order succeeds, and `model_runs.model_id` correctly
  enforces uniqueness.

**A note on how this phase was verified.** This sandbox has no network
access, so `SQLAlchemy` could not be `pip install`-ed here to run
`pytest` directly against the ORM code. Every file was syntax-checked
(`python -m py_compile`), reviewed by hand against SQLAlchemy 2.0's
declarative-mapping API, and the two SQLite behaviors the repository layer
depends on — `INSERT ... ON CONFLICT DO NOTHING` correctly reporting
`rowcount` for new-vs-duplicate rows, and `PRAGMA foreign_keys=ON` actually
rejecting a bad reference — were confirmed directly against this machine's
SQLite via the stdlib `sqlite3` module. Run `pip install -r requirements.txt`
followed by `pytest -q` on your machine to execute the full ORM-level suite;
if anything fails there, report it and it will be fixed before Phase 4, not
carried forward.

Still nothing that touches a data provider, an indicator, a model, or a
signal.

---

## 5d. Phase 4 — What Was Actually Built

- `src/data/base_provider.py` — the `DataProvider` abstract interface every
  concrete provider (mock now; CSV/PSX/broker later) implements:
  `connect()` / `disconnect()` / `is_connected()` / `get_index_observation()`
  / `get_stock_observation()`, usable as a context manager. Also defines
  `IndexObservation` / `StockObservation`, provider-agnostic value objects
  (deliberately separate from the `src/database/models.py` ORM classes —
  a provider should never need to know how, or whether, its output gets
  persisted).
- `src/data/mock_provider.py` — `MockProvider`, a seeded random walk per
  symbol. Every single observation it returns carries `is_synthetic=True`,
  `provider="mock"`, and `market_status="SYNTHETIC_OPEN"` — there is no code
  path in this provider that can produce output indistinguishable from real
  data, by construction, not by convention.
- Reproducibility: the same seed produces an identical sequence of
  observations (verified by test), which is what will make backtesting
  demonstrations and dashboard development reproducible during later phases.
- `src/main.py` gained a `provider-demo` command that connects the
  configured provider (mock only, for now) and prints one observation per
  symbol from `config/symbols.yaml`, with a `SYNTHETIC` banner and a
  reminder that this is a demonstration, not the real collector. `health`
  now includes a "Data provider (mock)" check.
- No new dependencies — `mock_provider.py` uses only `random`, `datetime`,
  and `dataclasses` from the standard library.
- 11 new tests in `tests/test_mock_provider.py`: calling a method before
  `connect()` raises, connect/disconnect/context-manager all toggle state
  correctly, every observation is correctly labeled synthetic, index/stock
  values are plausible (high ≥ value ≥ low, volume ≥ 0), stock high/low
  correctly track running extremes across calls, cumulative volume never
  decreases, identical seeds produce identical sequences, different seeds
  diverge, an unrecognized symbol falls back to a sane default instead of
  crashing, and custom base prices are honored exactly on the first call.

**Verification note.** Unlike Phases 2 and 3, this phase's code has zero
external dependencies (stdlib only), so I ran the full test suite for real
in this sandbox (via a lightweight `pytest.raises` shim, since `pytest`
itself still can't be installed here without network access) — **11/11
passed** — and exercised `provider-demo` and the new `health` check
end-to-end against the actual CLI, both of which worked exactly as shown
below.

Still no CSV importer, no real PSX connectivity, no data validation engine,
no candles, no indicators, no models, and no signals.

---

## 5e. Phase 5 — What Was Actually Built

- `src/data/csv_mapping.py` — pure column-detection and value-parsing logic,
  **zero dependency on the database or SQLAlchemy**. Given a CSV header row,
  it recognizes common spellings (`Date`/`Datetime`/`Timestamp`,
  `Close`/`LTP`/`Last`, `Change %`/`Chg Pct`, ...) for both stock and index
  files. If a required field (timestamp, and price) can't be determined, it
  raises `CSVMappingError` listing the detected headers — it never guesses
  silently, consistent with Section 3's "do not assume ... exists" rule.
  An explicit `--mapping` JSON file always overrides auto-detection.
- `src/data/csv_importer.py` — `CSVImporter.import_file()` reads a file row
  by row and, for every row that parses:
  1. writes an immutable `MarketTick` (the raw row, exactly as read, as
     Section 11 requires — never touched again after this)
  2. writes a `IndexTick`/`StockTick` through the same duplicate-safe
     repository functions from Phase 3, so re-importing the same file twice
     is a safe no-op, not a pile of duplicate rows.
  Validation is intentionally structural only — parseable timestamp,
  positive price, high ≥ low, non-negative volume — because the full
  VALID/SUSPECT/INVALID/DUPLICATE quality engine (Section 13) is Phase 6's
  job, not this one's. A row that fails structural validation is **never**
  written anywhere, including as a raw `MarketTick` — only rows we could
  actually parse into a coherent observation get persisted at all.
- `src/main.py` gained `import-csv --file ... --symbol ... --symbol-type
  {stock,index} [--provider-name NAME] [--mapping mapping.json]`, matching
  Section 95's required output: rows read, accepted, duplicate, and invalid
  counts, plus the first 10 invalid-row reasons inline (the rest go to
  `logs/data.log`). Every import also writes a `system_events` row
  summarizing the run.
- **A deliberate scope cut, stated plainly**: the project structure lists a
  `csv_provider.py` implementing the `DataProvider` streaming interface
  (for a future historical-replay backtester, Section 75/76). That is not
  built yet — nothing in Phase 1-17 needs it, and building it now would be
  speculative code with no caller, which the project's phase discipline
  explicitly rules out. It will be added in the phase that actually
  consumes it.
- 17 new tests: 12 in `tests/test_csv_mapping.py` (header normalization,
  auto-detection for both stock and index aliases, missing-required-field
  errors, explicit override, float/timestamp parsing including timezone
  localization to `Asia/Karachi` and preserving an explicit offset) and 5 in
  `tests/test_csv_importer.py` (full read/accepted/duplicate/invalid
  bucketing against a deliberately messy fixture file, raw+cleaned tables
  both populated correctly, explicit-mapping import, missing-column
  failure, and full idempotency on a second import of the same file).

**Verification.** `csv_mapping.py` has no database dependency, so I ran its
12 tests for real in this sandbox (via the same `pytest.raises` shim used in
earlier phases) — **12/12 passed**. `csv_importer.py` needs SQLAlchemy,
which still can't be installed here (no network), so instead I hand-replayed
the exact classification logic in `tests/test_csv_importer.py`'s fixture CSV
using the real `csv_mapping.py` functions and confirmed the expected
counts — accepted=2, duplicate=1, invalid=4 out of 7 rows read — match what
the test asserts. I also confirmed end-to-end that `import-csv` fails
honestly (not a crash) when SQLAlchemy is unavailable. Run
`pip install -r requirements.txt && pytest -q` to execute the full 19-test
Phase 5 suite for real.

Still no real PSX connectivity, no data-quality engine (VALID/SUSPECT/
INVALID/DUPLICATE), no candles, no indicators, no models, and no signals.

---

## 5f. Phase 6 — What Was Actually Built

- `src/data/validator.py` — `DataQualityEngine`, the real implementation of
  Section 13. Given one `QualityInput` (symbol, timestamp, price, OHLC,
  volume, market_status) and the symbol's prior state, it returns a
  `QualityResult(status, reasons)` where `status` is one of `VALID`,
  `SUSPECT`, `INVALID`, or `DUPLICATE` — **it never deletes or mutates
  anything itself**, per Section 13's explicit rule. Checks implemented:
  unknown symbol, non-positive price, impossible OHLC (high < low, or
  outside the open), negative volume, far-future timestamp, and
  out-of-order timestamp all return `INVALID`; an exact timestamp repeat
  returns `DUPLICATE`; an unrecognized market-session string, an extreme
  single-tick percentage move, and a price that hasn't changed for longer
  than `quality.stale_seconds` all return `SUSPECT`. An `INVALID` or
  `DUPLICATE` observation deliberately does **not** update the symbol's
  baseline, so one bad row can't drag every subsequent comparison off
  course.
- `config/settings.yaml` / `loader.py` gained a `quality:` section
  (`max_change_percent`, `stale_seconds`, `future_timestamp_tolerance_seconds`),
  validated like everything else. The "gap since last observation" check
  deliberately reuses `collector.stale_timeout_seconds`, which already
  existed for the live collector, instead of adding a redundant setting.
- `src/data/csv_importer.py` was **rewired**, not left alongside the engine:
  it no longer does its own ad-hoc bad-OHLC/negative-price/negative-volume
  checks (that was Phase 5's narrower, honestly-labeled stand-in). Every
  parseable row now goes through `DataQualityEngine`, and — this is the
  real behavior change — **rows that fail validation are now stored, not
  skipped**, tagged with whatever status the engine assigned. Only a row
  with no determinable timestamp at all is impossible to store (there's no
  key to store it under); everything else, however bad, gets a row and a
  reason. Every row physically read from the file also now gets a raw
  `MarketTick`, including that no-timestamp case — raw preservation is
  unconditional.
- A real bug caught and fixed during this phase: the original stale-price
  logic only started its "unchanged since" clock on the *second* repeated
  price, so a price that went stale exactly on the observation where the
  repeat began was never flagged — off by one reading. Caught by a test
  that asserts staleness after exactly one repeat past the threshold; fixed
  before shipping.
- 18 new tests in `tests/test_quality_engine.py` (pure logic, no database),
  covering every check above individually, the baseline-not-updated-on-
  INVALID/DUPLICATE behavior, and the `IndexObservation`/`StockObservation`
  convenience wrappers. `tests/test_csv_importer.py` was rewritten to match
  the new behavior: invalid rows are now asserted to be *written* (with the
  correct status and reason) rather than absent, and a new test confirms an
  unrecognized symbol is stored flagged `INVALID` rather than silently
  rejected.

**Verification.** `validator.py` has no database dependency, so its 18
tests ran for real in this sandbox — **18/18 passed**, including the one
that caught the off-by-one bug above before it could ship. The CSV
integration tests still need SQLAlchemy (unavailable here, no network), so
I re-derived the expected row-by-row classification for the fixture CSV
using the actual `validator.py` + `csv_mapping.py` code and confirmed it
against every assertion in `test_csv_importer.py` line by line before
finalizing them — see the comment block above `STOCK_CSV` in that file.

Still no real PSX connectivity, no candles, no indicators, no models, and
no signals.

---

## 5g. Phase 7 — What Was Actually Built

**Research done before writing any code, per Sections 3 and 116.** Web
searches against official and authoritative sources established:

1. Pakistan Stock Exchange (PSX, `psx.com.pk`) does not publish a free,
   self-service developer API for real-time or historical licensed market
   data. (Not to be confused with Nasdaq's unrelated "PSX" — the
   Philadelphia Stock Exchange — which turns up in generic searches for the
   same three letters.)
2. PSX's public Data Portal (`dps.psx.com.pk`) is free for an individual to
   *browse* — current-session data, EOD data, announcements, a screener —
   but the portal itself carries an explicit legal notice (read directly on
   the site): automated dissemination or commercial use of its market data
   "through any mode of communication, including ... electronic
   communications ... without acquiring respective rights/license from PSX
   is strictly prohibited," with PSX reserving the right to pursue legal
   action.
3. Real-time and licensed historical feeds (Level 1, Level 2/tick, EOD) are
   sold as commercial products through PSX's own "Data Services Vending"
   program and through PSX-authorized third-party vendors. The documented
   licensing contact is `marketdatarequest@psx.com.pk`.
4. Several open-source projects on GitHub/PyPI scrape `dps.psx.com.pk`
   directly, unlicensed. This project does not follow that example —
   Section 3 is explicit that scraping and bypassing licensing/access
   controls are off the table "regardless of how the request is framed,"
   and that includes not treating "other projects do it" as license to do
   the same.

**What got built, given that.** `src/data/psx_provider.py` — `PSXProvider`,
a fully-implemented `DataProvider`, but a **licensed-feed adapter, not a
scraper**:
- `connect()` refuses outright — raising `PSXLicenseRequiredError` with the
  full explanation and the licensing contact — unless `PSX_DATA_API_BASE_URL`
  and `PSX_DATA_API_KEY` are set (from a real PSX license or an authorized
  vendor's subscription). It never falls back to touching
  `dps.psx.com.pk`, licensed or not.
- Once configured, it speaks a minimal, generic HTTP contract this project
  defines (`GET {base_url}/health`, `GET {base_url}/index/{symbol}`,
  `GET {base_url}/stock/{symbol}`, JSON matching `IndexObservation`/
  `StockObservation`) — because no specific commercial vendor's actual API
  shape is known without reading that vendor's own documentation, which is
  outside this project's current scope. Whoever licenses real data fills
  this contract in, either because their vendor already speaks it or via a
  small translation shim in front of the vendor's real API.
- Missing fields in a vendor's response become `None`, never fabricated,
  matching every other provider in this project.
- `.env.example` documents exactly what's needed and why, and
  `config/settings.yaml`'s `data.provider` enum already accepted `"psx"`
  since Phase 2 — no config schema change was needed here.
- `src/main.py`'s `health` and `provider-demo` now branch on
  `data.provider`: `mock` behaves as before; `psx` attempts a real
  connection through the adapter above and reports the licensing-required
  message (or real data) accordingly, never silently falling back to mock
  behavior.

**Verification.** 12 new tests in `tests/test_psx_provider.py`, all fully
offline — no real network access is used or possible in this sandbox, and
none is needed: `connect()`/`get_*_observation()` are exercised against a
mocked `urllib.request.urlopen`, covering the license-required path,
environment-variable loading, successful connection, unreachable-endpoint
handling, response parsing for both observation types, missing-field
handling, and the optional secret header. All 12 passed. Beyond the mocked
unit tests, I additionally stood up a real local HTTP server on
`127.0.0.1` (not a real network call — pure loopback) and ran `PSXProvider`
against it end-to-end to confirm the whole request/parse path works over
actual sockets, not just mocks. I also ran the CLI itself both ways: with
`data.provider: psx` and no credentials (`provider-demo` and `health` both
fail with the exact licensing explanation, exit code 1) and confirmed
nothing in that path ever constructs a request to any real PSX-related
domain.

Still no trading calendar, no candles, no indicators, no models, and no
signals. Real PSX connectivity is now a **configuration step for whoever
has a license**, not a missing-code problem.

---

## 5h. Phase 8 — What Was Actually Built

**Research done before writing any code, per Sections 10/14/116.** Fetched
directly from PSX's own site (not a third-party aggregator):

- Trading hours: `https://www.psx.com.pk/psx/exchange/general/trading-hours`
  — Monday-Thursday runs pre-open 09:15-09:30, order-matching break
  09:30-09:32, continuous trading 09:32-15:30, post-close 15:35-15:50.
  Friday splits into two sessions around the Friday prayer break:
  09:00-09:17 pre-open/break then continuous 09:17-12:00, then
  14:15-14:32 pre-open/break then continuous 14:32-16:30 and post-close
  16:35-16:50.
- Calendar holidays:
  `https://www.psx.com.pk/psx/exchange/general/calendar-holidays` — PSX's
  full official 2026 list (17 published rows; two coincide on the same
  calendar date as another row, so 15 unique closed dates), each tagged
  moon-dependent or fixed. PSX itself flags Islamic-calendar dates as
  "Subject to appearance of Moon" and reserves the right to change any
  date — that caveat is preserved verbatim in the shipped config.

**What got built.**
- `config/market_calendar.yaml` — **every PSX-specific time and date lives
  here, not in Python.** Session windows, the full 2026 holiday list, and
  an (empty-by-default, worked-example) mechanism for date-ranged special
  sessions like Ramadan's shortened hours — which PSX announces separately
  each year and cannot be predicted in advance. The file cites its two
  source URLs and the fetch date, and says explicitly that it needs
  re-fetching periodically — this is the correct way to satisfy Section
  14's "do not hardcode assumptions" rule for something that itself expires.
- `src/market/trading_calendar.py` — `load_market_calendar()` (same
  exhaustive-validation pattern as every other loader in this project: bad
  time formats, inverted windows, overlapping session blocks, duplicate
  holiday dates, and overlapping special-session ranges are all caught,
  all at once) and `TradingCalendar`, implementing exactly Section 14's
  required surface: `is_market_open()`, `is_market_closed()`,
  `is_pre_open()`, `is_post_market()`, `current_session()`, `next_open()`,
  `next_close()`, `is_holiday()`. Every method accepts an optional `at:
  datetime` so behavior is fully testable without depending on the wall
  clock; naive datetimes are localized to the configured market timezone
  rather than silently assumed to already be correct.
- `src/data/collector.py` — `collect_indices_once()`, the first piece that
  actually wires a `DataProvider`, the `DataQualityEngine`, and the
  database together for real, gated by `TradingCalendar.is_market_open()`.
  It refuses to record when the market is closed unless explicitly
  `force`d (for testing with the mock provider — forcing it with a real
  feed would just store meaningless off-hours reads). This is deliberately
  a single "collect once" function, not the continuous poll/reconnect/
  backoff loop Section 15 describes — that robustness layer is Phase 9's
  job, built around this once it also covers stocks.
- `src/main.py` gained `market-status` (prints current session, holiday
  status, next open/close) and `record-indices [--force]`. `health` now
  includes a "Trading calendar" check.

**A design note on honesty over completeness.** I did not attempt to
compute Islamic-calendar (moon-sighting-dependent) holiday dates
algorithmically — that's not something a fixed calendar system can predict
correctly, and PSX's own table says so. The shipped 2026 dates are PSX's
official *estimates*, carried with their own caveat intact, not
presented as more certain than PSX itself claims they are.

**Verification.** I hand-traced the shipped calendar against twelve
distinct real timestamps — Monday pre-open/break/regular/post-close/late-
evening-closed, Friday's two sessions and the lunch gap, Saturday, Sunday,
and the Independence Day holiday — and every one matched the official
schedule exactly, including a live check that ran while writing this
phase (a Friday, correctly detected as mid-session-1 with the correct
next-open and next-close times). 25 new automated tests: 21 in
`tests/test_trading_calendar.py` (pure logic, no database — all 21 ran for
real and passed) and 4 in `tests/test_collector.py`, which exercises the
full `collect_indices_once()` orchestration — skip-when-closed, `--force`
override, per-symbol quality tagging, and DB-level duplicate handling —
using a fake in-memory repository module injected via `sys.modules`
(the same lazy-import pattern the rest of the project already uses for
SQLAlchemy independence), so all 4 also ran for real without SQLAlchemy
installed. I additionally ran the actual `record-indices` CLI path
end-to-end via the same fake-module technique and confirmed correct
output, correct exit codes, and correct database calls.

Still no candle engine, no indicators, no models, and no signals. Stock
recording and the robust continuous collector loop are next.

---

## 5i. Phase 9 — What Was Actually Built

- `src/data/collector.py` was extended, not just added to: the single-shot
  `collect_indices_once()` from Phase 8 now shares its DB-write logic with
  a new `collect_stocks_once()` via two small `_record_index_symbol()` /
  `_record_stock_symbol()` helpers, and a new `collect_once()` records
  indices *and* stocks together in one market-open check and one provider
  connection — this is what the continuous loop actually calls every
  cycle, not two separate calls.
- `src/data/live_collector.py` — `LiveCollector`, the Section 15 loop:
  - **Reconnect with exponential backoff**: failed connects retry at
    `reconnect_seconds, 2x, 4x, ...` capped at `8 × reconnect_seconds`
    (minimum 60s), not a fixed delay.
  - **Never crashes on one bad cycle**: any exception from a poll cycle is
    caught, logged, and the loop continues — Section 15's explicit
    requirement, verified by a test where the data feed fails on purpose.
  - **Graceful shutdown**: SIGINT/SIGTERM finish the current cycle,
    disconnect the provider cleanly, and exit — no mid-write state left
    behind.
  - **Outage recording (Section 71)**: the first failure of a run writes a
    `system_events` row (`connection_lost: <reason>`); the next success
    writes `connection_restored after <N>s outage` with the actual
    measured duration. This is the first phase where `system_events`
    (created back in Phase 3) actually gets used for something.
  - **Configurable retry limit**: after `collector.max_consecutive_failures`
    straight failures, a CRITICAL `system_events` row is written — but the
    collector keeps retrying with backoff regardless. It never gives up on
    its own; only the person running it decides that.
- `config/settings.yaml` / `loader.py` gained `collector.poll_interval_seconds`
  and `collector.max_consecutive_failures`, validated like everything else.
- `src/main.py` gained `record-stocks` (mirrors `record-indices`) and
  `run-collector [--max-cycles N]`. `record-indices`/`record-stocks` were
  refactored into one shared `_run_record()` rather than duplicated.
- `scripts/collect_market_data.py` and `run_collector.bat` — the
  Windows-native entry points Section 91/54 call for, both thin wrappers
  around `run-collector` so there is exactly one implementation to keep
  correct.

**A design choice worth explaining**: every source of real-world
nondeterminism in `LiveCollector` — the wall clock, `time.sleep`, and how
many cycles to run — is an injectable constructor argument. Production
code never passes them (real clock, real sleep, run forever); tests always
do. This is what makes a "runs forever" loop fully testable without either
mocking half the standard library inline or actually waiting in real time
— and it's also what let me verify backoff timing, outage-duration
calculation, and the CRITICAL-after-N-failures threshold exactly, not just
"it doesn't crash."

**Verification.** 6 new tests in `tests/test_live_collector.py`, using the
same fake-repository-module-injected-via-`sys.modules` technique as
`test_collector.py`, so all 6 ran for real, fully offline: N requested
cycles run and record correctly; a connect-failure-then-recovery sequence
produces the expected backoff sleep pattern; a mid-run feed failure
followed by success produces exactly one `connection_lost` and one
`connection_restored` event with the outage still correctly measured; N
consecutive failures produce exactly one CRITICAL event and the collector
keeps running past it; `request_stop()` halts the loop within one cycle;
and shutdown always disconnects the provider. Beyond the injected-clock
unit tests, I ran the actual `run-collector` CLI command for real (real
clock, real `time.sleep`, fake-module DB injection only) for 3 cycles at a
15-second poll interval — it correctly detected the sandbox's real weekend
and skipped recording each cycle, exactly as `market-status` did
independently in Phase 8.

Still no candle engine, no indicators, no models, and no signals. Candle
generation is next.

---

## 5j. Phase 10 — What Was Actually Built

- `src/data/candle_engine.py` — pure bucketing/aggregation logic
  (`build_intraday_candles`, `build_daily_candle`,
  `filter_regular_session_ticks`) with **zero database dependency**, plus
  one orchestration function (`build_candles_for_day`) that reads ticks
  and writes candles. Three decisions materially shape what a candle here
  means, all stated in the module's docstring, not buried:
  - **Only REGULAR-session ticks are used** — pre-open, pre-open-break, and
    post-close observations are excluded entirely (price and volume).
    Candles represent continuous-trading price action; auction-style
    pre-open/post-close prices behave differently and would silently
    distort OHLC if mixed in.
  - **INVALID ticks are excluded, SUSPECT ticks are not** — a tick the
    quality engine flagged as bad data (impossible OHLC, negative volume,
    unknown symbol) must never distort a derived candle; a merely unusual
    (SUSPECT) one is still a real observation.
  - **Volume convention is one global, explicit config setting**
    (`candles.volume_convention: cumulative | per_tick`), not inferred.
    Section 17 requires distinguishing per-tick from cumulative volume;
    this project has no way to detect that automatically from a feed, so
    the person configuring it states it, and a documented limitation
    covers the one edge case that can't be resolved without a field the
    schema doesn't have (pre-open volume potentially already baked into a
    real feed's cumulative counter by the first regular-session tick).
- `src/market/trading_calendar.py` gained one new public method,
  `regular_session_windows(date)`, so the candle engine (and anything
  else that needs it later) can get a day's actual session boundaries —
  correctly returning two windows on a Friday — without duplicating the
  calendar's own holiday/special-session logic.
- `src/database/repository.py` gained `get_index_ticks`/`get_stock_ticks`,
  which filter out `quality_status == 'INVALID'` at the query level, so
  the exclusion rule above is enforced once, in one place, not
  re-implemented by every caller.
- `config/settings.yaml`/`loader.py` gained the `candles.volume_convention`
  setting.
- `src/main.py` gained `build-candles --symbol S --date YYYY-MM-DD
  [--timeframes 1m,5m,...]`, which validates the symbol against
  `config/symbols.yaml`, builds all six timeframes by default, and reports
  written/duplicate counts per timeframe.
- **A duplication bug in `config/loader.py` and `config/settings.yaml`
  caught and fixed during this phase**: an editing collision from an
  earlier turn had left a duplicate `QualityConfig` class, a duplicate
  `_validate_quality` function, and a duplicate `quality:` YAML block in
  place (harmless — Python and YAML both silently let the second
  definition win — but dead weight and a real risk if the two copies had
  ever drifted apart). Found via a systematic duplicate-definition scan
  across every source file before starting Phase 10's own additions, and
  removed.

**Verification.** The bucketing/aggregation functions have no database
dependency, so all 10 tests in `tests/test_candle_engine.py` ran for real:
correct 5-minute bucket boundaries, correct OHLC extraction, per-tick
volume summing, cumulative-volume delta calculation (including the
backwards-counter/anomaly case, which returns `None` rather than a
fabricated or negative number), the `is_synthetic` flag correctly tracking
the `mock` provider, empty input producing no candles, and a Friday's two
sessions correctly producing one daily candle spanning session 1's open to
session 2's close. `build_candles_for_day`'s database orchestration was
exercised by 5 more tests in `tests/test_candle_engine_db.py` using the
same fake-repository-module-injected-via-`sys.modules` technique as the
Phase 8/9 collector tests — including one that plants a pre-open tick with
an obviously-wrong price (999.0) and confirms it never reaches the
resulting candle — so all 15 new tests ran for real, fully offline. I also
ran the actual `build-candles` CLI command end-to-end (fake DB modules,
real orchestration code) across all six timeframes and hand-verified the
cumulative-volume deltas by arithmetic against the injected fixture ticks.

Still no indicators, no models, and no signals. Technical indicators are
next.

---

## 5k. Phase 11 — What Was Actually Built

- `pandas`/`numpy` became real dependencies for the first time (both were
  already listed as "Phase 10+" placeholders in `requirements.txt` since
  Phase 1) — justified here: rolling-window and EWM math is exactly what
  they're for, and hand-rolling it in pure Python would be slower and more
  error-prone for no benefit.
- `src/indicators/trend.py`, `momentum.py`, `volatility.py`, `volume.py` —
  every indicator Section 18 lists: SMA, EMA, MACD, ADX, RSI, ROC,
  Momentum, Stochastic, ATR, Bollinger Bands, rolling volatility, OBV,
  VWAP, and distance from VWAP. Each function takes and returns plain
  `pandas.Series`/`DataFrame` — no config object, no database — so every
  one is independently testable.
- **Every indicator is causal by construction** (rolling/EWM operations
  only ever read data at or before the current index), and this is not
  just asserted — it's tested. Section 58 requires exactly this check
  ("modify future prices, recalculate, assert earlier values are
  unchanged, fail the test if they change"), and
  `tests/test_indicators_no_lookahead.py` runs it against all fifteen
  indicator outputs individually.
- `src/indicators/engine.py` — `compute_indicators()` (pure: candle
  DataFrame + config → one wide DataFrame, one column per indicator/
  period) and `compute_and_store_indicators()` (the only function that
  touches the database: reads candles via the Phase 10 pipeline, computes,
  and writes each value as a `Feature` row — reusing the exact long-format
  table Phase 3 built for this). A `NaN` value (not enough history yet) is
  never stored — only real, complete values reach the database.
- `config/settings.yaml`/`loader.py` gained a full `indicators:` section
  with every period from Section 18's own example (`sma_periods: [5, 10,
  20, 50, 100, 200]`, etc.) plus a validation rule that MACD's fast period
  must be less than its slow period — the one relationship between two
  settings where an individually-valid pair can still be nonsensical
  together.
- `src/database/repository.py` gained `get_candles`/`insert_feature`
  (the latter using the same duplicate-safe `ON CONFLICT DO NOTHING`
  pattern as every other insert function in this project). A
  `feature_version` constant (`indicators_v1`) is stored with every row,
  so a future change to a formula or default never silently conflates
  incomparable values under the same feature name.
- `src/main.py` gained `build-indicators --symbol S --timeframe T
  --start-date D --end-date D`.

**Verification — the strongest of any phase so far, because the tooling
allowed it.** Unlike SQLAlchemy, `pandas` and `numpy` are both available in
this sandbox, so every pure indicator function was verified for real, not
just reviewed: 15 no-look-ahead tests (one per indicator output) and 15
correctness tests (hand-calculated SMA/ROC/Momentum values, RSI pinned at
exactly 100/0 for monotonic series, Bollinger's middle band proven equal to
SMA, ATR and Stochastic proven bounded, OBV's running total checked bar by
bar) — **30/30 passed**, run directly against real pandas computation with
no shims of any kind needed. The database-touching half —
`compute_and_store_indicators` — was verified with 4 more tests using the
same fake-repository-injected-via-`sys.modules` technique as the Phase
10/9/8 tests, including one that supplies only 3 candles and confirms every
period-based indicator (needing 5+ candles) is correctly absent while
OBV/VWAP (cumulative from bar one, no warm-up by definition) are correctly
present — a real edge case, not a contrived one. I also ran the actual
`build-indicators` CLI command end-to-end with fake DB modules and real
pandas underneath: 27 indicator columns, 1,202 feature rows, from 60
synthetic candles, in one call.

Still no feature engine, no models, and no signals. Combining these
indicators with KSE-100/KMI-30 market-regime context is next.

---

## 5l. Phase 12 — What Was Actually Built

- `src/features/market_regime.py` — per-index features: returns over every
  configured horizon (reusing the ROC indicator — an N-bar return *is*
  N-bar rate of change), realized volatility, a numeric trend score
  (+1 up / 0 sideways / -1 down, from close vs a fast and slow SMA; numeric
  because the `features` table stores floats), and percentage position vs
  each configured moving average.
- `src/features/stock_features.py` — only features the Phase 11 indicators
  do *not* already provide (RSI/MACD/ATR/VWAP-distance are not recomputed):
  log returns, volume change and acceleration, distance from each SMA,
  high/low breakout flags, bar range, gap, relative volume, and
  `relative_performance` (stock return minus index return — the Section 20
  cross-feature).
- `src/features/engine.py` — `compute_and_store_features()` reads the
  stock's and both indices' candles, computes both groups, and stores them
  as versioned `Feature` rows (`features_v1`, independent of the
  indicators' version). Index features are stored under the *index's own*
  symbol; the stock's, including its relative performance vs each index,
  under the stock's. Alignment is exact-timestamp (`reindex`, no
  forward-fill): where an index has no candle for a stock's timestamp, the
  index-derived value is a gap, not a carried-forward guess.
- `config/settings.yaml`/`loader.py` gained a `features:` section
  (return horizons in bars, breakout window, trend fast/slow SMA periods —
  fast must be less than slow — and volatility window).
- `src/main.py` gained `build-features`.

**Two deliberate decisions worth stating plainly.**
1. *Breakout and relative-volume baselines exclude the current bar.* A
   breakout flag comparing a close to a rolling max that includes the
   current bar's own high is nearly meaningless, and a volume spike that is
   part of its own baseline average dilutes exactly the signal being
   measured. `high_low_breakout` was designed this way from the start;
   `relative_volume` was first written with the current bar included, and
   a hand-calculated test (volume `[100,100,100,100,200]`, period 4,
   expected ratio 2.0, got 1.6) caught it. Fixed to use the prior bars
   only, for the same reason, and the no-look-ahead test re-run against
   the corrected version.
2. *Market breadth and sector performance are not implemented.* Section 19
   makes both conditional on valid data. This project records the index
   level and the stocks you configure — not every KSE-100/KMI-30
   constituent — so a "breadth" figure built from a handful of watchlist
   stocks would be a misleading number presented as a market-wide one. It
   is left out, and the module docstring says why.

**Verification.** Everything here is pandas/numpy, which are installed in
this sandbox, so it ran for real. 13 no-look-ahead tests (every feature
function individually, same mutate-the-future method as the indicators) and
12 hand-calculated correctness tests (trend pinned to exactly +1/-1 for
strictly monotonic series, position-vs-SMA and log return checked by
arithmetic, breakout detection in both directions, the
exclude-the-current-bar rule, bar range, gap, relative volume,
relative performance) — all passing. 5 more tests cover the engine via the
same fake-repository technique as the earlier phases, including the case
where index candles are missing entirely: no relative-performance feature
is written at all, but the stock's own features still are. I also ran
`build-features` end to end: index regime features landed under `KSE100`
and `KMI30`, and the stock's under `THCCL`, relative-performance columns
included.

**A note on process.** Partway through this phase the sandbox's compute
container was reset and the working directory was lost, including
uncommitted Phase 12 work. The Phase 11 zip in persistent storage was
intact; the project was restored from it, integrity-checked (all files
compile, 69 files as before), and Phase 12 was rebuilt on top. Nothing from
Phases 1-11 was affected.

Still no labels, no models, and no signals. Prediction targets are next.

---

## 5m. Phase 13 — What Was Actually Built

**Labels are the one place this project deliberately uses future data** —
a label at time T is "what happened after T". Everything in this phase is
arranged so that can never leak into a feature.

- `src/features/labels.py` — the four targets Section 22 asks for, each
  computed for every configured horizon (in bars of the timeframe):
  - **C. future return** — percent change from this close to the close
    `h` bars later.
  - **A. direction** — 1.0 UP / 0.0 DOWN. A return of exactly zero has no
    direction, so it is left unlabeled rather than forced into a class.
  - **B. three-class** — +1 UP / 0 NEUTRAL / -1 DOWN, NEUTRAL being a
    configurable band around zero.
  - **D. take-profit before stop-loss** — price touching entry × (1 + TP%)
    before entry × (1 − SL%) within the next `h` bars. If a single bar
    touches *both* barriers, bar data cannot say which came first, so it
    takes the pessimistic reading (stop-loss first). If neither is touched
    within the horizon the label is 0 (the event "TP before SL" did not
    happen).
- **The last `h` bars of any series have no label.** Their future is not in
  the data, so they are NaN and never filled or stored. To label the final
  bars of a range, read candles past them — the CLI prints a note about this.
- **A new `labels` table** (`Label` model, `insert_label`), separate from
  `features` on purpose. This changes the schema from 14 to 15 tables;
  the Phase 3 write-up above describes the schema as it was at that point.
  Because labels live elsewhere, a query for model inputs cannot return one
  by accident.
- `src/features/label_engine.py` — `compute_labels()` (pure) and
  `compute_and_store_labels()`, with its own `labels_v1` version.
- `config/settings.yaml`/`loader.py` gained a `labels:` section (horizons,
  neutral band, take-profit %, stop-loss %). The take-profit/stop-loss
  defaults (1.0% / 0.5%) are illustrative, not tuned — nothing here has
  been backtested yet.
- `src/main.py` gained `build-labels`.

**Leakage tests (Sections 21 and 101).** Beyond hand-calculated checks of
each target, three tests pin down the *direction* of dependence:
mutating only bars *before* T leaves every label from T onward unchanged;
mutating only bars *after* T changes the label at T but leaves every
feature at or before T untouched (checked on three feature families); and
the same before/after check for the take-profit/stop-loss label. A fifth
check confirms the last `h` rows are NaN for every horizon.

**Verification.** All of this is pandas/numpy, so it ran for real: 13
label tests (hand-calculated returns, flat-return handling, band edges,
take-profit-first, stop-loss-first, the same-bar pessimistic case, timeout,
trailing NaNs, and the three leakage tests) and 5 engine tests through the
same fake-repository technique as earlier phases. The fake repository
exposes **no `insert_feature`**, so if the label engine ever wrote to the
features table the tests would fail outright. I also ran `build-labels` end
to end: from 60 candles it wrote exactly 55, 45 and 30 labels per target
for the 5-, 15- and 30-bar horizons (that is `n − h`), with none on the
final bars. The new `labels` table itself, like the rest of the SQLAlchemy
layer, could not be exercised here (no SQLAlchemy offline), so
`tests/test_database.py` — which checks that every model's table gets
created — still needs a local `pytest` run.

Still no models, no baselines, and no signals. Baseline strategies are
next, and they are the yardstick any ML model must later beat.

---

## 5n. Phase 14 — What Was Actually Built

**Scope, stated first:** baselines here are *prediction-quality benchmarks*
scored against the Phase 13 direction labels. They are not trading
strategies, and no fees, tax, slippage or spread is applied anywhere
(that is the Phase 16 backtester). Section 81's point — a 60%-accurate model
can still lose money after costs — is exactly why this phase cannot, and does
not, claim any baseline is profitable.

- `src/models/baseline.py` — the six predictors from Section 23, each
  returning 1.0 (UP), 0.0 (DOWN) or NaN (no opinion): **buy-and-hold**
  (always UP), **previous direction** (the last move continues),
  **moving average** (fast SMA vs slow SMA), **RSI** (oversold -> UP,
  overbought -> DOWN, otherwise abstain), **random** (seeded coin flip), and
  **majority class**. Abstaining is deliberate — forcing a guess where a rule
  has no signal would distort its apparent skill — so every result carries
  its *coverage*.
- The majority-class predictor takes the **training labels as an explicit
  argument**, so it is structurally unable to look at the test period. The
  rest are causal, verified by the same mutate-the-future test as the
  indicators.
- `src/models/evaluation.py` — confusion-matrix metrics (accuracy, balanced
  accuracy, precision/recall/F1 for UP; undefined ratios are NaN, never
  silently 0), a chronological split that **purges the last `horizon` train
  bars** (their labels reach into the test period), and the *null accuracy*:
  what a constant guess would have scored on the same bars. A baseline only
  shows skill if it clearly beats that number, not merely 50%.
- `src/models/baseline_runner.py` — scores every baseline on one shared
  chronological test segment and records one `model_runs` row each (Section
  100), with seed, Python version and package versions (Section 59), empty
  feature list (baselines read raw price only), and metrics as valid JSON
  (NaN stored as null).
- `config/settings.yaml`/`loader.py` gained a `baselines:` section
  (moving-average periods with fast < slow, RSI thresholds, train fraction,
  random seed). `src/main.py` gained `evaluate-baselines`.

**Two flaws found by reading the output, not just the exit code.** The first
end-to-end run, on a synthetic random walk (no structure exists to find),
showed:
1. **A wrong verdict.** A moving-average rule scored 39% accuracy, and its
   verdict read "ABOVE 50% BUT NO BETTER THAN ALWAYS GUESSING". The verdict
   logic had no branch for "significantly *worse* than chance". Fixed, with a
   regression test.
2. **A significance test that cried wolf.** The same row had p = 0.0000 on
   pure noise. I had documented that overlapping labels inflate significance,
   but I had not measured by how much. Measured over 300 random walks
   (nominal false-positive rate 5%): buy-and-hold 35%, moving-average 28%,
   RSI 13%. The cause: a horizon-5 label shares 4 of its 5 bars with its
   neighbour, so roughly `n / horizon` observations are independent, not `n`.
   p-values now use that *effective sample size*, which brought the measured
   rate to 3% or below — slightly conservative, the right direction for a
   yardstick. A guard test re-runs 80 noise series and fails if any baseline
   is falsely flagged more than 10% of the time, and a verdict now also
   needs at least 30 independent outcomes. This is still an approximation (a
   persistent signal such as a moving-average state stays correlated across
   bars); a small p-value means "look closer", not "proven".

**What the numbers on synthetic data do and do not mean.** Every number
produced during development came from random-walk candles I generated. On
those, every baseline correctly reads "not distinguishable from chance"
(with RSI abstaining on ~90% of bars and reporting insufficient data), and
buy-and-hold and majority-class coincide whenever the training majority is UP,
since both then always guess UP. That is the *correct* result for noise; it
says nothing about how these rules behave on real PSX data, which this
project does not have yet. Do not generalize from one symbol over one period.

**Verification.** All pure logic ran for real (pandas/numpy are installed
here): 10 predictor tests (hand-checked outputs, abstention, seed
reproducibility, tie handling, causality), 12 evaluation tests (confusion
counts and F1 checked by hand, abstention vs coverage, an always-UP
predictor on imbalanced labels shown to score 0.8 accuracy but 0.5 balanced
accuracy, p-values, every verdict branch, split and purge), and 14 runner
tests, including a majority-class predictor that must call DOWN after a
falling train period and be confidently wrong on a reversed test period,
mutating only test-period prices leaving the learned majority class
unchanged, and the `model_runs` recording through a fake repository. The
`model_runs` write path against a real database, like the rest of the
SQLAlchemy layer, could not be exercised offline.

Still no ML models, no backtester, and no signals. The first real model is next
— and now there is an honest bar for it to clear.

---

## 5o. Phase 15 — What Was Actually Built

**Scope first.** These models predict 5/15/30-bar *direction* and are judged on
prediction quality only. No fees, tax, slippage or spread is applied (Phase 16),
and nothing consumes the predictions yet. Nothing here says any model is
profitable.

- `src/models/dataset.py` — builds the feature matrix from candles using the
  same pure functions as the stored indicators/features, **scale-free only**:
  raw price levels (SMA/EMA values, Bollinger bands, VWAP, OBV, raw ATR/MACD)
  are dropped or re-expressed as a percentage of price, so a model cannot learn
  "where price is" instead of "what price is doing". A test multiplies every
  price by 10 and requires the feature matrix to come out identical.
- **Purged chronological split** (`train 60% / validation 20% / test 20%`,
  never shuffled). Rows whose label reaches `horizon` bars ahead into the next
  segment are dropped, using original bar positions rather than row counts
  (rows removed for NaN would otherwise make a count-based purge too small).
  It raises, rather than returning an unusable split, if a segment ends up empty.
- `src/models/ml_models.py` — Logistic Regression, Random Forest, Gradient
  Boosting, HistGradientBoosting: small fixed grids (4 candidates each), shallow
  trees, one thread, no deep learning. Hyperparameters are chosen by **validation
  log loss**; the winner is fit on train only. A Platt calibrator is then fit on
  validation. So validation is used twice (to choose, then to calibrate) — said
  so in the code and the output — and the test segment is touched by neither.
- `src/models/ml_metrics.py` — accuracy, balanced accuracy, precision/recall/F1
  for UP, ROC-AUC, PR-AUC, Brier, log loss, a calibration table and expected
  calibration error. Undefined metrics are NaN, never 0.
- `src/models/trainer.py` — runs everything and scores each model on the **same
  test rows as the Phase 14 baselines**. Because four models are tried on the
  same data, the significance bar is tightened to `0.05 / models compared`.
  A model is badged "clears baselines" only if its verdict is a genuine beat of
  a constant guess **and** its accuracy exceeds the best baseline's. Which model
  is "selected" is decided by validation log loss, not by test results; test
  results are shown for all of them. Permutation importance is computed on
  validation.
- `src/models/model_manager.py` — IDs like `THCCL_1M_5B_RF_20261023_001`, models
  saved as `.joblib` plus a metadata JSON (hyperparameters, feature list,
  periods, seed, metrics, and Python/scikit-learn/pandas/numpy versions), a
  version-mismatch warning on load. **`.joblib` files are pickles: only load
  files this project wrote.**
- `config/settings.yaml`/`loader.py` gained a `models:` section; `src/main.py`
  gained `train`; `requirements.txt` gained `scikit-learn`, `joblib`,
  `threadpoolctl` (SciPy comes with scikit-learn and is not imported directly).
- Each model is recorded in `model_runs` (status TRAINED) and its **out-of-sample
  test predictions** are stored in `predictions` — `probability_neutral` and
  `expected_return` are stored as NULL because this phase models neither.

**Deliberately not built.** (1) `HistGradientBoostingRegressor` / an
`expected_return` — predicting magnitude is not justified before direction has
been shown to carry signal. (2) The `predict` command — it belongs with the
signal pipeline (Phase 18); `load_model` already exists and is tested. (3) A
three-class (UP/NEUTRAL/DOWN) model. (4) Costs, backtest, walk-forward — Phases
16/17.

**What testing on controlled series showed** (all synthetic — this project has
no real PSX data, so none of this says how the models behave on the real market):
- *Pure random-walk noise:* the four models were never badged as clearing the
  baselines. Over 40 noise series (logistic and HistGradientBoosting), **0 of 40**
  were flagged — at a bar of 0.025 each, deliberately conservative.
- *But* the test **AUC** on that same noise ranged from **0.28 to 0.77**. AUC has
  no significance test of its own and a single test segment of overlapping,
  autocorrelated bars is very noisy, so **an AUC of 0.7 from one segment is not
  evidence of skill**. The `train` output now says so.
- *A planted momentum signal* (AR(1) returns): the models find it (AUC about
  0.74-0.78) and lean on the one-bar-return features — but the best baseline,
  `previous_direction` (about 0.74-0.76), is exactly the rule that encodes the
  planted mechanism, and **no model out-scores it**. The framework reports that
  rather than crowning the model.
- *A regime-dependent structure* (momentum in high-volatility blocks,
  mean-reversion in low-volatility ones) that no simple baseline can see: tree
  models do better than chance on average over four pre-chosen seeds, but
  single-seed results are noisy (about 100 independent test outcomes), so the test
  asserts the average, not a lucky seed.

**Four things I got wrong or nearly got wrong, found by looking:**
1. **A leakage control that was itself flawed.** A random forest trained on
   *shuffled* labels scored test AUC 0.69. Before touching the threshold I
   measured 30 shuffles: the mean was 0.49-0.50 with a standard deviation near
   0.09, and tails on *both* sides. A model fit to shuffled labels is a random
   function of the features, so when the features genuinely relate to the test
   labels its AUC lands far from 0.5 in either direction by random sign. No
   leakage — my single-draw assertion was wrong. The control now averages over
   many shuffles and requires the real model to sit well outside that
   distribution.
2. **Negative prices silently inverted the labels.** On the planted-signal
   series, the `previous_direction` baseline scored 0.25 where it should have
   scored 0.75. Raw arithmetic with no project code showed the relationship was
   a stable +0.77 agreement; the baseline scored exactly 1 − 0.77 wherever my
   synthetic price went **negative** (range -88 to 100), because a percentage
   return divides by the starting price and a negative start flips the sign.
   Real prices cannot be negative (the quality engine excludes such ticks), but
   a silent sign flip on impossible input is worth making loud: `future_return`
   now returns NaN for a non-positive start price, `build_dataset` raises on any
   non-positive close, and the generator now builds log-price. Every number
   above was re-measured on the corrected series.
3. **A stale verdict.** It still said "overlapping labels inflate significance"
   after Phase 14 had corrected for exactly that. Reworded.
4. **A split that could return an empty training set** after the purge. It now
   raises.

**Verification.** scikit-learn 1.8.0, scipy and joblib are installed in this
sandbox (I had assumed they would not be, and checked first), so every model
was trained for real and **307 tests ran and passed**. 16 more (`test_database.py`,
`test_csv_importer.py`) need SQLAlchemy and could not run; that includes the
`labels` table creation from Phase 13. The new `insert_prediction` and the
`model_runs` write path, like the rest of the SQLAlchemy layer, are untested
against a real database. Key tests: scale-invariance of the features; the
mutate-the-future check on the assembled matrix; **scrambling only the test
period leaves every hyperparameter, validation metric and importance
bit-for-bit unchanged** (Section 32); the corrected label-shuffle control;
reproducibility from a seed; and a saved model reproducing its own stored test
predictions after being loaded.

**A note on process.** The sandbox's compute container reset again during this
phase and took uncommitted work with it; I restored from the Phase 14 zip and
rebuilt, saving checkpoints to persistent storage as I went. Later, when some
earlier edits turned out to have already been applied, I stopped trusting my
own notes and re-verified the whole project from scratch (compile, duplicate
scans, entire suite) before continuing.

Still no backtester, no signals, and no evidence of anything profitable. The
event-driven backtester — where costs, slippage and the PSX fee schedule finally
enter — is next.

---

## 6. Installation (Windows)

```bat
cd psx_ai_trader
install.bat
```

`install.bat` creates a virtual environment, installs `requirements.txt`, and
runs `python -m src.main init`. See Section 7 below for the manual/cross-platform
equivalent.

## 7. Manual Installation (copy/paste ready)

```bash
cd psx_ai_trader
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux (for development off-Windows)
source .venv/bin/activate

pip install -r requirements.txt
python -m src.main init
python -m src.main validate-config
python -m src.main init-db
python -m src.main provider-demo
python -m src.main health
pytest -q
```

To import a real historical CSV file once you have one:

```bash
python -m src.main import-csv --file path\to\THCCL.csv --symbol THCCL --symbol-type stock
```

To check the current market session or record one KSE-100/KMI-30 observation:

```bash
python -m src.main market-status
python -m src.main record-indices          # only records when the market is open
python -m src.main record-indices --force  # override, for testing with the mock provider
python -m src.main record-stocks
python -m src.main run-collector --max-cycles 3   # try a few cycles before running forever
```

To run the collector continuously (Ctrl+C to stop):

```bash
python -m src.main run-collector
# or on Windows:
run_collector.bat
```

To build candles from recorded ticks once you have some:

```bash
python -m src.main build-candles --symbol THCCL --date 2026-09-21
python -m src.main build-candles --symbol KSE100 --date 2026-09-21 --timeframes 5m,daily
```

To compute technical indicators from candles you've already built:

```bash
python -m src.main build-indicators --symbol THCCL --timeframe 1m --start-date 2026-09-21 --end-date 2026-09-21
```

To compute stock + KSE-100/KMI-30 market-regime features (build candles for
the stock AND both indices at the same timeframe first):

```bash
python -m src.main build-features --symbol THCCL --timeframe 1m --start-date 2026-09-21 --end-date 2026-09-21
```

To build the prediction targets (labels look *forward*, so the final bars of
the range stay unlabeled — extend `--end-date` past the bars you care about):

```bash
python -m src.main build-labels --symbol THCCL --timeframe 1m --start-date 2026-09-21 --end-date 2026-09-21
```

To score the baseline predictors (candles must exist for the symbol; `--horizon`
must be one of `labels.horizons`). This reports prediction quality only:

```bash
python -m src.main evaluate-baselines --symbol THCCL --timeframe 1m --start-date 2026-09-01 --end-date 2026-09-21 --horizon 5
```

To train and compare the ML models (candles must exist for the symbol AND for
each index at the same timeframe; needs `pip install -r requirements.txt` for
scikit-learn). Prediction quality only — no costs:

```bash
python -m src.main train --symbol THCCL --timeframe 1m --start-date 2026-09-01 --end-date 2026-09-21 --horizon 5
python -m src.main train --symbol THCCL --timeframe 1m --start-date 2026-09-01 --end-date 2026-09-21 --horizon 5 --models logistic,random_forest
```

## 8. Safety Notice (applies to the whole project, not just Phase 1)

- Default mode is **RESEARCH**. Nothing in this codebase places an order.
- `LIVE_TRADING_ENABLED=false` in `.env.example` is the master switch. It will
  gate all live order placement starting Phase 26, and the code will refuse to
  send a live order if it is not explicitly `true` — this is never optional,
  never silently bypassed, and never removed.
- No claim of guaranteed or reliable prediction will ever appear in this
  project's output. If a future phase's backtest shows a strategy losing money,
  or a model failing to beat its baseline, that will be reported as-is.
