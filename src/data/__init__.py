"""Data provider abstraction layer (Phase 4+).

Per the project's data-source rules: the rest of the application never talks
to a concrete provider directly. It depends only on `DataProvider` (defined
in `base_provider.py`). Concrete implementations are added one at a time:

    Phase 4  — MockProvider    (synthetic data, no external dependency)
    Phase 5  — CSVProvider      (historical CSV import)
    Phase 7  — PSXProvider       (real PSX data, pending provider research)
    Phase 26 — BrokerProvider     (only via an officially documented API)

Nothing in this package is allowed to scrape a website, bypass
authentication/rate limits, or assume an undocumented endpoint exists.
"""
