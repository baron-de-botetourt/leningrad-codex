# BDB Source Strategy

## Selected source

- Primary BDB corpus: Sefaria BDB dictionary API (`https://www.sefaria.org/api/index/BDB`).
- Why selected: provides a chain-linked dictionary corpus with `prev/next` navigation from first to last headword, suitable for one-pass complete crawl and reproducible caching.
- Open Scriptures BDB XML is no longer the primary BDB text source.

## Completeness model

Completeness is verified from the cached raw crawl (`data/bdb_sefaria_raw.json`) by proving:

1. Crawl starts at index `start_ref` and walks `next` until terminal `next = null`.
2. No duplicate refs (no cycle).
3. Each entry's `prev`/`next` exactly matches neighbors in saved order.
4. All index letter anchor refs exist in the crawled set.
5. Last crawled headword matches index `last_word`.

If any condition fails, the import is rejected.

## Validation model

1. Source-chain completeness checks (from raw crawl).
2. Scan-backed phrase checks against Wikisource BDB pages.
3. Encoding checks:
   - headword coverage threshold,
   - no trailing editorial marker artifacts,
   - no Unicode control/replacement characters,
   - NFC normalization.

Command:

```bash
./scripts/verify_bdb_authoritative.py --bdb data/bdb_full.json --raw data/bdb_sefaria_raw.json
```

## One-pass crawl rule

- Crawl full API chain once and save to `data/bdb_sefaria_raw.json`.
- Subsequent imports must re-use cache unless explicitly refreshed.
