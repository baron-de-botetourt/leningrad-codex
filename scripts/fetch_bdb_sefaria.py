#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from bdb_sefaria_source import build_bdb_json_from_raw, crawl_bdb_sefaria_raw, save_json


def main():
    parser = argparse.ArgumentParser(
        description="Fetch complete BDB from Sefaria via one-pass next-chain crawl and cache raw output."
    )
    parser.add_argument("--raw-out", default="data/bdb_sefaria_raw.json", help="Path to write raw crawl JSON")
    parser.add_argument("--bdb-out", default="data/bdb_full.json", help="Path to write parsed BDB JSON")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume/checkpoint file path (defaults to --raw-out)",
    )
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between API calls in milliseconds")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save resume checkpoint after this many entries",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print crawl progress every N entries",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional cap for debugging; omit for full crawl",
    )
    args = parser.parse_args()

    resume_path = Path(args.resume_from) if args.resume_from else Path(args.raw_out)
    raw = crawl_bdb_sefaria_raw(
        delay_ms=args.delay_ms,
        max_entries=args.max_entries,
        resume_path=resume_path,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
    )
    bdb = build_bdb_json_from_raw(raw)

    raw_path = Path(args.raw_out)
    bdb_path = Path(args.bdb_out)
    save_json(raw_path, raw)
    save_json(bdb_path, bdb)

    print(f"Wrote {raw_path}")
    print(f"Wrote {bdb_path}")
    print(f"Entries: {len(bdb.get('entries', {}))}")
    print(f"First ref: {raw.get('index', {}).get('start_ref')}")
    print(f"Last ref: {raw.get('entries', [])[-1].get('ref') if raw.get('entries') else ''}")


if __name__ == "__main__":
    main()
