#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check(condition, errors, message):
    if not condition:
        errors.append(message)


def validate_wlc(wlc):
    errors = []
    books = wlc.get("books")
    check(isinstance(books, dict), errors, "WLC: top-level 'books' must be an object")
    if not isinstance(books, dict):
        return errors, set()

    bdb_refs = set()
    for book_name, book_data in books.items():
        chapters = book_data.get("chapters")
        check(isinstance(chapters, dict), errors, f"WLC: {book_name}.chapters must be an object")
        if not isinstance(chapters, dict):
            continue

        for chapter_num, chapter_data in chapters.items():
            verses = chapter_data.get("verses")
            check(
                isinstance(verses, dict),
                errors,
                f"WLC: {book_name}.{chapter_num}.verses must be an object",
            )
            if not isinstance(verses, dict):
                continue

            for verse_num, verse_data in verses.items():
                words = verse_data.get("words")
                ref = f"{book_name} {chapter_num}:{verse_num}"
                check(isinstance(words, list), errors, f"WLC: {ref}.words must be an array")
                if not isinstance(words, list):
                    continue

                for i, word in enumerate(words):
                    check(isinstance(word, dict), errors, f"WLC: {ref}.words[{i}] must be an object")
                    if not isinstance(word, dict):
                        continue
                    for key in ("text", "lemma", "morph", "bdb"):
                        check(
                            isinstance(word.get(key), str),
                            errors,
                            f"WLC: {ref}.words[{i}].{key} must be a string",
                        )
                    bdb = word.get("bdb")
                    if isinstance(bdb, str) and bdb:
                        bdb_refs.add(bdb)

    return errors, bdb_refs


def validate_bdb(bdb):
    errors = []
    entries = bdb.get("entries")
    check(isinstance(entries, dict), errors, "BDB: top-level 'entries' must be an object")
    if not isinstance(entries, dict):
        return errors, set()

    ids = set(entries.keys())
    for entry_id, entry in entries.items():
        check(isinstance(entry, dict), errors, f"BDB: entries.{entry_id} must be an object")
        if not isinstance(entry, dict):
            continue
        check(isinstance(entry.get("headword"), str), errors, f"BDB: entries.{entry_id}.headword must be a string")
        check(isinstance(entry.get("definition"), str), errors, f"BDB: entries.{entry_id}.definition must be a string")
        glosses = entry.get("glosses")
        check(isinstance(glosses, list), errors, f"BDB: entries.{entry_id}.glosses must be an array")
        if isinstance(glosses, list):
            for i, gloss in enumerate(glosses):
                check(isinstance(gloss, str), errors, f"BDB: entries.{entry_id}.glosses[{i}] must be a string")

    return errors, ids


def main():
    parser = argparse.ArgumentParser(description="Validate WLC/BDB JSON schema and cross references.")
    parser.add_argument("--wlc", default="data/wlc_full.json", help="Path to WLC JSON")
    parser.add_argument("--bdb", default="data/bdb_full.json", help="Path to BDB JSON")
    parser.add_argument("--max-errors", type=int, default=50, help="Maximum errors printed")
    parser.add_argument("--max-missing-refs", type=int, default=20, help="Maximum missing refs printed")
    args = parser.parse_args()

    wlc = load_json(Path(args.wlc))
    bdb = load_json(Path(args.bdb))

    wlc_errors, wlc_bdb_refs = validate_wlc(wlc)
    bdb_errors, bdb_ids = validate_bdb(bdb)

    missing_refs = sorted(ref for ref in wlc_bdb_refs if ref not in bdb_ids)
    all_errors = wlc_errors + bdb_errors

    print("Validation report")
    print(f"- WLC schema errors: {len(wlc_errors)}")
    print(f"- BDB schema errors: {len(bdb_errors)}")
    print(f"- WLC bdb refs: {len(wlc_bdb_refs)}")
    print(f"- BDB entry ids: {len(bdb_ids)}")
    print(f"- Missing WLC->BDB refs: {len(missing_refs)}")

    if all_errors:
        print("\nSchema errors (truncated):")
        for err in all_errors[: args.max_errors]:
            print(f"- {err}")

    if missing_refs:
        print("\nMissing refs (truncated):")
        for ref in missing_refs[: args.max_missing_refs]:
            print(f"- {ref}")

    raise SystemExit(1 if all_errors or missing_refs else 0)


if __name__ == "__main__":
    main()
