#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_wlc(data):
    books = data.get("books", {})
    chapter_count = 0
    verse_count = 0
    word_count = 0
    morph_tags = set()
    lemmas = set()
    bdb_ids = set()

    for book_data in books.values():
        chapters = book_data.get("chapters", {})
        chapter_count += len(chapters)
        for chapter_data in chapters.values():
            verses = chapter_data.get("verses", {})
            verse_count += len(verses)
            for verse_data in verses.values():
                words = verse_data.get("words", [])
                word_count += len(words)
                for word in words:
                    morph = word.get("morph")
                    lemma = word.get("lemma")
                    bdb = word.get("bdb")
                    if morph:
                        morph_tags.add(morph)
                    if lemma:
                        lemmas.add(lemma)
                    if bdb:
                        bdb_ids.add(bdb)

    print("WLC Summary")
    print(f"- books: {len(books)}")
    print(f"- chapters: {chapter_count}")
    print(f"- verses: {verse_count}")
    print(f"- words: {word_count}")
    print(f"- unique lemmas: {len(lemmas)}")
    print(f"- unique morph tags: {len(morph_tags)}")
    print(f"- unique bdb refs in words: {len(bdb_ids)}")


def summarize_bdb(data):
    entries = data.get("entries", {})
    gloss_total = 0
    with_headword = 0
    with_definition = 0

    for entry in entries.values():
        gloss_total += len(entry.get("glosses", []))
        if entry.get("headword"):
            with_headword += 1
        if entry.get("definition"):
            with_definition += 1

    print("BDB Summary")
    print(f"- entries: {len(entries)}")
    print(f"- entries with headword: {with_headword}")
    print(f"- entries with definition: {with_definition}")
    print(f"- total glosses: {gloss_total}")


def main():
    parser = argparse.ArgumentParser(description="Summarize WLC/BDB JSON files.")
    parser.add_argument("--wlc", default="data/wlc_full.json", help="Path to WLC JSON")
    parser.add_argument("--bdb", default="data/bdb_full.json", help="Path to BDB JSON")
    args = parser.parse_args()

    wlc_path = Path(args.wlc)
    bdb_path = Path(args.bdb)

    if wlc_path.exists():
        summarize_wlc(load_json(wlc_path))
        print()
    else:
        print(f"Skipping WLC (missing): {wlc_path}")

    if bdb_path.exists():
        summarize_bdb(load_json(bdb_path))
    else:
        print(f"Skipping BDB (missing): {bdb_path}")


if __name__ == "__main__":
    main()
