#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

MORPHHB_ZIP = "https://codeload.github.com/openscriptures/morphhb/zip/refs/heads/master"
LEXICON_ZIP = "https://codeload.github.com/openscriptures/HebrewLexicon/zip/refs/heads/master"

BOOK_NAMES = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "Isaiah", "Jeremiah", "Ezekiel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah",
    "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Psalms", "Job", "Proverbs",
    "Ruth", "Song of Songs", "Ecclesiastes", "Lamentations", "Esther", "Daniel", "Ezra", "Nehemiah", "1 Chronicles",
    "2 Chronicles",
]


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def text_of(el: ET.Element) -> str:
    return "".join(el.itertext()).strip()


def normalize_lemma(lemma: str) -> str:
    return (lemma or "").strip()


def lemma_lookup_candidates(lemma: str):
    raw = (lemma or "").strip()
    if not raw:
        return []

    candidates = [raw]

    # Remove OSHB prefix markers like c/, b/, l/, m/, k/ chains.
    no_prefix = re.sub(r"^(?:[a-z]/)+", "", raw)
    candidates.append(no_prefix)

    # Remove homograph markers written as trailing space + letter, e.g. "1121 a".
    no_space_type = re.sub(r"\s+[a-zA-Z]$", "", no_prefix)
    candidates.append(no_space_type)

    # Also try compact form, e.g. "1121a".
    compact = re.sub(r"\s+", "", no_prefix)
    candidates.append(compact)

    # Remove trailing plus markers used in some token IDs.
    no_plus = re.sub(r"\+$", "", no_space_type)
    candidates.append(no_plus)
    candidates.append(re.sub(r"^(?:[a-z]/)+", "", no_plus))

    # De-dupe while preserving order.
    seen = set()
    ordered = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def parse_ref(verse_el: ET.Element):
    osis = verse_el.attrib.get("osisID", "")
    m = re.search(r"([A-Za-z0-9]+)\.(\d+)\.(\d+)$", osis)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def resolve_book(osis_book: str) -> str:
    return osis_book


def parse_morphhb(morphhb_root: Path):
    wlc_dir = morphhb_root / "wlc"
    book_files = sorted(wlc_dir.glob("*.xml"))
    if not book_files:
        raise RuntimeError("No OSIS XML files found under morphhb/wlc")

    books = {}
    for file in book_files:
        tree = ET.parse(file)
        root = tree.getroot()

        for verse in root.iter():
            if strip_ns(verse.tag) != "verse":
                continue
            ref = parse_ref(verse)
            if not ref:
                continue
            osis_book, chapter, verse_num = ref
            book = resolve_book(osis_book)

            words = []
            for w in verse.iter():
                if strip_ns(w.tag) != "w":
                    continue
                txt = text_of(w)
                lemma = normalize_lemma(w.attrib.get("lemma", ""))
                morph = w.attrib.get("morph", "")
                words.append({
                    "text": txt,
                    "lemma": lemma,
                    "morph": morph,
                    "bdb": lemma,
                })

            if not words:
                continue

            books.setdefault(book, {"chapters": {}})
            chapters = books[book]["chapters"]
            chapters.setdefault(chapter, {"verses": {}})
            chapters[chapter]["verses"][verse_num] = {"words": words}

    return {"books": books}


def parse_lexical_index(lex_path: Path):
    # Build map of Strong ID -> BDB code when present.
    tree = ET.parse(lex_path)
    root = tree.getroot()
    mapping = {}

    for el in root.iter():
        attrs = {k.lower(): v for k, v in el.attrib.items()}
        strong = attrs.get("strong") or attrs.get("strongs") or attrs.get("id")
        bdb = attrs.get("bdb") or attrs.get("bdbid") or attrs.get("brown")
        if strong and bdb:
            mapping[str(strong).strip()] = str(bdb).strip()

    return mapping


def parse_bdb(bdb_path: Path):
    tree = ET.parse(bdb_path)
    root = tree.getroot()

    entries = {}
    for el in root.iter():
        tag = strip_ns(el.tag).lower()
        if tag not in {"entry", "item"}:
            continue

        attrs = {k.lower(): v for k, v in el.attrib.items()}
        entry_id = attrs.get("id") or attrs.get("xml:id") or attrs.get("code")
        if not entry_id:
            continue

        heb = attrs.get("word") or attrs.get("head") or ""
        definition = text_of(el)
        definition = re.sub(r"\s+", " ", definition).strip()
        if not definition:
            continue

        entries[str(entry_id)] = {
            "headword": heb,
            "glosses": [],
            "definition": definition,
        }

    return {"entries": entries}


def relink_bdb_codes(wlc_data, strong_to_bdb):
    for book in wlc_data["books"].values():
        for chapter in book["chapters"].values():
            for verse in chapter["verses"].values():
                for word in verse["words"]:
                    lemma = word.get("lemma", "")
                    bdb = None
                    for candidate in lemma_lookup_candidates(lemma):
                        bdb = strong_to_bdb.get(candidate)
                        if bdb:
                            break
                    word["bdb"] = bdb or lemma


def download_and_unpack(url: str, dest: Path):
    zip_path = dest / "archive.zip"
    urllib.request.urlretrieve(url, zip_path)
    shutil.unpack_archive(str(zip_path), str(dest))


def find_repo_root(temp_root: Path, prefix: str):
    matches = [p for p in temp_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not matches:
        raise RuntimeError(f"Could not find unpacked folder for {prefix}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Import OSHB/WLC + BDB into app JSON format.")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        morph_tmp = tmp / "morphhb"
        morph_tmp.mkdir()
        download_and_unpack(MORPHHB_ZIP, morph_tmp)
        morph_root = find_repo_root(morph_tmp, "morphhb-")

        lex_tmp = tmp / "hebrewlex"
        lex_tmp.mkdir()
        download_and_unpack(LEXICON_ZIP, lex_tmp)
        lex_root = find_repo_root(lex_tmp, "HebrewLexicon-")

        wlc_data = parse_morphhb(morph_root)

        strong_to_bdb = parse_lexical_index(lex_root / "LexicalIndex.xml")
        relink_bdb_codes(wlc_data, strong_to_bdb)

        bdb_data = parse_bdb(lex_root / "BrownDriverBriggs.xml")

    (out_dir / "wlc_full.json").write_text(json.dumps(wlc_data, ensure_ascii=False), encoding="utf-8")
    (out_dir / "bdb_full.json").write_text(json.dumps(bdb_data, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_dir / 'wlc_full.json'}")
    print(f"Wrote {out_dir / 'bdb_full.json'}")


if __name__ == "__main__":
    main()
