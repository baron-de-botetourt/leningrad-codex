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


def clean_strongs_key(raw_id: str) -> str:
    value = (raw_id or "").strip()
    if not value:
        return ""
    value = value.replace(" ", "")
    m = re.match(r"^[Hh]?([0-9]+)([A-Za-z]?)$", value)
    if not m:
        return value
    number = str(int(m.group(1)))
    suffix = m.group(2).lower()
    return f"{number}{suffix}"


def split_strongs_key(key: str):
    m = re.match(r"^([0-9]+)([a-z]?)$", key)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def strongs_key_variants(key: str):
    parsed = split_strongs_key(key)
    if parsed is None:
        return [key]
    number, suffix = parsed
    suffix_upper = suffix.upper()

    variants = [
        f"{number}{suffix}",
        f"{number}{suffix_upper}" if suffix else f"{number}",
        f"{number:04d}{suffix}",
        f"{number:04d}{suffix_upper}" if suffix else f"{number:04d}",
        f"H{number}{suffix}",
        f"H{number}{suffix_upper}" if suffix else f"H{number}",
        f"h{number}{suffix}",
        f"h{number}{suffix_upper}" if suffix else f"h{number}",
    ]
    if suffix:
        variants.extend(
            [
                f"{number} {suffix}",
                f"{number} {suffix_upper}",
                f"H{number} {suffix}",
                f"H{number} {suffix_upper}",
            ]
        )

    seen = set()
    ordered = []
    for variant in variants:
        if variant not in seen:
            ordered.append(variant)
            seen.add(variant)
    return ordered


def append_unique(values, candidate):
    val = (candidate or "").strip()
    if val and val not in values:
        values.append(val)


def parse_strongs_hebrew(strongs_path: Path):
    tree = ET.parse(strongs_path)
    root = tree.getroot()
    entries = {}

    for el in root.iter():
        tag = strip_ns(el.tag).lower()
        attrs = {k.lower(): v for k, v in el.attrib.items()}
        entry_id = attrs.get("id") or attrs.get("{http://www.w3.org/XML/1998/namespace}id")
        if tag not in {"entry", "item"} or not entry_id:
            continue

        normalized_key = clean_strongs_key(str(entry_id))
        if not normalized_key:
            continue

        headword = ""
        transliteration = ""
        pronunciation = ""
        source = ""
        meaning = ""
        usage = ""
        glosses = []

        for child in list(el):
            child_tag = strip_ns(child.tag).lower()
            child_text = re.sub(r"\s+", " ", text_of(child)).strip()
            if child_tag in {"w", "word", "headword"}:
                if child_text:
                    headword = child_text
                child_attrs = {k.lower(): v for k, v in child.attrib.items()}
                if not transliteration:
                    transliteration = child_attrs.get("xlit", "").strip()
                if not pronunciation:
                    pronunciation = child_attrs.get("pron", "").strip()
            elif child_tag in {"source", "etym", "etymology"} and child_text:
                source = child_text
            elif child_tag in {"meaning", "gloss", "definition", "def"} and child_text:
                meaning = child_text
                append_unique(glosses, re.split(r"[;,]", child_text)[0])
            elif child_tag in {"usage", "kjv", "kjv_def"} and child_text:
                usage = child_text

        if not glosses:
            append_unique(glosses, meaning)
        if not glosses:
            append_unique(glosses, headword)

        definition_parts = []
        if source:
            definition_parts.append(f"Source: {source}")
        if meaning:
            definition_parts.append(f"Meaning: {meaning}")
        if usage:
            definition_parts.append(f"Usage: {usage}")
        definition = " ".join(definition_parts).strip()

        payload = {
            "headword": headword,
            "transliteration": transliteration,
            "pronunciation": pronunciation,
            "glosses": glosses,
            "definition": definition,
            "source": source,
            "meaning": meaning,
            "usage": usage,
        }

        for key_variant in strongs_key_variants(normalized_key):
            entries[key_variant] = payload

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


def resolve_lexicon_file(lex_root: Path, candidates: list[str], label: str) -> Path:
    for rel in candidates:
        path = lex_root / rel
        if path.exists():
            return path
    attempted = ", ".join(str(lex_root / rel) for rel in candidates)
    raise RuntimeError(f"Could not find {label}. Tried: {attempted}")


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

        lexical_index_path = resolve_lexicon_file(
            lex_root,
            ["LexicalIndex.xml", "lexicalindex.xml"],
            "LexicalIndex XML",
        )
        bdb_path = resolve_lexicon_file(
            lex_root,
            ["BrownDriverBriggs.xml", "browndriverbriggs.xml"],
            "BrownDriverBriggs XML",
        )
        strongs_path = resolve_lexicon_file(
            lex_root,
            ["HebrewStrong.xml", "StrongHebrew.xml", "hebrewstrong.xml", "stronghebrew.xml"],
            "HebrewStrong XML",
        )

        strong_to_bdb = parse_lexical_index(lexical_index_path)
        relink_bdb_codes(wlc_data, strong_to_bdb)

        bdb_data = parse_bdb(bdb_path)
        strongs_data = parse_strongs_hebrew(strongs_path)

    (out_dir / "wlc_full.json").write_text(json.dumps(wlc_data, ensure_ascii=False), encoding="utf-8")
    (out_dir / "bdb_full.json").write_text(json.dumps(bdb_data, ensure_ascii=False), encoding="utf-8")
    (out_dir / "strongs_full.json").write_text(json.dumps(strongs_data, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_dir / 'wlc_full.json'}")
    print(f"Wrote {out_dir / 'bdb_full.json'}")
    print(f"Wrote {out_dir / 'strongs_full.json'}")


if __name__ == "__main__":
    main()
