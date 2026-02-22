#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from bdb_sefaria_source import (
    build_bdb_json_from_raw,
    crawl_bdb_sefaria_raw,
    load_raw,
    save_json,
)

MORPHHB_ZIP = "https://codeload.github.com/openscriptures/morphhb/zip/refs/heads/master"
LEXICON_ZIP = "https://codeload.github.com/openscriptures/HebrewLexicon/zip/refs/heads/master"

BOOK_NAMES = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "Isaiah", "Jeremiah", "Ezekiel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah",
    "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Psalms", "Job", "Proverbs",
    "Ruth", "Song of Songs", "Ecclesiastes", "Lamentations", "Esther", "Daniel", "Ezra", "Nehemiah", "1 Chronicles",
    "2 Chronicles",
]

BDB_STATUS_MARKERS = {"base", "made", "ref", "done"}
BDB_HEADWORD_TAGS = {"w", "word", "head", "headword", "orth", "lemma", "form", "hebrew", "hw"}
BDB_INVISIBLE_FORMAT_CHARS_RE = re.compile(r"[\u200c\u200d\u200e\u200f\u202a-\u202e]")
BDB_HEBREW_RUN_RE = re.compile(r"[\u0590-\u05ff]+")
BDB_TRAILING_STATUS_RE = re.compile(r"(?:[\s;,\-]+(?:base|made|ref|done))+[\s;,\-]*$", re.IGNORECASE)
BDB_ENGLISH_WORD_RE = re.compile(r"[A-Za-z']+")


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def text_of(el: ET.Element) -> str:
    return "".join(el.itertext()).strip()


def normalize_unicode_text(value: str) -> str:
    cleaned = BDB_INVISIBLE_FORMAT_CHARS_RE.sub("", value or "")
    return unicodedata.normalize("NFC", cleaned)


def normalize_flat_text(value: str) -> str:
    return re.sub(r"\s+", " ", normalize_unicode_text(value)).strip()


def normalize_hebrew_lookup(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value or "")
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    letters_only = re.sub(r"[^\u05d0-\u05ea]", "", without_marks)
    return unicodedata.normalize("NFC", letters_only).strip()


def iter_bdb_entry_text(el: ET.Element):
    if el.text:
        yield el.text

    for child in list(el):
        tag = strip_ns(child.tag).lower()
        child_text = (child.text or "").strip().lower()
        if tag == "status" and child_text in BDB_STATUS_MARKERS:
            # Skip editorial status markers such as done/ref/base/made.
            pass
        else:
            yield from iter_bdb_entry_text(child)

        if child.tail:
            yield child.tail


def clean_bdb_definition(raw_definition: str) -> str:
    definition = normalize_flat_text(raw_definition)
    definition = BDB_TRAILING_STATUS_RE.sub("", definition).strip()
    return definition


def extract_bdb_headword(entry_el: ET.Element, definition: str) -> str:
    attrs = {k.lower(): v for k, v in entry_el.attrib.items()}
    for key in ("word", "head", "headword", "hw", "hebrew"):
        candidate = normalize_flat_text(attrs.get(key, ""))
        if candidate:
            return candidate

    for child in entry_el.iter():
        if child is entry_el:
            continue
        tag = strip_ns(child.tag).lower()
        if tag not in BDB_HEADWORD_TAGS:
            continue
        candidate = normalize_flat_text(text_of(child))
        if BDB_HEBREW_RUN_RE.search(candidate):
            return candidate

    m = BDB_HEBREW_RUN_RE.search(definition)
    if m:
        return m.group(0)

    return ""


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

        raw_definition = "".join(iter_bdb_entry_text(el))
        definition = clean_bdb_definition(raw_definition)
        if not definition:
            continue
        heb = extract_bdb_headword(el, definition)

        entries[str(entry_id)] = {
            "headword": heb,
            "glosses": [],
            "definition": definition,
        }

    return {
        "metadata": {
            "source_provider": "Open Scriptures",
            "source_lexicon_name": "HebrewLexicon BDB",
            "rendering_id": "bdb.openscriptures",
            "rendering_name": "HebrewLexicon BDB (Open Scriptures)",
        },
        "entries": entries,
    }


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


def normalize_legacy_definition(value: str) -> str:
    text = normalize_unicode_text(value or "")
    text = BDB_TRAILING_STATUS_RE.sub("", text).strip()
    return re.sub(r"\s+", " ", text).strip()


def extract_legacy_headword(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    headword = normalize_flat_text(entry.get("headword", ""))
    if headword:
        return headword
    definition = normalize_legacy_definition(entry.get("definition", ""))
    m = BDB_HEBREW_RUN_RE.search(definition)
    if m:
        return m.group(0)
    return ""


def english_keywords(text: str, limit: int = 14):
    text_wo_hebrew = re.sub(r"[\u0590-\u05ff]+", " ", (text or "").lower())
    words = BDB_ENGLISH_WORD_RE.findall(text_wo_hebrew)
    stop = {
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "on",
        "a",
        "an",
        "is",
        "as",
        "by",
        "be",
        "from",
        "that",
        "this",
        "at",
        "it",
        "its",
        "very",
        "often",
        "only",
        "base",
        "etc",
        "late",
        "poet",
    }
    return [w for w in words if w not in stop][:limit]


def overlap_score(keywords: list[str], candidate_text: str) -> int:
    haystack = (candidate_text or "").lower()
    return sum(1 for word in keywords if word in haystack)


def collect_wlc_bdb_refs(wlc_data: dict):
    refs = set()
    for book in wlc_data.get("books", {}).values():
        for chapter in book.get("chapters", {}).values():
            for verse in chapter.get("verses", {}).values():
                for word in verse.get("words", []):
                    ref = (word.get("bdb") or "").strip()
                    if ref:
                        refs.add(ref)
    return refs


def load_legacy_bdb_entries(path_text: str):
    path = Path(path_text)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", {})
        if isinstance(entries, dict):
            return entries

    # Optional fallback for local development if running from git checkout
    try:
        raw = subprocess.check_output(["git", "show", "HEAD:data/bdb_full.json"], text=True)
        payload = json.loads(raw)
        entries = payload.get("entries", {})
        if isinstance(entries, dict):
            return entries
    except Exception:
        pass

    return {}


def build_bdb_legacy_aliases_from_snapshot(bdb_data: dict, legacy_entries: dict):
    bdb_entries = bdb_data.get("entries", {})

    headword_to_refs = {}
    bdb_definitions_lower = {}
    for ref, payload in bdb_entries.items():
        if not isinstance(payload, dict):
            continue
        headword = normalize_hebrew_lookup(payload.get("headword", ""))
        if headword:
            headword_to_refs.setdefault(headword, []).append(ref)
        bdb_definitions_lower[ref] = (payload.get("definition") or "").lower()

    aliases = {}
    stage_exact = 0
    stage_keyword = 0
    unresolved = 0

    for legacy_id, legacy_entry in legacy_entries.items():
        if legacy_id in bdb_entries:
            continue
        if not isinstance(legacy_entry, dict):
            continue

        legacy_def = normalize_legacy_definition(legacy_entry.get("definition", ""))
        legacy_headword = normalize_hebrew_lookup(extract_legacy_headword(legacy_entry))
        candidates = headword_to_refs.get(legacy_headword, []) if legacy_headword else []
        if candidates:
            if len(candidates) == 1:
                aliases[legacy_id] = candidates[0]
                stage_exact += 1
                continue
            keywords = english_keywords(legacy_def)
            best_score = -1
            best_ref = None
            for ref in candidates:
                score = overlap_score(keywords, bdb_definitions_lower.get(ref, ""))
                if score > best_score:
                    best_score = score
                    best_ref = ref
            if best_ref:
                aliases[legacy_id] = best_ref
                stage_exact += 1
                continue

        # Fallback only if lexical headword match failed:
        # choose a clear keyword winner across all BDB definitions.
        keywords = english_keywords(legacy_def)
        if len(keywords) < 2:
            unresolved += 1
            continue
        scored = []
        for ref, definition in bdb_definitions_lower.items():
            score = overlap_score(keywords, definition)
            if score:
                scored.append((score, ref))
        if not scored:
            unresolved += 1
            continue

        scored.sort(reverse=True)
        best_score, best_ref = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1
        # Conservative threshold keeps false mappings low.
        if best_score >= 5 and best_score >= second_score + 1:
            aliases[legacy_id] = best_ref
            stage_keyword += 1
        else:
            unresolved += 1

    stats = {
        "legacy_entries": len(legacy_entries),
        "aliases_built": len(aliases),
        "mapped_by_headword": stage_exact,
        "mapped_by_keyword_fallback": stage_keyword,
        "unresolved": unresolved,
    }
    return aliases, stats


def supplement_missing_legacy_entries(wlc_data: dict, bdb_data: dict, legacy_entries: dict, aliases: dict):
    bdb_entries = bdb_data.get("entries", {})
    wlc_refs = collect_wlc_bdb_refs(wlc_data)
    resolved_refs = set(bdb_entries.keys()) | set(aliases.keys())

    missing_refs = sorted(ref for ref in wlc_refs if ref not in resolved_refs)
    supplemented = []

    for ref in missing_refs:
        legacy_entry = legacy_entries.get(ref)
        if not isinstance(legacy_entry, dict):
            continue
        definition = normalize_legacy_definition(legacy_entry.get("definition", ""))
        if not definition:
            continue
        headword = normalize_unicode_text(extract_legacy_headword(legacy_entry))
        bdb_entries[ref] = {
            "headword": headword,
            "glosses": [],
            "definition": definition,
            "source_ref": ref,
            "source_note": "supplemented_from_legacy_bridge",
        }
        supplemented.append(ref)

    stats = {
        "wlc_refs_total": len(wlc_refs),
        "missing_before_supplement": len(missing_refs),
        "supplemented_entries": len(supplemented),
        "still_missing_after_supplement": len(missing_refs) - len(supplemented),
    }
    return supplemented, stats


def add_strongs_alias_fallbacks(wlc_data: dict, strongs_data: dict, bdb_data: dict, aliases: dict):
    strongs_entries = strongs_data.get("entries", {})
    bdb_entries = bdb_data.get("entries", {})
    wlc_refs = collect_wlc_bdb_refs(wlc_data)

    headword_to_refs = {}
    bdb_definitions = {}
    for ref, payload in bdb_entries.items():
        if not isinstance(payload, dict):
            continue
        normalized = normalize_hebrew_lookup(payload.get("headword", ""))
        if normalized:
            headword_to_refs.setdefault(normalized, []).append(ref)
        bdb_definitions[ref] = (payload.get("definition") or "").lower()

    unresolved = sorted(ref for ref in wlc_refs if ref not in bdb_entries and ref not in aliases)
    added = 0
    unresolved_after = 0

    for ref in unresolved:
        strong_payload = None
        for candidate in lemma_lookup_candidates(ref):
            candidate_payload = strongs_entries.get(candidate)
            if isinstance(candidate_payload, dict):
                strong_payload = candidate_payload
                break
        if not strong_payload:
            unresolved_after += 1
            continue

        normalized_headword = normalize_hebrew_lookup(strong_payload.get("headword", ""))
        if not normalized_headword:
            unresolved_after += 1
            continue

        candidates = headword_to_refs.get(normalized_headword, [])
        if not candidates:
            unresolved_after += 1
            continue

        if len(candidates) == 1:
            aliases[ref] = candidates[0]
            added += 1
            continue

        strongs_text_parts = [strong_payload.get("definition", "")]
        glosses = strong_payload.get("glosses")
        if isinstance(glosses, list):
            strongs_text_parts.extend(g for g in glosses if isinstance(g, str))
        strongs_keywords = english_keywords(" ".join(strongs_text_parts))
        if not strongs_keywords:
            unresolved_after += 1
            continue

        best_score = -1
        best_ref = None
        second_score = -1
        for candidate_ref in candidates:
            score = overlap_score(strongs_keywords, bdb_definitions.get(candidate_ref, ""))
            if score > best_score:
                second_score = best_score
                best_score = score
                best_ref = candidate_ref
            elif score > second_score:
                second_score = score

        if best_ref and best_score >= 1 and best_score >= second_score:
            aliases[ref] = best_ref
            added += 1
        else:
            unresolved_after += 1

    stats = {
        "added_aliases": added,
        "unresolved_before": len(unresolved),
        "unresolved_after": unresolved_after,
    }
    return stats


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
    parser.add_argument(
        "--bdb-source",
        choices=["sefaria", "openscriptures"],
        default="sefaria",
        help="BDB source provider (default: sefaria)",
    )
    parser.add_argument(
        "--bdb-sefaria-raw",
        default="data/bdb_sefaria_raw.json",
        help="Cache path for raw Sefaria BDB crawl payload",
    )
    parser.add_argument(
        "--refresh-bdb-sefaria",
        action="store_true",
        help="Refresh Sefaria BDB raw cache from network",
    )
    parser.add_argument(
        "--bdb-sefaria-delay-ms",
        type=int,
        default=0,
        help="Delay between Sefaria requests when refreshing raw cache",
    )
    parser.add_argument(
        "--bdb-sefaria-checkpoint-every",
        type=int,
        default=25,
        help="Checkpoint interval while crawling Sefaria BDB",
    )
    parser.add_argument(
        "--bdb-sefaria-progress-every",
        type=int,
        default=500,
        help="Progress print interval while crawling Sefaria BDB",
    )
    parser.add_argument(
        "--bdb-legacy-bridge",
        default="data/bdb_legacy_snapshot.json",
        help=(
            "Optional legacy BDB snapshot JSON used only for ID bridging "
            "(legacy-id aliases and gap supplements)."
        ),
    )
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
        strongs_path = resolve_lexicon_file(
            lex_root,
            ["HebrewStrong.xml", "StrongHebrew.xml", "hebrewstrong.xml", "stronghebrew.xml"],
            "HebrewStrong XML",
        )

        strong_to_bdb = parse_lexical_index(lexical_index_path)
        relink_bdb_codes(wlc_data, strong_to_bdb)

        if args.bdb_source == "openscriptures":
            bdb_path = resolve_lexicon_file(
                lex_root,
                ["BrownDriverBriggs.xml", "browndriverbriggs.xml"],
                "BrownDriverBriggs XML",
            )
            bdb_data = parse_bdb(bdb_path)
        else:
            raw_cache_path = Path(args.bdb_sefaria_raw)
            if args.refresh_bdb_sefaria or not raw_cache_path.exists():
                raw = crawl_bdb_sefaria_raw(
                    delay_ms=args.bdb_sefaria_delay_ms,
                    resume_path=raw_cache_path,
                    checkpoint_every=args.bdb_sefaria_checkpoint_every,
                    progress_every=args.bdb_sefaria_progress_every,
                )
                save_json(raw_cache_path, raw)
            else:
                raw = load_raw(raw_cache_path)
            bdb_data = build_bdb_json_from_raw(raw)
        metadata = bdb_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if args.bdb_source == "sefaria":
            metadata.setdefault("source_provider", "Sefaria")
            metadata.setdefault("source_lexicon_name", "BDB Dictionary")
            metadata.setdefault("rendering_id", "bdb.sefaria")
            metadata.setdefault("rendering_name", "BDB Dictionary (Sefaria)")
        bdb_data["metadata"] = metadata
        strongs_data = parse_strongs_hebrew(strongs_path)

        if args.bdb_source == "sefaria":
            legacy_entries = load_legacy_bdb_entries(args.bdb_legacy_bridge)
            if legacy_entries:
                aliases, alias_stats = build_bdb_legacy_aliases_from_snapshot(bdb_data, legacy_entries)
                strongs_alias_stats = add_strongs_alias_fallbacks(
                    wlc_data=wlc_data,
                    strongs_data=strongs_data,
                    bdb_data=bdb_data,
                    aliases=aliases,
                )
                supplemented, supplement_stats = supplement_missing_legacy_entries(
                    wlc_data=wlc_data,
                    bdb_data=bdb_data,
                    legacy_entries=legacy_entries,
                    aliases=aliases,
                )
            else:
                aliases = {}
                alias_stats = {
                    "legacy_entries": 0,
                    "aliases_built": 0,
                    "mapped_by_headword": 0,
                    "mapped_by_keyword_fallback": 0,
                    "unresolved": 0,
                }
                strongs_alias_stats = {
                    "added_aliases": 0,
                    "unresolved_before": 0,
                    "unresolved_after": 0,
                }
                supplemented = []
                supplement_stats = {
                    "wlc_refs_total": len(collect_wlc_bdb_refs(wlc_data)),
                    "missing_before_supplement": 0,
                    "supplemented_entries": 0,
                    "still_missing_after_supplement": 0,
                }

            bdb_data["aliases"] = aliases
            metadata = bdb_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["legacy_alias_stats"] = alias_stats
            metadata["strongs_alias_fallback_stats"] = strongs_alias_stats
            metadata["legacy_supplement_stats"] = supplement_stats
            metadata["legacy_bridge_source"] = args.bdb_legacy_bridge
            metadata["legacy_supplement_enabled"] = bool(legacy_entries)
            metadata["legacy_supplemented_refs"] = supplemented[:20]
            bdb_data["metadata"] = metadata

    (out_dir / "wlc_full.json").write_text(json.dumps(wlc_data, ensure_ascii=False), encoding="utf-8")
    (out_dir / "bdb_full.json").write_text(json.dumps(bdb_data, ensure_ascii=False), encoding="utf-8")
    (out_dir / "strongs_full.json").write_text(json.dumps(strongs_data, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_dir / 'wlc_full.json'}")
    print(f"Wrote {out_dir / 'bdb_full.json'}")
    print(f"Wrote {out_dir / 'strongs_full.json'}")


if __name__ == "__main__":
    main()
