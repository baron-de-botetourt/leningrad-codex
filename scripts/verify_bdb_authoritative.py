#!/usr/bin/env python3
import argparse
import json
import re
import unicodedata
from pathlib import Path

from bdb_sefaria_source import normalize_hebrew, verify_raw_completeness

STATUS_MARKERS = ("base", "made", "ref", "done")
FORMAT_CHAR_RE = re.compile(r"[\u200c\u200d\u200e\u200f\u202a-\u202e]")
TRAILING_STATUS_RE = re.compile(r"(?:[\s;,\-]+(?:base|made|ref|done))\s*$", re.IGNORECASE)

# Scan-backed pages:
# https://en.wikisource.org/wiki/Page:A_Hebrew_and_English_Lexicon_(Brown-Driver-Briggs).djvu/25
# https://en.wikisource.org/wiki/Page:A_Hebrew_and_English_Lexicon_(Brown-Driver-Briggs).djvu/26
AUTHORITATIVE_SAMPLES = [
    {
        "entry_id": "BDB, א",
        "headword": "א",
        "source": "https://en.wikisource.org/wiki/Page:A_Hebrew_and_English_Lexicon_(Brown-Driver-Briggs).djvu/25",
        "must_contain": [
            "first letter; in post Biblical Hebrew = numeral 1",
            "no evidence of this usage in OT times",
        ],
    },
    {
        "entry_id": "BDB, אָבִיב",
        "headword": "אָבִיב",
        "source": "https://en.wikisource.org/wiki/Page:A_Hebrew_and_English_Lexicon_(Brown-Driver-Briggs).djvu/25",
        "must_contain": [
            "fresh, young ears of barley",
            "month of Exodus & passover",
        ],
    },
    {
        "entry_id": "BDB, אֲבַגְתָא",
        "headword": "אֲבַגְתָא",
        "source": "https://en.wikisource.org/wiki/Page:A_Hebrew_and_English_Lexicon_(Brown-Driver-Briggs).djvu/25",
        "must_contain": [
            "eunuch of Ahasuerus Est 1:10",
        ],
    },
    {
        "entry_id": "BDB, אָבַד",
        "headword": "אָבַד",
        "source": "https://en.wikisource.org/wiki/Page:A_Hebrew_and_English_Lexicon_(Brown-Driver-Briggs).djvu/25",
        "must_contain": [
            "vb. perish",
        ],
        "must_contain_any": [
            "be lost, property",
            "strayed, beasts",
        ],
    },
    {
        "entry_id": "BDB, אֹבֵד",
        "headword": "אֹבֵד",
        "source": "https://en.wikisource.org/wiki/Page:A_Hebrew_and_English_Lexicon_(Brown-Driver-Briggs).djvu/26",
        "must_contain": [
            "destruction",
        ],
    },
    {
        "entry_id": "BDB, אֲבֵדָה",
        "headword": "אֲבֵדָה",
        "source": "https://en.wikisource.org/wiki/Page:A_Hebrew_and_English_Lexicon_(Brown-Driver-Briggs).djvu/26",
        "must_contain": [
            "n.f. a lost thing",
        ],
    },
    {
        "entry_id": "BDB, אָב²",
        "headword": "אָב²",
        "source": "https://www.sefaria.org/BDB%2C_%D7%90%D6%B8%D7%91%C2%B2?lang=bi",
        "must_contain": [
            "אָב1191 n.m. father",
            "father of individual",
        ],
    },
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = FORMAT_CHAR_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_candidate_entries(entries: dict, headword: str):
    target = normalize_hebrew(headword)
    out = []
    for entry_id, payload in entries.items():
        if not isinstance(payload, dict):
            continue
        cand = normalize_hebrew(payload.get("headword", ""))
        if cand == target:
            out.append((entry_id, payload))
    return out


def check_authoritative_samples(entries: dict):
    errors = []
    for sample in AUTHORITATIVE_SAMPLES:
        source = sample["source"]
        headword = sample["headword"]
        entry_id = sample.get("entry_id")
        candidates = []
        if entry_id:
            payload = entries.get(entry_id)
            if isinstance(payload, dict):
                candidates = [(entry_id, payload)]
        if not candidates:
            candidates = find_candidate_entries(entries, headword)
        if not candidates:
            errors.append(f"Missing headword {headword!r} (source: {source})")
            continue

        matched = False
        for entry_id, entry in candidates:
            definition = normalize_text(entry.get("definition", ""))
            if not all(normalize_text(phrase) in definition for phrase in sample["must_contain"]):
                continue
            any_phrases = sample.get("must_contain_any", [])
            if any_phrases and not any(normalize_text(phrase) in definition for phrase in any_phrases):
                continue
            if normalize_text(headword) and normalize_text(headword) not in normalize_text(entry.get("headword", "")):
                # Entry-id fast path should still align to expected headword.
                continue
            matched = True
            break

        if not matched:
            errors.append(
                f"Headword {headword!r} present, but expected phrases not found (source: {source})"
            )
    return errors


def audit_encoding(entries: dict):
    total = len(entries)
    non_empty_headwords = 0
    nfc_mismatches = 0
    replacement_char_fields = 0
    control_char_fields = 0
    trailing_status = 0

    sample_control = []
    sample_status = []
    sample_headword_missing = []
    supplemental_entries = 0

    for entry_id, payload in entries.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("source_note") == "supplemented_from_legacy_bridge":
            supplemental_entries += 1

        headword = payload.get("headword", "")
        if normalize_text(headword):
            non_empty_headwords += 1
        elif len(sample_headword_missing) < 10:
            sample_headword_missing.append(entry_id)

        definition = payload.get("definition", "")
        normalized_definition = normalize_text(definition)
        if TRAILING_STATUS_RE.search(normalized_definition):
            trailing_status += 1
            if len(sample_status) < 10:
                sample_status.append(entry_id)

        for field in ("headword", "definition"):
            value = payload.get(field, "")
            if not isinstance(value, str):
                continue
            if "\ufffd" in value:
                replacement_char_fields += 1
            if unicodedata.normalize("NFC", value) != value:
                nfc_mismatches += 1

            bad = []
            for ch in value:
                category = unicodedata.category(ch)
                if category == "Cf":
                    bad.append(f"U+{ord(ch):04X}")
                elif category.startswith("C") and ch not in "\n\r\t":
                    bad.append(f"U+{ord(ch):04X}")
            if bad:
                control_char_fields += 1
                if len(sample_control) < 10:
                    sample_control.append((entry_id, field, ",".join(sorted(set(bad)))))

    headword_ratio = (non_empty_headwords / total) if total else 0.0
    return {
        "total_entries": total,
        "non_empty_headwords": non_empty_headwords,
        "headword_ratio": headword_ratio,
        "nfc_mismatch_fields": nfc_mismatches,
        "replacement_char_fields": replacement_char_fields,
        "control_char_fields": control_char_fields,
        "trailing_status_entries": trailing_status,
        "sample_control": sample_control,
        "sample_status": sample_status,
        "sample_headword_missing": sample_headword_missing,
        "supplemental_entries": supplemental_entries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify BDB JSON against scan-backed samples, encoding rules, and raw source completeness."
    )
    parser.add_argument("--bdb", default="data/bdb_full.json", help="Path to BDB JSON")
    parser.add_argument(
        "--raw",
        default="data/bdb_sefaria_raw.json",
        help="Path to cached raw Sefaria crawl payload",
    )
    parser.add_argument("--min-entries", type=int, default=10000, help="Minimum acceptable entry count")
    parser.add_argument(
        "--min-headword-ratio",
        type=float,
        default=0.99,
        help="Minimum fraction of entries that must have non-empty headword",
    )
    args = parser.parse_args()

    bdb = load_json(Path(args.bdb))
    entries = bdb.get("entries")
    if not isinstance(entries, dict):
        print("ERROR: BDB top-level 'entries' must be an object")
        raise SystemExit(1)

    raw = load_json(Path(args.raw))
    raw_errors = verify_raw_completeness(raw)

    sample_errors = check_authoritative_samples(entries)
    audit = audit_encoding(entries)

    errors = []
    errors.extend(raw_errors)
    errors.extend(sample_errors)

    if audit["total_entries"] < args.min_entries:
        errors.append(f"Entry count too low: {audit['total_entries']} < {args.min_entries}")
    if audit["headword_ratio"] < args.min_headword_ratio:
        errors.append(
            f"Headword coverage too low: {audit['non_empty_headwords']}/{audit['total_entries']} "
            f"({audit['headword_ratio']:.2%}) < {args.min_headword_ratio:.2%}"
        )
    raw_entry_count = len(raw.get("entries", []))
    expected_total = raw_entry_count + audit["supplemental_entries"]
    if audit["total_entries"] != expected_total:
        errors.append(
            "Raw/parsed entry count mismatch after supplement adjustment: "
            f"raw={raw_entry_count}, supplemental={audit['supplemental_entries']}, "
            f"parsed={audit['total_entries']}"
        )
    if audit["trailing_status_entries"] > 0:
        errors.append(
            f"Found {audit['trailing_status_entries']} entries ending with editorial status markers "
            f"({', '.join(STATUS_MARKERS)})"
        )
    if audit["replacement_char_fields"] > 0:
        errors.append(f"Found replacement-character corruption in {audit['replacement_char_fields']} fields")
    if audit["control_char_fields"] > 0:
        errors.append(f"Found Unicode control/format characters in {audit['control_char_fields']} fields")
    if audit["nfc_mismatch_fields"] > 0:
        errors.append(f"Found {audit['nfc_mismatch_fields']} non-NFC text fields")

    print("BDB verification report")
    print(f"- Parsed entries: {audit['total_entries']}")
    print(f"- Raw cached entries: {raw_entry_count}")
    print(f"- Supplemental bridge entries: {audit['supplemental_entries']}")
    print(f"- Authoritative sample checks: {len(AUTHORITATIVE_SAMPLES)}")
    print(f"- Non-empty headwords: {audit['non_empty_headwords']} ({audit['headword_ratio']:.2%})")
    print(f"- Trailing status markers: {audit['trailing_status_entries']}")
    print(f"- Replacement-char fields: {audit['replacement_char_fields']}")
    print(f"- Unicode control/format fields: {audit['control_char_fields']}")
    print(f"- Non-NFC fields: {audit['nfc_mismatch_fields']}")
    print(f"- Raw completeness errors: {len(raw_errors)}")

    if errors:
        print("\nFailures:")
        for err in errors:
            print(f"- {err}")

    if audit["sample_headword_missing"]:
        print("\nSample missing headwords:")
        for entry_id in audit["sample_headword_missing"]:
            print(f"- {entry_id}")

    if audit["sample_status"]:
        print("\nSample trailing status entries:")
        for entry_id in audit["sample_status"]:
            print(f"- {entry_id}")

    if audit["sample_control"]:
        print("\nSample control/format characters:")
        for entry_id, field, bad in audit["sample_control"]:
            print(f"- {entry_id}.{field}: {bad}")

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
