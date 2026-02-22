#!/usr/bin/env python3
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from itertools import product
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

SEFARIA_BDB_INDEX_URL = "https://www.sefaria.org/api/index/BDB"
SEFARIA_BDB_TEXT_URL = "https://www.sefaria.org/api/texts/{ref}?context=0&pad=0"
SEFARIA_BDB_BULKTEXT_URL = "https://www.sefaria.org/api/bulktext/{ref}"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def normalize_hebrew(value: str) -> str:
    # Compare headwords independent of cantillation/niqqud.
    decomposed = unicodedata.normalize("NFD", value or "")
    filtered = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", filtered).strip()


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    # Remove invisible bidi/zero-width marks and non-whitespace control chars.
    text = re.sub(r"[\u200c\u200d\u200e\u200f\u202a-\u202e]", "", text)
    text = "".join(ch for ch in text if not (unicodedata.category(ch).startswith("C") and ch not in "\n\r\t"))
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html or "")
    return normalize_text(parser.text())


def extract_headword_from_ref(ref: str) -> str:
    if "," not in ref:
        return ref.strip()
    return ref.split(",", 1)[1].strip()


def _fetch_json(url: str, timeout: int = 30, retries: int = 7, base_backoff: float = 1.25):
    req = urllib.request.Request(url, headers={"User-Agent": "peter-test-bdb-importer/1.0"})
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except HTTPError as err:
            last_err = err
            # Retry transient server/rate-limit failures.
            if err.code not in {429, 500, 502, 503, 504}:
                raise
        except URLError as err:
            last_err = err
        except TimeoutError as err:
            last_err = err

        if attempt < retries:
            sleep_s = base_backoff * (2 ** (attempt - 1))
            time.sleep(min(sleep_s, 20.0))

    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to fetch JSON payload and no error was captured")


def _fetch_bulktext_entry(ref: str):
    url = SEFARIA_BDB_BULKTEXT_URL.format(ref=urllib.parse.quote(ref, safe=""))
    payload = _fetch_json(url)
    if not isinstance(payload, dict):
        return None
    entry = payload.get(ref)
    if not isinstance(entry, dict):
        return None
    return entry


def _hebrew_headword_from_ref(ref: str) -> str:
    if "," not in ref:
        return ref.strip()
    return ref.split(",", 1)[1].strip()


def _bdb_ref_with_headword(original_ref: str, headword: str) -> str:
    if "," not in original_ref:
        return headword
    prefix = original_ref.split(",", 1)[0].strip()
    return f"{prefix}, {headword}"


def _strip_hebrew_accents(value: str) -> str:
    # Keep niqqud but remove cantillation/accents.
    return "".join(ch for ch in value if not (0x0591 <= ord(ch) <= 0x05AF))


def _strip_all_marks(value: str) -> str:
    d = unicodedata.normalize("NFD", value)
    d = "".join(ch for ch in d if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", d)


def _headword_variants(headword: str):
    # Sefaria occasionally serves malformed niqqud forms in `next`.
    # Generate conservative variants to find canonical refs.
    VOWEL_SWAPS = {
        "\u05b7": ("\u05b7", "\u05b8"),  # patah <-> qamats
        "\u05b8": ("\u05b8", "\u05b7"),
        "\u05b6": ("\u05b6", "\u05b5"),  # segol <-> tsere
        "\u05b5": ("\u05b5", "\u05b6"),
    }

    forms = [headword, _strip_hebrew_accents(headword), _strip_all_marks(headword)]
    for form in list(forms):
        pools = []
        for ch in form:
            pools.append(VOWEL_SWAPS.get(ch, (ch,)))
        count = 1
        for pool in pools:
            count *= len(pool)
        if count > 64:
            continue
        for chars in product(*pools):
            forms.append("".join(chars))

    seen = set()
    out = []
    for f in forms:
        s = f.strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _resolve_bad_ref(ref: str):
    """
    Recover from Sefaria 400 on malformed `next` refs:
    - fetch entry text via bulktext using original ref
    - find a canonical candidate whose `prev` points back to original ref
    """
    bulk_entry = _fetch_bulktext_entry(ref)
    if bulk_entry is None:
        return None

    headword = _hebrew_headword_from_ref(ref)
    continuation_payload = None
    continuation_ref = None
    for variant in _headword_variants(headword):
        candidate_ref = _bdb_ref_with_headword(ref, variant)
        if candidate_ref == ref:
            continue
        url = SEFARIA_BDB_TEXT_URL.format(ref=urllib.parse.quote(candidate_ref, safe=""))
        try:
            candidate_payload = _fetch_json(url)
        except Exception:
            continue
        if str(candidate_payload.get("prev") or "") == ref:
            continuation_payload = candidate_payload
            continuation_ref = str(candidate_payload.get("ref") or candidate_ref)
            break

    text = bulk_entry.get("en")
    if not isinstance(text, str) or not text.strip():
        return None

    resolved = {
        "ref": ref,
        "heRef": str(bulk_entry.get("heRef") or ""),
        "text": [text],
        "prev": None,  # caller fills actual previous ref from local chain
        "next": continuation_ref,
    }
    return resolved, continuation_payload


def _payload_to_entry(payload: dict, fallback_ref: str):
    text = payload.get("text")
    if not isinstance(text, list) or len(text) != 1:
        raise RuntimeError(f"Unexpected text payload at {fallback_ref}; expected single-item list")
    return {
        "ref": str(payload.get("ref") or fallback_ref),
        "heRef": str(payload.get("heRef") or ""),
        "prev": payload.get("prev"),
        "next": payload.get("next"),
        "text_html": str(text[0]),
    }


def _next_anchor_not_seen(letter_refs: list[str], seen: set[str]):
    for anchor in letter_refs:
        if anchor not in seen:
            return anchor
    return None


def _recover_gap_from_anchor(letter_refs: list[str], seen: set[str], limit: int = 6000):
    anchor = _next_anchor_not_seen(letter_refs, seen)
    if not anchor:
        return []

    chain = []
    ref = anchor
    for _ in range(limit):
        payload = _fetch_json(SEFARIA_BDB_TEXT_URL.format(ref=urllib.parse.quote(ref, safe="")))
        entry = _payload_to_entry(payload, ref)
        current_ref = entry["ref"]
        if current_ref in seen:
            break
        chain.append(entry)
        prev_ref = entry.get("prev")
        if not prev_ref:
            # Could not reconnect.
            return []
        if prev_ref in seen:
            break
        ref = prev_ref

    if not chain:
        return []
    # chain is anchor->... backwards; return forward order.
    return list(reversed(chain))


def _first_chain_break(entries: list[dict]):
    for i in range(len(entries) - 1):
        a = entries[i]
        b = entries[i + 1]
        a_ref = str(a.get("ref") or "")
        b_ref = str(b.get("ref") or "")
        if a.get("next") != b_ref:
            return i, f"next mismatch at {a_ref}: {a.get('next')!r} != {b_ref!r}"
        if b.get("prev") != a_ref:
            return i, f"prev mismatch at {b_ref}: {b.get('prev')!r} != {a_ref!r}"
    return None


def _dictionary_node(index_payload: dict):
    schema = index_payload.get("schema", {})
    nodes = schema.get("nodes", [])
    for node in nodes:
        if node.get("nodeType") == "DictionaryNode":
            return node
    raise RuntimeError("Could not locate DictionaryNode in Sefaria BDB index payload")


def crawl_bdb_sefaria_raw(
    delay_ms: int = 0,
    max_entries: int | None = None,
    resume_path: Path | None = None,
    checkpoint_every: int = 25,
    progress_every: int = 500,
):
    index = _fetch_json(SEFARIA_BDB_INDEX_URL)
    node = _dictionary_node(index)

    headword_map = node.get("headwordMap") or []
    if not headword_map:
        raise RuntimeError("BDB DictionaryNode has no headwordMap")

    start_ref = str(headword_map[0][1])
    letter_refs = [str(item[1]) for item in headword_map]

    raw = {
        "source": {
            "provider": "Sefaria",
            "index_url": SEFARIA_BDB_INDEX_URL,
            "text_url_template": SEFARIA_BDB_TEXT_URL,
        },
        "index": {
            "title": index.get("title"),
            "lexicon_name": node.get("lexiconName"),
            "first_word": node.get("firstWord"),
            "last_word": node.get("lastWord"),
            "headword_map": headword_map,
            "letter_refs": letter_refs,
            "start_ref": start_ref,
        },
        "crawl": {
            "crawled_at": None,
            "delay_ms": delay_ms,
            "max_entries": max_entries,
            "checkpoint_every": checkpoint_every,
            "progress_every": progress_every,
            "completed": False,
        },
        "entries": [],
    }

    entries = []
    seen = set()
    ref = start_ref
    resumed = False

    if resume_path and resume_path.exists():
        try:
            prior = load_raw(resume_path)
        except Exception:
            prior = None

        if isinstance(prior, dict):
            prior_start_ref = str(prior.get("index", {}).get("start_ref") or "")
            prior_entries = prior.get("entries")
            if prior_start_ref == start_ref and isinstance(prior_entries, list) and prior_entries:
                refs = [str(e.get("ref") or "") for e in prior_entries if isinstance(e, dict)]
                if refs and len(refs) == len(set(refs)):
                    raw = prior
                    entries = list(prior_entries)
                    chain_break = _first_chain_break(entries)
                    if chain_break is not None:
                        break_idx, reason = chain_break
                        entries = entries[: break_idx + 1]
                        raw["entries"] = entries
                        print(
                            f"Trimmed broken checkpoint chain at index {break_idx} ({reason}); will resume from repaired boundary",
                            flush=True,
                        )
                    seen = set(refs)
                    if chain_break is not None:
                        seen = set(str(e.get("ref") or "") for e in entries)
                    ref = prior_entries[-1].get("next")
                    if chain_break is not None:
                        ref = entries[-1].get("next")
                    resumed = True
                    raw.setdefault("crawl", {})
                    raw["crawl"]["resumed_from_entries"] = len(entries)
                    raw["crawl"]["completed"] = False
                    if ref is None:
                        # Don't assume complete based only on null next pointer from checkpoint.
                        # Sefaria occasionally yields broken terminal refs mid-chain.
                        checkpoint_errors = verify_raw_completeness(raw)
                        if not checkpoint_errors:
                            raw["crawl"]["crawled_at"] = datetime.now(timezone.utc).isoformat()
                            raw["crawl"]["completed"] = True
                            return raw
                        recovered = _recover_gap_from_anchor(letter_refs, seen)
                        if recovered:
                            if entries:
                                # Reconnect local chain at recovery boundary.
                                entries[-1]["next"] = recovered[0]["ref"]
                                recovered[0]["prev"] = entries[-1]["ref"]
                            for rec in recovered:
                                rec_ref = rec["ref"]
                                if rec_ref in seen:
                                    continue
                                entries.append(rec)
                                seen.add(rec_ref)
                            raw["entries"] = entries
                            ref = entries[-1].get("next")
                            print(
                                f"Backfilled {len(recovered)} entries from anchor recovery before resume; boundary now {entries[-1]['ref']}",
                                flush=True,
                            )
                        else:
                            ref = _next_anchor_not_seen(letter_refs, seen)
                            if not ref:
                                raise RuntimeError(
                                    "Checkpoint has null next but fails completeness checks, and no remaining anchor is available"
                                )
                            print(
                                f"Resuming from next missing letter anchor due to incomplete checkpoint: {ref}",
                                flush=True,
                            )

    while ref:
        if ref in seen:
            raise RuntimeError(f"Cycle detected while crawling BDB chain at {ref}")
        seen.add(ref)

        url = SEFARIA_BDB_TEXT_URL.format(ref=urllib.parse.quote(ref, safe=""))
        try:
            payload = _fetch_json(url)
        except HTTPError as err:
            if err.code == 400:
                resolved = _resolve_bad_ref(ref)
                if resolved is None:
                    if resume_path:
                        raw["entries"] = entries
                        save_json(resume_path, raw)
                    raise
                recovered_payload, continuation_payload = resolved
                recovered_payload["prev"] = entries[-1]["ref"] if entries else None
                payload = recovered_payload
                # Pre-warm and validate continuation if available.
                if continuation_payload is not None and payload.get("next") is None:
                    payload["next"] = continuation_payload.get("ref")
                print(f"Recovered malformed BDB ref via bulktext: {ref}", flush=True)
            else:
                if resume_path:
                    raw["entries"] = entries
                    save_json(resume_path, raw)
                raise
        except Exception:
            if resume_path:
                raw["entries"] = entries
                save_json(resume_path, raw)
            raise

        entry = _payload_to_entry(payload, ref)
        entries.append(entry)
        current_ref = entry["ref"]
        seen.add(current_ref)

        count = len(entries)
        if progress_every > 0 and count % progress_every == 0:
            print(f"Crawled {count} entries; current ref: {entry['ref']}", flush=True)

        if resume_path and checkpoint_every > 0 and count % checkpoint_every == 0:
            raw["entries"] = entries
            raw.setdefault("crawl", {})
            raw["crawl"]["last_checkpoint_at"] = datetime.now(timezone.utc).isoformat()
            save_json(resume_path, raw)

        if max_entries is not None and count >= max_entries:
            break
        ref = entry.get("next")

        # If chain unexpectedly terminates before index last_word, recover by re-joining from next anchor.
        if ref is None:
            last_word = normalize_hebrew(str(raw.get("index", {}).get("last_word") or ""))
            current_headword = normalize_hebrew(_hebrew_headword_from_ref(current_ref))
            if max_entries is None and (not last_word or current_headword != last_word):
                recovered = _recover_gap_from_anchor(letter_refs, seen)
                if recovered:
                    if entries:
                        # Reconnect local chain at recovery boundary.
                        entries[-1]["next"] = recovered[0]["ref"]
                        recovered[0]["prev"] = entries[-1]["ref"]
                    for rec in recovered:
                        rec_ref = rec["ref"]
                        if rec_ref in seen:
                            continue
                        entries.append(rec)
                        seen.add(rec_ref)
                        if progress_every > 0 and len(entries) % progress_every == 0:
                            print(f"Crawled {len(entries)} entries; current ref: {rec_ref}", flush=True)
                    print(
                        f"Recovered gap via anchor backfill; added {len(recovered)} entries, resumed at {entries[-1]['ref']}",
                        flush=True,
                    )
                    ref = entries[-1].get("next")
                    if resume_path and checkpoint_every > 0 and len(entries) % checkpoint_every == 0:
                        raw["entries"] = entries
                        raw.setdefault("crawl", {})
                        raw["crawl"]["last_checkpoint_at"] = datetime.now(timezone.utc).isoformat()
                        save_json(resume_path, raw)

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    raw["entries"] = entries
    raw.setdefault("crawl", {})
    raw["crawl"]["crawled_at"] = datetime.now(timezone.utc).isoformat()
    raw["crawl"]["delay_ms"] = delay_ms
    raw["crawl"]["max_entries"] = max_entries
    raw["crawl"]["completed"] = (ref is None and (max_entries is None or len(entries) < max_entries))
    if resumed:
        raw["crawl"]["resumed"] = True

    if resume_path:
        save_json(resume_path, raw)

    return raw


def verify_raw_completeness(raw: dict):
    errors = []
    entries = raw.get("entries")
    index = raw.get("index", {})
    if not isinstance(entries, list) or not entries:
        return ["Raw crawl has no entries"]

    refs = [str(e.get("ref") or "") for e in entries]
    unique_refs = set(refs)
    if len(unique_refs) != len(refs):
        errors.append("Duplicate refs detected in raw crawl entries")

    start_ref = str(index.get("start_ref") or "")
    if refs[0] != start_ref:
        errors.append(f"First ref mismatch: expected {start_ref!r}, got {refs[0]!r}")

    letter_refs = [str(x) for x in index.get("letter_refs", [])]
    missing_letter_refs = [r for r in letter_refs if r not in unique_refs]
    if missing_letter_refs:
        errors.append(f"Missing letter anchor refs: {missing_letter_refs[:5]}")

    for i, entry in enumerate(entries):
        ref = refs[i]
        prev_ref = entries[i - 1]["ref"] if i > 0 else None
        next_ref = entries[i + 1]["ref"] if i + 1 < len(entries) else None
        if i == 0 and entry.get("prev") is not None:
            errors.append(f"First entry has non-null prev: {entry.get('prev')!r}")
        if i > 0 and entry.get("prev") != prev_ref:
            errors.append(f"Broken prev chain at {ref}: expected {prev_ref!r}, got {entry.get('prev')!r}")
        if i + 1 < len(entries) and entry.get("next") != next_ref:
            errors.append(f"Broken next chain at {ref}: expected {next_ref!r}, got {entry.get('next')!r}")
        if i + 1 == len(entries) and entry.get("next") is not None:
            errors.append(f"Last entry has non-null next: {entry.get('next')!r}")

    last_word = normalize_hebrew(str(index.get("last_word") or ""))
    last_headword = normalize_hebrew(extract_headword_from_ref(refs[-1]))
    if last_word and last_word not in last_headword and last_headword not in last_word:
        errors.append(f"Last word mismatch: index={index.get('last_word')!r}, last_ref={refs[-1]!r}")

    return errors


def build_bdb_entries_from_raw(raw: dict):
    entries = {}
    for payload in raw.get("entries", []):
        ref = str(payload.get("ref") or "").strip()
        if not ref:
            continue
        headword = extract_headword_from_ref(ref)
        definition = html_to_text(str(payload.get("text_html") or ""))
        entries[ref] = {
            "headword": headword,
            "glosses": [],
            "definition": definition,
            "source_ref": ref,
            "source_he_ref": str(payload.get("heRef") or ""),
        }
    return entries


def build_bdb_json_from_raw(raw: dict):
    completeness_errors = verify_raw_completeness(raw)
    if completeness_errors:
        raise RuntimeError("Sefaria raw crawl completeness checks failed: " + "; ".join(completeness_errors[:5]))
    entries = build_bdb_entries_from_raw(raw)
    return {
        "metadata": {
            "source_provider": "Sefaria",
            "source_index_title": raw.get("index", {}).get("title"),
            "source_lexicon_name": raw.get("index", {}).get("lexicon_name"),
            "rendering_id": "bdb.sefaria",
            "rendering_name": "BDB Dictionary (Sefaria)",
            "source_first_word": raw.get("index", {}).get("first_word"),
            "source_last_word": raw.get("index", {}).get("last_word"),
            "entry_count": len(entries),
            "raw_crawled_at": raw.get("crawl", {}).get("crawled_at"),
        },
        "entries": entries,
    }


def load_raw(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
