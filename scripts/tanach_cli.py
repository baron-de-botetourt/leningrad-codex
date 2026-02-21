#!/usr/bin/env python3
"""Headless CLI for Tanach text analysis, lexicon integration, and annotation layers."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DB_PATH = Path("data/tanach_cli.sqlite")
DEFAULT_WLC_PATH = Path("data/wlc_full.json")
DEFAULT_BDB_PATH = Path("data/bdb_full.json")
DEFAULT_STRONGS_PATH = Path("data/strongs_full.json")


BOOK_ORDER = [
    "Gen",
    "Exod",
    "Lev",
    "Num",
    "Deut",
    "Josh",
    "Judg",
    "Ruth",
    "1Sam",
    "2Sam",
    "1Kgs",
    "2Kgs",
    "Isa",
    "Jer",
    "Ezek",
    "Hos",
    "Joel",
    "Amos",
    "Obad",
    "Jonah",
    "Mic",
    "Nah",
    "Hab",
    "Zeph",
    "Hag",
    "Zech",
    "Mal",
    "Ps",
    "Prov",
    "Job",
    "Song",
    "Lam",
    "Eccl",
    "Esth",
    "Dan",
    "Ezra",
    "Neh",
    "1Chr",
    "2Chr",
]


BOOK_ALIASES = {
    "gen": "Gen",
    "genesis": "Gen",
    "exod": "Exod",
    "exodus": "Exod",
    "lev": "Lev",
    "leviticus": "Lev",
    "num": "Num",
    "numbers": "Num",
    "deut": "Deut",
    "deuteronomy": "Deut",
    "josh": "Josh",
    "joshua": "Josh",
    "judg": "Judg",
    "judges": "Judg",
    "ruth": "Ruth",
    "1sam": "1Sam",
    "1samuel": "1Sam",
    "2sam": "2Sam",
    "2samuel": "2Sam",
    "1kgs": "1Kgs",
    "1kings": "1Kgs",
    "2kgs": "2Kgs",
    "2kings": "2Kgs",
    "1chr": "1Chr",
    "1chron": "1Chr",
    "1chronicles": "1Chr",
    "2chr": "2Chr",
    "2chron": "2Chr",
    "2chronicles": "2Chr",
    "ezra": "Ezra",
    "neh": "Neh",
    "nehemiah": "Neh",
    "esth": "Esth",
    "esther": "Esth",
    "job": "Job",
    "ps": "Ps",
    "psalm": "Ps",
    "psalms": "Ps",
    "prov": "Prov",
    "proverbs": "Prov",
    "eccl": "Eccl",
    "ecclesiastes": "Eccl",
    "song": "Song",
    "songofsongs": "Song",
    "sos": "Song",
    "isa": "Isa",
    "isaiah": "Isa",
    "jer": "Jer",
    "jeremiah": "Jer",
    "lam": "Lam",
    "lamentations": "Lam",
    "ezek": "Ezek",
    "ezekiel": "Ezek",
    "dan": "Dan",
    "daniel": "Dan",
    "hos": "Hos",
    "hosea": "Hos",
    "joel": "Joel",
    "amos": "Amos",
    "obad": "Obad",
    "obadiah": "Obad",
    "jonah": "Jonah",
    "mic": "Mic",
    "micah": "Mic",
    "nah": "Nah",
    "nahum": "Nah",
    "hab": "Hab",
    "habakkuk": "Hab",
    "zeph": "Zeph",
    "zephaniah": "Zeph",
    "hag": "Hag",
    "haggai": "Hag",
    "zech": "Zech",
    "zechariah": "Zech",
    "mal": "Mal",
    "malachi": "Mal",
}


CREATE_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tokens (
  book TEXT NOT NULL,
  chapter INTEGER NOT NULL,
  verse INTEGER NOT NULL,
  word_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  lemma TEXT NOT NULL,
  morph TEXT NOT NULL,
  bdb TEXT NOT NULL,
  PRIMARY KEY (book, chapter, verse, word_index)
);

CREATE INDEX IF NOT EXISTS idx_tokens_ref ON tokens (book, chapter, verse, word_index);
CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens (lemma);
CREATE INDEX IF NOT EXISTS idx_tokens_morph ON tokens (morph);
CREATE INDEX IF NOT EXISTS idx_tokens_bdb ON tokens (bdb);
CREATE INDEX IF NOT EXISTS idx_tokens_text ON tokens (text);

CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  path TEXT NOT NULL,
  root_path TEXT NOT NULL DEFAULT 'entries',
  key_field TEXT NOT NULL DEFAULT '',
  lookup_token_field TEXT NOT NULL DEFAULT 'lemma',
  enabled INTEGER NOT NULL DEFAULT 1,
  config_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS source_entries (
  source_id TEXT NOT NULL,
  entry_key TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (source_id, entry_key),
  FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_source_entries_source ON source_entries (source_id);

CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  layer TEXT NOT NULL,
  target_kind TEXT NOT NULL CHECK (target_kind IN ('word', 'verse', 'passage')),
  book TEXT NOT NULL,
  chapter_start INTEGER NOT NULL,
  verse_start INTEGER NOT NULL,
  chapter_end INTEGER NOT NULL,
  verse_end INTEGER NOT NULL,
  word_index INTEGER NOT NULL DEFAULT -1,
  value_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(layer, target_kind, book, chapter_start, verse_start, chapter_end, verse_end, word_index)
);

CREATE INDEX IF NOT EXISTS idx_annotations_lookup
  ON annotations (layer, book, chapter_start, verse_start, chapter_end, verse_end, target_kind, word_index);

CREATE TABLE IF NOT EXISTS translations (
  version TEXT NOT NULL,
  book TEXT NOT NULL,
  chapter INTEGER NOT NULL,
  verse INTEGER NOT NULL,
  text TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (version, book, chapter, verse)
);

CREATE INDEX IF NOT EXISTS idx_translations_ref ON translations (book, chapter, verse);
"""


class CLIError(Exception):
    """Expected CLI error with user-friendly message."""


@dataclass(frozen=True)
class VerseRef:
    book: str
    chapter: int
    verse: int

    def label(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"

    def scalar(self) -> int:
        return self.chapter * 1000 + self.verse


def normalize_book_key(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.lower())


def canonical_book_order_key(book: str) -> tuple[int, str]:
    try:
        return (BOOK_ORDER.index(book), book)
    except ValueError:
        return (999, book)


def connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(CREATE_SCHEMA_SQL)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def iterate_wlc_tokens(wlc_data: dict[str, Any]) -> Iterable[tuple[str, int, int, int, str, str, str, str]]:
    books = wlc_data.get("books")
    if not isinstance(books, dict):
        raise CLIError("WLC JSON is missing a top-level 'books' object.")

    for book in sorted(books.keys(), key=canonical_book_order_key):
        book_data = books.get(book, {})
        chapters = book_data.get("chapters", {})
        if not isinstance(chapters, dict):
            continue
        for chapter_key in sorted(chapters.keys(), key=lambda x: int(x)):
            chapter_data = chapters.get(chapter_key, {})
            verses = chapter_data.get("verses", {})
            if not isinstance(verses, dict):
                continue
            for verse_key in sorted(verses.keys(), key=lambda x: int(x)):
                verse_data = verses.get(verse_key, {})
                words = verse_data.get("words", [])
                if not isinstance(words, list):
                    continue
                for word_index, word in enumerate(words):
                    if not isinstance(word, dict):
                        continue
                    yield (
                        book,
                        int(chapter_key),
                        int(verse_key),
                        int(word_index),
                        as_text(word.get("text")),
                        as_text(word.get("lemma")),
                        as_text(word.get("morph")),
                        as_text(word.get("bdb")),
                    )


def upsert_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, json.dumps(value, ensure_ascii=False)),
    )


def resolve_path(path_text: str) -> Path:
    return Path(path_text).expanduser()


def root_select(value: Any, root_path: str) -> Any:
    if not root_path:
        return value
    current = value
    for segment in [part for part in root_path.split(".") if part]:
        if isinstance(current, dict):
            if segment not in current:
                raise CLIError(f"Source root path segment not found: {segment}")
            current = current[segment]
            continue
        if isinstance(current, list):
            try:
                index = int(segment)
            except ValueError as exc:
                raise CLIError(f"Source root path segment must be a list index: {segment}") from exc
            if index < 0 or index >= len(current):
                raise CLIError(f"Source root path index out of range: {index}")
            current = current[index]
            continue
        raise CLIError(f"Source root path cannot traverse through a {type(current).__name__}.")
    return current


def extract_source_rows(source_doc: Any, root_path: str, key_field: str) -> Iterable[tuple[str, str]]:
    root = root_select(source_doc, root_path)

    if isinstance(root, dict):
        if not key_field:
            for key, payload in root.items():
                yield str(key), json.dumps(payload, ensure_ascii=False)
            return
        for payload in root.values():
            if not isinstance(payload, dict):
                continue
            key = payload.get(key_field)
            if key is None:
                continue
            yield str(key), json.dumps(payload, ensure_ascii=False)
        return

    if isinstance(root, list):
        if not key_field:
            raise CLIError("List-based source roots require --key-field.")
        for payload in root:
            if not isinstance(payload, dict):
                continue
            key = payload.get(key_field)
            if key is None:
                continue
            yield str(key), json.dumps(payload, ensure_ascii=False)
        return

    raise CLIError("Source root must resolve to an object or array.")


def source_exists(conn: sqlite3.Connection, source_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    return row is not None


def register_source(
    conn: sqlite3.Connection,
    source_id: str,
    source_type: str,
    path: str,
    root_path: str,
    key_field: str,
    lookup_token_field: str,
    config_json: str,
) -> None:
    conn.execute(
        """
        INSERT INTO sources (
          source_id, source_type, path, root_path, key_field, lookup_token_field, config_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source_id) DO UPDATE SET
          source_type = excluded.source_type,
          path = excluded.path,
          root_path = excluded.root_path,
          key_field = excluded.key_field,
          lookup_token_field = excluded.lookup_token_field,
          config_json = excluded.config_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (source_id, source_type, path, root_path, key_field, lookup_token_field, config_json),
    )


def ingest_source(conn: sqlite3.Connection, source_id: str) -> int:
    source = conn.execute(
        """
        SELECT source_id, path, root_path, key_field
        FROM sources
        WHERE source_id = ? AND enabled = 1
        """,
        (source_id,),
    ).fetchone()
    if source is None:
        raise CLIError(f"Unknown or disabled source: {source_id}")

    source_path = resolve_path(source["path"])
    if not source_path.exists():
        raise CLIError(f"Source file does not exist: {source_path}")

    source_doc = read_json(source_path)
    rows = list(extract_source_rows(source_doc, source["root_path"], source["key_field"]))

    conn.execute("DELETE FROM source_entries WHERE source_id = ?", (source_id,))
    conn.executemany(
        "INSERT INTO source_entries(source_id, entry_key, payload_json) VALUES (?, ?, ?)",
        ((source_id, key, payload_json) for key, payload_json in rows),
    )
    return len(rows)


def initialize_db(
    db_path: Path,
    wlc_path: Path,
    bdb_path: Path,
    strongs_path: Path,
    rebuild: bool,
) -> dict[str, Any]:
    if not wlc_path.exists():
        raise CLIError(f"WLC file does not exist: {wlc_path}")
    if not bdb_path.exists():
        raise CLIError(f"BDB file does not exist: {bdb_path}")

    wlc_data = read_json(wlc_path)

    conn = connect_db(db_path)
    try:
        ensure_schema(conn)
        with conn:
            if rebuild:
                conn.execute("DELETE FROM tokens")
                conn.execute("DELETE FROM source_entries")
                conn.execute("DELETE FROM sources")

            token_rows = list(iterate_wlc_tokens(wlc_data))
            conn.execute("DELETE FROM tokens")
            conn.executemany(
                """
                INSERT INTO tokens(book, chapter, verse, word_index, text, lemma, morph, bdb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                token_rows,
            )

            register_source(
                conn=conn,
                source_id="bdb",
                source_type="keyed-json",
                path=str(bdb_path),
                root_path="entries",
                key_field="",
                lookup_token_field="bdb",
                config_json="{}",
            )
            bdb_entry_count = ingest_source(conn, "bdb")
            strongs_entry_count = 0
            strongs_loaded = False
            if strongs_path.exists():
                register_source(
                    conn=conn,
                    source_id="strongs",
                    source_type="keyed-json",
                    path=str(strongs_path),
                    root_path="entries",
                    key_field="",
                    lookup_token_field="lemma",
                    config_json=json.dumps({"lemma_normalizer": "strongs"}),
                )
                strongs_entry_count = ingest_source(conn, "strongs")
                strongs_loaded = True

            books_count = conn.execute("SELECT COUNT(DISTINCT book) FROM tokens").fetchone()[0]
            chapter_count = conn.execute("SELECT COUNT(DISTINCT book || ':' || chapter) FROM tokens").fetchone()[0]
            verse_count = conn.execute("SELECT COUNT(DISTINCT book || ':' || chapter || ':' || verse) FROM tokens").fetchone()[0]

            upsert_meta(conn, "wlc_path", str(wlc_path))
            upsert_meta(conn, "bdb_path", str(bdb_path))
            upsert_meta(conn, "strongs_path", str(strongs_path))
            upsert_meta(
                conn,
                "counts",
                {
                    "books": books_count,
                    "chapters": chapter_count,
                    "verses": verse_count,
                    "tokens": len(token_rows),
                    "bdb_entries": bdb_entry_count,
                    "strongs_entries": strongs_entry_count,
                },
            )

        return {
            "db": str(db_path),
            "books": books_count,
            "chapters": chapter_count,
            "verses": verse_count,
            "tokens": len(token_rows),
            "bdb_entries": bdb_entry_count,
            "strongs_loaded": strongs_loaded,
            "strongs_entries": strongs_entry_count,
        }
    finally:
        conn.close()


def load_book_lookup(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT DISTINCT book FROM tokens").fetchall()
    books = [row["book"] for row in rows]
    if not books:
        books = list(BOOK_ORDER)

    lookup: dict[str, str] = {}
    for book in books:
        lookup[normalize_book_key(book)] = book
    for alias, canonical in BOOK_ALIASES.items():
        if canonical in books:
            lookup[normalize_book_key(alias)] = canonical
    return lookup


def parse_ref(text: str, book_lookup: dict[str, str]) -> VerseRef:
    value = text.strip()
    patterns = [
        r"^(.+?)\s+(\d+):(\d+)$",
        r"^(.+?)\s+(\d+)\s+(\d+)$",
        r"^([A-Za-z0-9]+)[.:](\d+)[.:](\d+)$",
    ]

    match: re.Match[str] | None = None
    for pattern in patterns:
        match = re.match(pattern, value)
        if match:
            break
    if not match:
        raise CLIError(f"Could not parse ref '{text}'. Use e.g. 'Gen 1:1'.")

    book_raw, chapter_raw, verse_raw = match.group(1), match.group(2), match.group(3)
    normalized = normalize_book_key(book_raw)
    if normalized not in book_lookup:
        known = ", ".join(sorted(set(book_lookup.values()), key=canonical_book_order_key))
        raise CLIError(f"Unknown book '{book_raw}'. Known books: {known}")
    return VerseRef(book_lookup[normalized], int(chapter_raw), int(verse_raw))


def parse_source_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def lemma_lookup_candidates(lemma: str) -> list[str]:
    raw = (lemma or "").strip()
    if not raw:
        return []

    candidates = [raw]
    no_prefix = re.sub(r"^(?:[a-z]/)+", "", raw)
    candidates.append(no_prefix)

    no_space_suffix = re.sub(r"\s+([a-zA-Z])$", r"\1", no_prefix)
    candidates.append(no_space_suffix)

    no_space_type = re.sub(r"\s+[a-zA-Z]$", "", no_prefix)
    candidates.append(no_space_type)

    compact = re.sub(r"\s+", "", no_prefix)
    candidates.append(compact)

    no_plus = re.sub(r"\+$", "", no_space_type)
    candidates.append(no_plus)
    candidates.append(re.sub(r"^(?:[a-z]/)+", "", no_plus))

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        value = candidate.strip()
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def row_to_token_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "book": row["book"],
        "chapter": row["chapter"],
        "verse": row["verse"],
        "word_index": row["word_index"],
        "text": row["text"],
        "lemma": row["lemma"],
        "morph": row["morph"],
        "bdb": row["bdb"],
    }


def check_token_table_has_data(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT COUNT(*) AS c FROM tokens").fetchone()
    if row is None or row["c"] == 0:
        raise CLIError("No token data found. Run 'scripts/tanach_cli.py init-db' first.")


def fetch_verse_tokens(conn: sqlite3.Connection, verse_ref: VerseRef) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT book, chapter, verse, word_index, text, lemma, morph, bdb
        FROM tokens
        WHERE book = ? AND chapter = ? AND verse = ?
        ORDER BY word_index
        """,
        (verse_ref.book, verse_ref.chapter, verse_ref.verse),
    ).fetchall()
    return [row_to_token_dict(row) for row in rows]


def load_source_config_map(conn: sqlite3.Connection, source_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not source_ids:
        return {}
    placeholders = ",".join(["?"] * len(source_ids))
    rows = conn.execute(
        f"""
        SELECT source_id, lookup_token_field, config_json
        FROM sources
        WHERE source_id IN ({placeholders}) AND enabled = 1
        """,
        tuple(source_ids),
    ).fetchall()
    config_map = {row["source_id"]: row for row in rows}
    missing = [source_id for source_id in source_ids if source_id not in config_map]
    if missing:
        raise CLIError(f"Unknown source(s): {', '.join(missing)}")
    return config_map


def lookup_source_payload(conn: sqlite3.Connection, source_id: str, entry_key: str) -> Any | None:
    row = conn.execute(
        """
        SELECT payload_json
        FROM source_entries
        WHERE source_id = ? AND entry_key = ?
        """,
        (source_id, entry_key),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["payload_json"])


def entry_key_for_token(token: dict[str, Any], source_config: sqlite3.Row) -> str:
    lookup_field = source_config["lookup_token_field"]
    if lookup_field not in {"text", "lemma", "morph", "bdb"}:
        raise CLIError(f"Invalid source lookup_token_field: {lookup_field}")
    return as_text(token.get(lookup_field))


def source_entry_keys_for_token(token: dict[str, Any], source_config: sqlite3.Row) -> list[str]:
    key = entry_key_for_token(token, source_config)
    keys = [key]

    config: dict[str, Any] = {}
    raw_config = source_config["config_json"]
    if isinstance(raw_config, str) and raw_config.strip():
        try:
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                config = parsed
        except json.JSONDecodeError:
            config = {}

    if source_config["lookup_token_field"] == "lemma" and config.get("lemma_normalizer") == "strongs":
        keys = lemma_lookup_candidates(key)

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in keys:
        value = candidate.strip()
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def lookup_source_payload_for_token(
    conn: sqlite3.Connection,
    source_id: str,
    token: dict[str, Any],
    source_config: sqlite3.Row,
) -> Any | None:
    for key in source_entry_keys_for_token(token, source_config):
        payload = lookup_source_payload(conn, source_id, key)
        if payload is not None:
            return payload
    return None


def extract_annotation_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for candidate in ("gloss", "text", "value", "translation"):
            raw = value.get(candidate)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def short_gloss_from_payload(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    if isinstance(payload, dict):
        glosses = payload.get("glosses")
        if isinstance(glosses, list):
            for gloss in glosses:
                if isinstance(gloss, str) and gloss.strip():
                    return gloss.strip()

        for candidate in ("gloss", "translation", "headword"):
            raw = payload.get(candidate)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()

        definition = payload.get("definition")
        if isinstance(definition, str) and definition.strip():
            first_chunk = re.split(r"[.;]", definition.strip())[0]
            normalized = re.sub(r"\s+", " ", first_chunk).strip()
            if normalized:
                return " ".join(normalized.split(" ")[:8])
    return None


def value_to_json_text(value: str | None, value_json: str | None) -> str:
    if value_json is not None:
        try:
            parsed = json.loads(value_json)
        except json.JSONDecodeError as exc:
            raise CLIError(f"--value-json is not valid JSON: {exc}") from exc
        return json.dumps(parsed, ensure_ascii=False)
    if value is None:
        raise CLIError("Provide either --value or --value-json.")
    return json.dumps(value, ensure_ascii=False)


def parse_target(args: argparse.Namespace, book_lookup: dict[str, str]) -> dict[str, Any]:
    kind = args.kind
    if kind in {"word", "verse"}:
        if not args.ref:
            raise CLIError(f"--ref is required for kind={kind}.")
        verse_ref = parse_ref(args.ref, book_lookup)
        word_index = -1
        if kind == "word":
            if args.word_index is None:
                raise CLIError("--word-index is required for kind=word.")
            word_index = args.word_index
            if word_index < 0:
                raise CLIError("--word-index must be >= 0.")
        return {
            "target_kind": kind,
            "book": verse_ref.book,
            "chapter_start": verse_ref.chapter,
            "verse_start": verse_ref.verse,
            "chapter_end": verse_ref.chapter,
            "verse_end": verse_ref.verse,
            "word_index": word_index,
        }

    if kind == "passage":
        if not args.start or not args.end:
            raise CLIError("--start and --end are required for kind=passage.")
        start_ref = parse_ref(args.start, book_lookup)
        end_ref = parse_ref(args.end, book_lookup)
        if start_ref.book != end_ref.book:
            raise CLIError("Passage start/end must be in the same book.")
        if start_ref.scalar() > end_ref.scalar():
            raise CLIError("Passage start must be <= passage end.")
        return {
            "target_kind": "passage",
            "book": start_ref.book,
            "chapter_start": start_ref.chapter,
            "verse_start": start_ref.verse,
            "chapter_end": end_ref.chapter,
            "verse_end": end_ref.verse,
            "word_index": -1,
        }

    raise CLIError(f"Unsupported annotation kind: {kind}")


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def print_tsv(rows: list[dict[str, Any]], columns: list[str], include_header: bool) -> None:
    if include_header:
        print("\t".join(columns))
    for row in rows:
        values: list[str] = []
        for column in columns:
            raw = row.get(column)
            if isinstance(raw, (dict, list)):
                values.append(json.dumps(raw, ensure_ascii=False))
            elif raw is None:
                values.append("")
            else:
                values.append(str(raw))
        print("\t".join(values))


def cmd_init_db(args: argparse.Namespace) -> int:
    summary = initialize_db(
        db_path=resolve_path(args.db),
        wlc_path=resolve_path(args.wlc),
        bdb_path=resolve_path(args.bdb),
        strongs_path=resolve_path(args.strongs),
        rebuild=args.rebuild,
    )

    if args.format == "json":
        print_json(summary)
    else:
        for key in ("db", "books", "chapters", "verses", "tokens", "bdb_entries", "strongs_loaded", "strongs_entries"):
            print(f"{key}: {summary[key]}")
    return 0


def cmd_books(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        check_token_table_has_data(conn)
        rows = conn.execute(
            """
            SELECT
              book,
              COUNT(DISTINCT chapter) AS chapter_count,
              COUNT(DISTINCT chapter || ':' || verse) AS verse_count,
              COUNT(*) AS word_count
            FROM tokens
            GROUP BY book
            """
        ).fetchall()

        books = [
            {
                "book": row["book"],
                "chapter_count": row["chapter_count"],
                "verse_count": row["verse_count"],
                "word_count": row["word_count"],
            }
            for row in rows
        ]
        books.sort(key=lambda item: canonical_book_order_key(item["book"]))

        if args.format == "json":
            print_json(books)
            return 0
        if args.format == "tsv":
            print_tsv(books, ["book", "chapter_count", "verse_count", "word_count"], not args.no_header)
            return 0

        for book in books:
            print(f"{book['book']}\tchapters={book['chapter_count']}\tverses={book['verse_count']}\twords={book['word_count']}")
        return 0
    finally:
        conn.close()


def cmd_verse(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        check_token_table_has_data(conn)
        book_lookup = load_book_lookup(conn)
        verse_ref = parse_ref(args.ref, book_lookup)
        tokens = fetch_verse_tokens(conn, verse_ref)
        if not tokens:
            raise CLIError(f"No tokens found for {verse_ref.label()}.")

        source_ids = parse_source_ids(args.sources)
        source_config_map = load_source_config_map(conn, source_ids)

        for token in tokens:
            enriched: dict[str, Any] = {}
            for source_id in source_ids:
                source_config = source_config_map[source_id]
                payload = lookup_source_payload_for_token(conn, source_id, token, source_config)
                if payload is not None:
                    enriched[source_id] = payload
            if enriched:
                token["sources"] = enriched

        verse_annotations: dict[str, Any] = {}
        word_annotations: dict[str, dict[str, Any]] = {}
        for layer in parse_source_ids(args.annotation_layers):
            verse_row = conn.execute(
                """
                SELECT value_json
                FROM annotations
                WHERE layer = ? AND target_kind = 'verse'
                  AND book = ? AND chapter_start = ? AND verse_start = ?
                """,
                (layer, verse_ref.book, verse_ref.chapter, verse_ref.verse),
            ).fetchone()
            if verse_row is not None:
                verse_annotations[layer] = json.loads(verse_row["value_json"])

            word_rows = conn.execute(
                """
                SELECT word_index, value_json
                FROM annotations
                WHERE layer = ? AND target_kind = 'word'
                  AND book = ? AND chapter_start = ? AND verse_start = ?
                ORDER BY word_index
                """,
                (layer, verse_ref.book, verse_ref.chapter, verse_ref.verse),
            ).fetchall()
            for row in word_rows:
                slot = word_annotations.setdefault(str(row["word_index"]), {})
                slot[layer] = json.loads(row["value_json"])

        hebrew = " ".join(token["text"] for token in tokens)

        result = {
            "ref": verse_ref.label(),
            "hebrew": hebrew,
            "tokens": tokens,
        }
        if verse_annotations:
            result["verse_annotations"] = verse_annotations
        if word_annotations:
            result["word_annotations"] = word_annotations

        if args.format == "json":
            print_json(result)
            return 0
        if args.format == "tsv":
            print_tsv(tokens, ["book", "chapter", "verse", "word_index", "text", "lemma", "morph", "bdb", "sources"], not args.no_header)
            return 0

        if args.tokens:
            for token in tokens:
                print(f"{token['word_index']}\t{token['text']}\t{token['lemma']}\t{token['morph']}\t{token['bdb']}")
            return 0
        print(hebrew)
        return 0
    finally:
        conn.close()


def cmd_search(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        check_token_table_has_data(conn)
        where_parts: list[str] = []
        params: list[Any] = []

        if args.text:
            where_parts.append("text = ?")
            params.append(args.text)
        if args.text_like:
            where_parts.append("text LIKE ?")
            params.append(args.text_like)
        if args.lemma:
            where_parts.append("lemma = ?")
            params.append(args.lemma)
        if args.morph:
            where_parts.append("morph = ?")
            params.append(args.morph)
        if args.bdb:
            where_parts.append("bdb = ?")
            params.append(args.bdb)
        if args.book:
            book_lookup = load_book_lookup(conn)
            normalized = normalize_book_key(args.book)
            if normalized not in book_lookup:
                raise CLIError(f"Unknown book: {args.book}")
            where_parts.append("book = ?")
            params.append(book_lookup[normalized])
        if args.ref:
            book_lookup = load_book_lookup(conn)
            verse_ref = parse_ref(args.ref, book_lookup)
            where_parts.extend(["book = ?", "chapter = ?", "verse = ?"])
            params.extend([verse_ref.book, verse_ref.chapter, verse_ref.verse])

        where_sql = ""
        if where_parts:
            where_sql = "WHERE " + " AND ".join(where_parts)

        rows = conn.execute(
            f"""
            SELECT book, chapter, verse, word_index, text, lemma, morph, bdb
            FROM tokens
            {where_sql}
            ORDER BY book, chapter, verse, word_index
            LIMIT ? OFFSET ?
            """,
            tuple(params + [args.limit, args.offset]),
        ).fetchall()

        result = [
            {
                "ref": f"{row['book']} {row['chapter']}:{row['verse']}",
                "book": row["book"],
                "chapter": row["chapter"],
                "verse": row["verse"],
                "word_index": row["word_index"],
                "text": row["text"],
                "lemma": row["lemma"],
                "morph": row["morph"],
                "bdb": row["bdb"],
            }
            for row in rows
        ]

        if args.format == "json":
            print_json(result)
            return 0
        if args.format == "tsv":
            print_tsv(result, ["ref", "word_index", "text", "lemma", "morph", "bdb"], not args.no_header)
            return 0

        for item in result:
            print(f"{item['ref']}[{item['word_index']}]\t{item['text']}\t{item['lemma']}\t{item['morph']}\t{item['bdb']}")
        return 0
    finally:
        conn.close()


def cmd_source_list(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT
              s.source_id,
              s.source_type,
              s.path,
              s.root_path,
              s.key_field,
              s.lookup_token_field,
              s.enabled,
              COUNT(se.entry_key) AS entry_count
            FROM sources s
            LEFT JOIN source_entries se ON se.source_id = s.source_id
            GROUP BY s.source_id, s.source_type, s.path, s.root_path, s.key_field, s.lookup_token_field, s.enabled
            ORDER BY s.source_id
            """
        ).fetchall()
        result = [dict(row) for row in rows]

        if args.format == "json":
            print_json(result)
            return 0
        if args.format == "tsv":
            print_tsv(
                result,
                ["source_id", "source_type", "lookup_token_field", "entry_count", "path", "root_path", "key_field", "enabled"],
                not args.no_header,
            )
            return 0

        for row in result:
            print(
                f"{row['source_id']}\ttype={row['source_type']}\tlookup={row['lookup_token_field']}\t"
                f"entries={row['entry_count']}\tpath={row['path']}"
            )
        return 0
    finally:
        conn.close()


def cmd_source_register(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        with conn:
            register_source(
                conn=conn,
                source_id=args.id,
                source_type=args.type,
                path=args.path,
                root_path=args.root,
                key_field=args.key_field,
                lookup_token_field=args.lookup_token_field,
                config_json=args.config_json,
            )
            ingested = 0
            if args.ingest:
                ingested = ingest_source(conn, args.id)

        result = {
            "source_id": args.id,
            "registered": True,
            "ingested_entries": ingested,
        }
        if args.format == "json":
            print_json(result)
        else:
            print(f"Registered source '{args.id}'.")
            if args.ingest:
                print(f"Ingested entries: {ingested}")
        return 0
    finally:
        conn.close()


def cmd_source_ingest(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        with conn:
            entry_count = ingest_source(conn, args.id)
        result = {"source_id": args.id, "entry_count": entry_count}
        if args.format == "json":
            print_json(result)
        else:
            print(f"Ingested {entry_count} entries for source '{args.id}'.")
        return 0
    finally:
        conn.close()


def cmd_source_lookup(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        payload = lookup_source_payload(conn, args.id, args.key)
        if payload is None:
            raise CLIError(f"No entry for key '{args.key}' in source '{args.id}'.")

        if args.format == "json":
            print_json({"source_id": args.id, "key": args.key, "entry": payload})
            return 0
        if args.format == "tsv":
            print_tsv(
                [{"source_id": args.id, "key": args.key, "entry": payload}],
                ["source_id", "key", "entry"],
                not args.no_header,
            )
            return 0

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


def cmd_annotate_set(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        check_token_table_has_data(conn)
        book_lookup = load_book_lookup(conn)
        target = parse_target(args, book_lookup)
        value_json = value_to_json_text(args.value, args.value_json)

        with conn:
            conn.execute(
                """
                INSERT INTO annotations(
                  layer, target_kind, book, chapter_start, verse_start, chapter_end, verse_end, word_index,
                  value_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(layer, target_kind, book, chapter_start, verse_start, chapter_end, verse_end, word_index)
                DO UPDATE SET value_json = excluded.value_json, updated_at = CURRENT_TIMESTAMP
                """,
                (
                    args.layer,
                    target["target_kind"],
                    target["book"],
                    target["chapter_start"],
                    target["verse_start"],
                    target["chapter_end"],
                    target["verse_end"],
                    target["word_index"],
                    value_json,
                ),
            )

        result = {"layer": args.layer, **target, "value": json.loads(value_json)}
        if args.format == "json":
            print_json(result)
        else:
            print(f"Saved annotation in layer '{args.layer}' for {target['target_kind']}.")
        return 0
    finally:
        conn.close()


def cmd_annotate_get(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        check_token_table_has_data(conn)
        book_lookup = load_book_lookup(conn)
        target = parse_target(args, book_lookup)

        row = conn.execute(
            """
            SELECT id, layer, target_kind, book, chapter_start, verse_start, chapter_end, verse_end, word_index, value_json,
                   created_at, updated_at
            FROM annotations
            WHERE layer = ? AND target_kind = ? AND book = ? AND chapter_start = ? AND verse_start = ?
              AND chapter_end = ? AND verse_end = ? AND word_index = ?
            """,
            (
                args.layer,
                target["target_kind"],
                target["book"],
                target["chapter_start"],
                target["verse_start"],
                target["chapter_end"],
                target["verse_end"],
                target["word_index"],
            ),
        ).fetchone()
        if row is None:
            raise CLIError("Annotation not found.")

        record = dict(row)
        record["value"] = json.loads(record.pop("value_json"))

        if args.format == "json":
            print_json(record)
            return 0
        if args.format == "tsv":
            print_tsv([record], list(record.keys()), not args.no_header)
            return 0

        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


def cmd_annotate_list(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        check_token_table_has_data(conn)

        where: list[str] = []
        params: list[Any] = []

        if args.layer:
            where.append("layer = ?")
            params.append(args.layer)
        if args.kind:
            where.append("target_kind = ?")
            params.append(args.kind)
        if args.book:
            book_lookup = load_book_lookup(conn)
            normalized = normalize_book_key(args.book)
            if normalized not in book_lookup:
                raise CLIError(f"Unknown book: {args.book}")
            where.append("book = ?")
            params.append(book_lookup[normalized])

        if args.ref:
            book_lookup = load_book_lookup(conn)
            verse_ref = parse_ref(args.ref, book_lookup)
            scalar = verse_ref.scalar()
            where.append("book = ?")
            params.append(verse_ref.book)
            where.append("(chapter_start * 1000 + verse_start) <= ?")
            params.append(scalar)
            where.append("(chapter_end * 1000 + verse_end) >= ?")
            params.append(scalar)

        where_sql = ""
        if where:
            where_sql = "WHERE " + " AND ".join(where)

        rows = conn.execute(
            f"""
            SELECT id, layer, target_kind, book, chapter_start, verse_start, chapter_end, verse_end,
                   word_index, value_json, created_at, updated_at
            FROM annotations
            {where_sql}
            ORDER BY book, chapter_start, verse_start, word_index
            LIMIT ?
            """,
            tuple(params + [args.limit]),
        ).fetchall()

        records = []
        for row in rows:
            record = dict(row)
            record["value"] = json.loads(record.pop("value_json"))
            records.append(record)

        if args.format == "json":
            print_json(records)
            return 0
        if args.format == "tsv":
            columns = [
                "id",
                "layer",
                "target_kind",
                "book",
                "chapter_start",
                "verse_start",
                "chapter_end",
                "verse_end",
                "word_index",
                "value",
            ]
            print_tsv(records, columns, not args.no_header)
            return 0

        for record in records:
            print(
                f"{record['id']}\t{record['layer']}\t{record['target_kind']}\t"
                f"{record['book']} {record['chapter_start']}:{record['verse_start']}"
                f"-{record['chapter_end']}:{record['verse_end']}\tword={record['word_index']}\t{record['value']}"
            )
        return 0
    finally:
        conn.close()


def cmd_translation_set(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        check_token_table_has_data(conn)
        book_lookup = load_book_lookup(conn)
        verse_ref = parse_ref(args.ref, book_lookup)
        meta_json = args.meta_json or "{}"
        try:
            parsed_meta = json.loads(meta_json)
        except json.JSONDecodeError as exc:
            raise CLIError(f"--meta-json is not valid JSON: {exc}") from exc

        with conn:
            conn.execute(
                """
                INSERT INTO translations(version, book, chapter, verse, text, meta_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(version, book, chapter, verse)
                DO UPDATE SET text = excluded.text, meta_json = excluded.meta_json, updated_at = CURRENT_TIMESTAMP
                """,
                (args.version, verse_ref.book, verse_ref.chapter, verse_ref.verse, args.text, json.dumps(parsed_meta, ensure_ascii=False)),
            )

        result = {
            "version": args.version,
            "ref": verse_ref.label(),
            "text": args.text,
            "meta": parsed_meta,
        }
        if args.format == "json":
            print_json(result)
        else:
            print(f"Saved translation '{args.version}' for {verse_ref.label()}.")
        return 0
    finally:
        conn.close()


def cmd_translation_get(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        check_token_table_has_data(conn)
        book_lookup = load_book_lookup(conn)
        verse_ref = parse_ref(args.ref, book_lookup)
        row = conn.execute(
            """
            SELECT version, book, chapter, verse, text, meta_json, created_at, updated_at
            FROM translations
            WHERE version = ? AND book = ? AND chapter = ? AND verse = ?
            """,
            (args.version, verse_ref.book, verse_ref.chapter, verse_ref.verse),
        ).fetchone()
        if row is None:
            raise CLIError(f"No translation found for {args.version} {verse_ref.label()}.")

        record = dict(row)
        record["ref"] = f"{record['book']} {record['chapter']}:{record['verse']}"
        record["meta"] = json.loads(record.pop("meta_json"))

        if args.format == "json":
            print_json(record)
            return 0
        if args.format == "tsv":
            print_tsv([record], ["version", "ref", "text", "meta"], not args.no_header)
            return 0

        print(record["text"])
        return 0
    finally:
        conn.close()


def cmd_translation_list(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        where: list[str] = []
        params: list[Any] = []

        if args.version:
            where.append("version = ?")
            params.append(args.version)
        if args.ref:
            book_lookup = load_book_lookup(conn)
            verse_ref = parse_ref(args.ref, book_lookup)
            where.extend(["book = ?", "chapter = ?", "verse = ?"])
            params.extend([verse_ref.book, verse_ref.chapter, verse_ref.verse])

        where_sql = ""
        if where:
            where_sql = "WHERE " + " AND ".join(where)

        rows = conn.execute(
            f"""
            SELECT version, book, chapter, verse, text, meta_json, updated_at
            FROM translations
            {where_sql}
            ORDER BY version, book, chapter, verse
            LIMIT ?
            """,
            tuple(params + [args.limit]),
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["ref"] = f"{record['book']} {record['chapter']}:{record['verse']}"
            record["meta"] = json.loads(record.pop("meta_json"))
            records.append(record)

        if args.format == "json":
            print_json(records)
            return 0
        if args.format == "tsv":
            print_tsv(records, ["version", "ref", "text", "meta", "updated_at"], not args.no_header)
            return 0

        for record in records:
            print(f"{record['version']}\t{record['ref']}\t{record['text']}")
        return 0
    finally:
        conn.close()


def cmd_translation_draft(args: argparse.Namespace) -> int:
    conn = connect_db(resolve_path(args.db))
    try:
        ensure_schema(conn)
        check_token_table_has_data(conn)
        book_lookup = load_book_lookup(conn)
        verse_ref = parse_ref(args.ref, book_lookup)
        tokens = fetch_verse_tokens(conn, verse_ref)
        if not tokens:
            raise CLIError(f"No tokens found for {verse_ref.label()}.")

        annotation_rows = conn.execute(
            """
            SELECT word_index, value_json
            FROM annotations
            WHERE layer = ? AND target_kind = 'word'
              AND book = ? AND chapter_start = ? AND verse_start = ?
            """,
            (args.annotation_layer, verse_ref.book, verse_ref.chapter, verse_ref.verse),
        ).fetchall()
        word_annotations: dict[int, Any] = {
            row["word_index"]: json.loads(row["value_json"]) for row in annotation_rows
        }

        source_ids = parse_source_ids(args.sources)
        source_configs = load_source_config_map(conn, source_ids)

        chosen_tokens: list[dict[str, Any]] = []
        for token in tokens:
            chosen_gloss: str | None = None
            chosen_source = "none"

            annotation_value = word_annotations.get(token["word_index"])
            if annotation_value is not None:
                chosen_gloss = extract_annotation_text(annotation_value)
                if chosen_gloss:
                    chosen_source = f"annotation:{args.annotation_layer}"

            if not chosen_gloss:
                for source_id in source_ids:
                    config = source_configs[source_id]
                    payload = lookup_source_payload_for_token(conn, source_id, token, config)
                    if payload is None:
                        continue
                    gloss = short_gloss_from_payload(payload)
                    if gloss:
                        chosen_gloss = gloss
                        chosen_source = f"source:{source_id}"
                        break

            if not chosen_gloss:
                chosen_gloss = token["text"]

            chosen_tokens.append(
                {
                    "word_index": token["word_index"],
                    "text": token["text"],
                    "lemma": token["lemma"],
                    "morph": token["morph"],
                    "bdb": token["bdb"],
                    "gloss": chosen_gloss,
                    "gloss_source": chosen_source,
                }
            )

        draft = " ".join(item["gloss"] for item in chosen_tokens)

        if args.persist_version:
            with conn:
                meta = {
                    "method": "draft",
                    "annotation_layer": args.annotation_layer,
                    "sources": source_ids,
                }
                conn.execute(
                    """
                    INSERT INTO translations(version, book, chapter, verse, text, meta_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(version, book, chapter, verse)
                    DO UPDATE SET text = excluded.text, meta_json = excluded.meta_json, updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        args.persist_version,
                        verse_ref.book,
                        verse_ref.chapter,
                        verse_ref.verse,
                        draft,
                        json.dumps(meta, ensure_ascii=False),
                    ),
                )

        result = {
            "ref": verse_ref.label(),
            "draft_translation": draft,
            "annotation_layer": args.annotation_layer,
            "sources": source_ids,
            "tokens": chosen_tokens,
        }

        if args.format == "json":
            print_json(result)
            return 0
        if args.format == "tsv":
            print_tsv(chosen_tokens, ["word_index", "text", "gloss", "gloss_source", "lemma", "morph", "bdb"], not args.no_header)
            return 0

        print(draft)
        return 0
    finally:
        conn.close()


def add_output_arguments(parser: argparse.ArgumentParser, default: str = "text") -> None:
    parser.add_argument("--format", choices=["text", "json", "tsv"], default=default, help="Output format.")
    parser.add_argument("--no-header", action="store_true", help="Suppress TSV header row.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tanach CLI for analysis, annotation layers, and translation workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Ingest WLC/BDB/Strongs JSON into the local SQLite db.")
    init_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    init_parser.add_argument("--wlc", default=str(DEFAULT_WLC_PATH), help="Path to WLC JSON.")
    init_parser.add_argument("--bdb", default=str(DEFAULT_BDB_PATH), help="Path to BDB JSON.")
    init_parser.add_argument("--strongs", default=str(DEFAULT_STRONGS_PATH), help="Path to Strongs JSON.")
    init_parser.add_argument("--rebuild", action="store_true", help="Rebuild token/source tables before ingest.")
    add_output_arguments(init_parser)
    init_parser.set_defaults(func=cmd_init_db)

    books_parser = subparsers.add_parser("books", help="List books and token counts.")
    books_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    add_output_arguments(books_parser)
    books_parser.set_defaults(func=cmd_books)

    verse_parser = subparsers.add_parser("verse", help="Print or inspect a verse.")
    verse_parser.add_argument("ref", help="Reference, e.g. 'Gen 1:1'.")
    verse_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    verse_parser.add_argument("--tokens", action="store_true", help="For text output, print one token per line.")
    verse_parser.add_argument("--sources", default="", help="Comma-separated source ids to enrich token rows.")
    verse_parser.add_argument(
        "--annotation-layers",
        default="",
        help="Comma-separated annotation layers to include in JSON output.",
    )
    add_output_arguments(verse_parser)
    verse_parser.set_defaults(func=cmd_verse)

    search_parser = subparsers.add_parser("search", help="Search tokens by text/lemma/morph/bdb/ref.")
    search_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    search_parser.add_argument("--text", help="Exact surface form.")
    search_parser.add_argument("--text-like", help="SQL LIKE pattern for surface form.")
    search_parser.add_argument("--lemma", help="Exact lemma.")
    search_parser.add_argument("--morph", help="Exact morph code.")
    search_parser.add_argument("--bdb", help="Exact bdb id.")
    search_parser.add_argument("--book", help="Filter by book.")
    search_parser.add_argument("--ref", help="Filter to one verse.")
    search_parser.add_argument("--limit", type=int, default=100, help="Max rows.")
    search_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination.")
    add_output_arguments(search_parser)
    search_parser.set_defaults(func=cmd_search)

    source_parser = subparsers.add_parser("source", help="Manage extensible lexicon/reference sources.")
    source_subparsers = source_parser.add_subparsers(dest="source_command", required=True)

    source_list_parser = source_subparsers.add_parser("list", help="List registered sources.")
    source_list_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    add_output_arguments(source_list_parser)
    source_list_parser.set_defaults(func=cmd_source_list)

    source_register_parser = source_subparsers.add_parser("register", help="Register or update a source.")
    source_register_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    source_register_parser.add_argument("--id", required=True, help="Source id.")
    source_register_parser.add_argument("--type", default="keyed-json", help="Source type.")
    source_register_parser.add_argument("--path", required=True, help="JSON path to ingest.")
    source_register_parser.add_argument("--root", default="entries", help="Root object/list within JSON.")
    source_register_parser.add_argument("--key-field", default="", help="Entry key field (for list roots).")
    source_register_parser.add_argument(
        "--lookup-token-field",
        choices=["text", "lemma", "morph", "bdb"],
        default="lemma",
        help="Token field used for lookup key.",
    )
    source_register_parser.add_argument("--config-json", default="{}", help="Free-form JSON config.")
    source_register_parser.add_argument("--ingest", action="store_true", help="Ingest source entries after registration.")
    add_output_arguments(source_register_parser)
    source_register_parser.set_defaults(func=cmd_source_register)

    source_ingest_parser = source_subparsers.add_parser("ingest", help="Ingest entries for a registered source.")
    source_ingest_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    source_ingest_parser.add_argument("--id", required=True, help="Source id.")
    add_output_arguments(source_ingest_parser)
    source_ingest_parser.set_defaults(func=cmd_source_ingest)

    source_lookup_parser = source_subparsers.add_parser("lookup", help="Lookup one source entry by key.")
    source_lookup_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    source_lookup_parser.add_argument("--id", required=True, help="Source id.")
    source_lookup_parser.add_argument("--key", required=True, help="Entry key to lookup.")
    add_output_arguments(source_lookup_parser)
    source_lookup_parser.set_defaults(func=cmd_source_lookup)

    annotate_parser = subparsers.add_parser("annotate", help="Manage layered annotations.")
    annotate_subparsers = annotate_parser.add_subparsers(dest="annotation_command", required=True)

    annotate_set_parser = annotate_subparsers.add_parser("set", help="Create or update an annotation.")
    annotate_set_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    annotate_set_parser.add_argument("--layer", required=True, help="Annotation layer name.")
    annotate_set_parser.add_argument("--kind", choices=["word", "verse", "passage"], required=True, help="Annotation scope.")
    annotate_set_parser.add_argument("--ref", help="Single verse ref for word/verse kinds.")
    annotate_set_parser.add_argument("--word-index", type=int, help="Word index for kind=word.")
    annotate_set_parser.add_argument("--start", help="Passage start ref for kind=passage.")
    annotate_set_parser.add_argument("--end", help="Passage end ref for kind=passage.")
    annotate_set_parser.add_argument("--value", help="Simple string value.")
    annotate_set_parser.add_argument("--value-json", help="Structured JSON value.")
    add_output_arguments(annotate_set_parser)
    annotate_set_parser.set_defaults(func=cmd_annotate_set)

    annotate_get_parser = annotate_subparsers.add_parser("get", help="Fetch one annotation.")
    annotate_get_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    annotate_get_parser.add_argument("--layer", required=True, help="Annotation layer name.")
    annotate_get_parser.add_argument("--kind", choices=["word", "verse", "passage"], required=True, help="Annotation scope.")
    annotate_get_parser.add_argument("--ref", help="Single verse ref for word/verse kinds.")
    annotate_get_parser.add_argument("--word-index", type=int, help="Word index for kind=word.")
    annotate_get_parser.add_argument("--start", help="Passage start ref for kind=passage.")
    annotate_get_parser.add_argument("--end", help="Passage end ref for kind=passage.")
    add_output_arguments(annotate_get_parser)
    annotate_get_parser.set_defaults(func=cmd_annotate_get)

    annotate_list_parser = annotate_subparsers.add_parser("list", help="List annotations with filters.")
    annotate_list_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    annotate_list_parser.add_argument("--layer", help="Layer name.")
    annotate_list_parser.add_argument("--kind", choices=["word", "verse", "passage"], help="Target scope.")
    annotate_list_parser.add_argument("--book", help="Book filter.")
    annotate_list_parser.add_argument("--ref", help="List annotations overlapping this verse ref.")
    annotate_list_parser.add_argument("--limit", type=int, default=100, help="Max rows.")
    add_output_arguments(annotate_list_parser)
    annotate_list_parser.set_defaults(func=cmd_annotate_list)

    translation_parser = subparsers.add_parser("translation", help="Manage translation layers and drafts.")
    translation_subparsers = translation_parser.add_subparsers(dest="translation_command", required=True)

    translation_set_parser = translation_subparsers.add_parser("set", help="Set a verse translation text.")
    translation_set_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    translation_set_parser.add_argument("--version", required=True, help="Translation version/layer id.")
    translation_set_parser.add_argument("--ref", required=True, help="Verse ref.")
    translation_set_parser.add_argument("--text", required=True, help="Translation text.")
    translation_set_parser.add_argument("--meta-json", help="Optional metadata JSON.")
    add_output_arguments(translation_set_parser)
    translation_set_parser.set_defaults(func=cmd_translation_set)

    translation_get_parser = translation_subparsers.add_parser("get", help="Get one verse translation text.")
    translation_get_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    translation_get_parser.add_argument("--version", required=True, help="Translation version/layer id.")
    translation_get_parser.add_argument("--ref", required=True, help="Verse ref.")
    add_output_arguments(translation_get_parser)
    translation_get_parser.set_defaults(func=cmd_translation_get)

    translation_list_parser = translation_subparsers.add_parser("list", help="List saved translations.")
    translation_list_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    translation_list_parser.add_argument("--version", help="Filter by translation version.")
    translation_list_parser.add_argument("--ref", help="Filter by verse ref.")
    translation_list_parser.add_argument("--limit", type=int, default=100, help="Max rows.")
    add_output_arguments(translation_list_parser)
    translation_list_parser.set_defaults(func=cmd_translation_list)

    translation_draft_parser = translation_subparsers.add_parser(
        "draft",
        help="Generate a draft translation from annotation layer + lexicon sources.",
    )
    translation_draft_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    translation_draft_parser.add_argument("--ref", required=True, help="Verse ref.")
    translation_draft_parser.add_argument(
        "--annotation-layer",
        default="gloss",
        help="Word annotation layer to prefer for glosses.",
    )
    translation_draft_parser.add_argument(
        "--sources",
        default="bdb,strongs",
        help="Comma-separated fallback source ids used when annotation gloss is absent.",
    )
    translation_draft_parser.add_argument(
        "--persist-version",
        help="If set, save the draft in translations under this version id.",
    )
    add_output_arguments(translation_draft_parser)
    translation_draft_parser.set_defaults(func=cmd_translation_draft)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
