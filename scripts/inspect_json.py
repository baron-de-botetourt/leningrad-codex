#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_path(path_expr: str):
    if not path_expr:
        return []
    return [p for p in path_expr.strip().split(".") if p]


def get_at_path(obj, parts):
    current = obj
    traversed = []
    for part in parts:
        traversed.append(part)
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(".".join(traversed))
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError as e:
                raise KeyError(".".join(traversed)) from e
            if idx < 0 or idx >= len(current):
                raise KeyError(".".join(traversed))
            current = current[idx]
        else:
            raise KeyError(".".join(traversed))
    return current


def preview(value, max_items=10):
    if isinstance(value, dict):
        keys = list(value.keys())
        return {
            "type": "object",
            "size": len(value),
            "keys_preview": keys[:max_items],
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "size": len(value),
            "items_preview": value[:max_items],
        }
    return {"type": type(value).__name__, "value": value}


def main():
    parser = argparse.ArgumentParser(description="Inspect a JSON file by dot-path.")
    parser.add_argument("file", help="Path to JSON file")
    parser.add_argument(
        "--path",
        default="",
        help="Dot path (e.g. books.Genesis.chapters.1.verses.1.words.0)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full value at path (default prints compact preview)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = parse_path(args.path)

    try:
        value = get_at_path(data, parts)
    except KeyError as err:
        print(f"Path not found: {err}")
        raise SystemExit(1)

    if args.full:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(preview(value), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
