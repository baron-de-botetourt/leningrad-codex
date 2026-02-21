# Hebrew Bible Analysis Toolkit

This repo now includes a headless, non-interactive, pipeable CLI for Tanach analysis and translation workflows:

- `scripts/tanach_cli.py`

The existing static web app is still present, but the CLI is designed for automation and AI tool calls.

## CLI goals

- Clean command surface with explicit inputs/outputs.
- Extensible data model for additional lexica and alignment datasets.
- Performant querying via local SQLite indexes.
- Annotation layers at word/verse/passage scope.
- Translation layers plus draft generation from gloss layers + lexicon sources.

## Quick start

1) Build the SQLite index from JSON:

```bash
cd /Users/tindelllockett/projects/peter-test
./scripts/tanach_cli.py init-db --wlc data/wlc_full.json --bdb data/bdb_full.json --format json
```

2) Read a verse:

```bash
./scripts/tanach_cli.py verse "Gen 1:1"
./scripts/tanach_cli.py verse "Gen 1:1" --format json
./scripts/tanach_cli.py verse "Gen 1:1" --sources strongs --format json
```

3) Search tokens:

```bash
./scripts/tanach_cli.py search --lemma "430" --limit 20 --format tsv
./scripts/tanach_cli.py search --morph "HVqp3ms" --limit 20
```

4) Add and read annotations:

```bash
./scripts/tanach_cli.py annotate set --layer gloss --kind word --ref "Gen 1:1" --word-index 0 --value "in the beginning"
./scripts/tanach_cli.py annotate get --layer gloss --kind word --ref "Gen 1:1" --word-index 0 --format json
./scripts/tanach_cli.py annotate list --ref "Gen 1:1" --format json
```

5) Generate and persist a draft translation:

```bash
./scripts/tanach_cli.py translation draft --ref "Gen 1:1" --annotation-layer gloss --sources bdb,strongs --format json
./scripts/tanach_cli.py translation draft --ref "Gen 1:1" --persist-version draft.v1
./scripts/tanach_cli.py translation get --version draft.v1 --ref "Gen 1:1"
```

## Data model (SQLite)

`scripts/tanach_cli.py init-db` creates:

- `tokens`: normalized WLC words (`book`, `chapter`, `verse`, `word_index`, `text`, `lemma`, `morph`, `bdb`).
- `sources`: registry for lexica/reference sources (type, path, root, lookup strategy).
- `source_entries`: key/value payload rows for each registered source.
- `annotations`: layered annotations at `word`, `verse`, or `passage` scope.
- `translations`: verse-level translation layers (`version` namespace).

This gives you a stable base for:

- lexical lookup (`bdb` + `strongs`),
- translation mapping by verse,
- apparatus/masoretic notes as additional source ids or annotation layers.

## Extending with new sources

Register and ingest a keyed JSON source:

```bash
./scripts/tanach_cli.py source register \
  --id strongs \
  --path data/strongs_full.json \
  --root entries \
  --lookup-token-field lemma \
  --ingest
```

Inspect sources and look up entries:

```bash
./scripts/tanach_cli.py source list --format tsv
./scripts/tanach_cli.py source lookup --id bdb --key t.ad.ag --format json
./scripts/tanach_cli.py source lookup --id strongs --key 7225 --format json
```

Notes:

- `lookup-token-field` controls how words link to source keys (`text`, `lemma`, `morph`, or `bdb`).
- If an external dataset needs explicit per-token links, you can add that as a dedicated annotation layer first, then use it in translation/query flows.

## Output modes

Most commands support:

- `--format text` (human-readable default)
- `--format json` (machine-friendly objects)
- `--format tsv` (pipe-friendly rows)

For TSV, use `--no-header` to suppress headers.

## Import full WLC + BDB automatically

This repo includes a converter:

```bash
cd /Users/tindelllockett/projects/peter-test
./scripts/import_wlc_bdb.py
```

It downloads and converts:

- Westminster Leningrad / MorphHB source: `https://github.com/openscriptures/morphhb`
- Hebrew Lexicon (includes Brown-Driver-Briggs XML): `https://github.com/openscriptures/HebrewLexicon`
  - Strongs Hebrew dataset: `StrongHebrew.xml` in the same repo (open XML source)

## Notes on source texts

- Ensure you have rights/licenses for any BHS/WLC and BDB datasets you import.
- This repo ships sample data and local conversion scripts; validate licensing for downstream redistribution.
