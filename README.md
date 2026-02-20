# Hebrew Bible Annotation App

A static web app for annotating Hebrew words with English glosses while viewing BHS/WLC text and BDB dictionary hints.

## Features

- Displays Hebrew words by `book/chapter/verse`.
- Click any word to see lemma, morphology, and linked BDB entry.
- Add your own English annotation per word.
- Saves annotations in browser `localStorage`.

## Run

Any static server works. Example:

```bash
cd /Users/tindelllockett/projects/peter-test
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

## Data files

The app loads:

- `/data/wlc_full.json` (preferred, generated)
- `/data/bdb_full.json` (preferred, generated)
- Falls back to `/data/wlc_sample.json` and `/data/bdb_sample.json` if full files are absent.

Replace these with full datasets using the same schema.

## Import full WLC + BDB automatically

This repo includes a converter:

```bash
cd /Users/tindelllockett/projects/peter-test
./scripts/import_wlc_bdb.py
```

It downloads and converts:

- Westminster Leningrad / MorphHB source: `https://github.com/openscriptures/morphhb`
- Hebrew Lexicon (includes Brown-Driver-Briggs XML): `https://github.com/openscriptures/HebrewLexicon`

### WLC schema

```json
{
  "books": {
    "Genesis": {
      "chapters": {
        "1": {
          "verses": {
            "1": {
              "words": [
                {
                  "text": "בְּרֵאשִׁית",
                  "lemma": "רֵאשִׁית",
                  "morph": "N-fs",
                  "bdb": "BDB-0912"
                }
              ]
            }
          }
        }
      }
    }
  }
}
```

### BDB schema

```json
{
  "entries": {
    "BDB-0912": {
      "headword": "רֵאשִׁית",
      "glosses": ["beginning", "first"],
      "definition": "..."
    }
  }
}
```

## Notes on source texts

- Ensure you have rights/licenses for any BHS/WLC and BDB datasets you import.
- This repo ships only tiny sample data for demonstration.
