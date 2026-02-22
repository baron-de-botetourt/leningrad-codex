#!/usr/bin/env python3
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bdb_sefaria_source  # noqa: E402


class BdbSefariaSourceTests(unittest.TestCase):
    def sample_raw(self):
        return {
            "index": {
                "start_ref": "BDB, א",
                "last_word": "אָב",
                "letter_refs": ["BDB, א", "BDB, אָב"],
            },
            "entries": [
                {
                    "ref": "BDB, א",
                    "prev": None,
                    "next": "BDB, אָב",
                    "text_html": "<big><span dir='rtl'>א</span></big> first letter",
                },
                {
                    "ref": "BDB, אָב",
                    "prev": "BDB, א",
                    "next": None,
                    "text_html": "<span dir='rtl'>אָב</span> father",
                },
            ],
        }

    def test_verify_raw_completeness_accepts_valid_chain(self):
        errors = bdb_sefaria_source.verify_raw_completeness(self.sample_raw())
        self.assertEqual(errors, [])

    def test_verify_raw_completeness_rejects_broken_links(self):
        raw = self.sample_raw()
        raw["entries"][0]["next"] = "BDB, X"
        errors = bdb_sefaria_source.verify_raw_completeness(raw)
        self.assertTrue(any("Broken next chain" in e for e in errors))

    def test_build_entries_from_raw(self):
        entries = bdb_sefaria_source.build_bdb_entries_from_raw(self.sample_raw())
        self.assertIn("BDB, א", entries)
        self.assertEqual(entries["BDB, אָב"]["headword"], "אָב")
        self.assertEqual(entries["BDB, אָב"]["definition"], "אָב father")

    def test_headword_variants_include_expected_niqqud_swaps(self):
        variants = bdb_sefaria_source._headword_variants("עַפְרָה")
        self.assertIn("עָפְרָה", variants)
        self.assertIn("עפרה", variants)


if __name__ == "__main__":
    unittest.main()
