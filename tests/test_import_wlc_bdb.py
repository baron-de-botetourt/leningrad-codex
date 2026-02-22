#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import import_wlc_bdb  # noqa: E402


class ParseBdbTests(unittest.TestCase):
    def parse_entries(self, xml_text: str):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "BrownDriverBriggs.xml"
            path.write_text(xml_text, encoding="utf-8")
            return import_wlc_bdb.parse_bdb(path)["entries"]

    def test_parse_bdb_strips_status_markers_and_extracts_headword(self):
        entries = self.parse_entries(
            """<?xml version="1.0" encoding="UTF-8"?>
<lexicon>
  <entry id="a.aa.aa">
    <w>א</w>
    Āleph, first letter.
    <status p="4">done</status>
  </entry>
</lexicon>
"""
        )
        self.assertIn("a.aa.aa", entries)
        self.assertEqual(entries["a.aa.aa"]["headword"], "א")
        self.assertEqual(entries["a.aa.aa"]["definition"], "א Āleph, first letter.")

    def test_parse_bdb_falls_back_to_hebrew_run_for_headword(self):
        entries = self.parse_entries(
            """<?xml version="1.0" encoding="UTF-8"?>
<lexicon>
  <entry id="a.ab.ac">
    אָבִיב n.m. fresh, young ears of barley done
  </entry>
</lexicon>
"""
        )
        self.assertIn("a.ab.ac", entries)
        self.assertEqual(entries["a.ab.ac"]["headword"], "אָבִיב")
        self.assertTrue(entries["a.ab.ac"]["definition"].endswith("barley"))

    def test_parse_bdb_normalizes_unicode_and_removes_format_chars(self):
        entries = self.parse_entries(
            """<?xml version="1.0" encoding="UTF-8"?>
<lexicon>
  <entry id="e.ab.aa">
    <w>א</w>
    test\u200d\u200f value
    <status>ref</status>
  </entry>
</lexicon>
"""
        )
        definition = entries["e.ab.aa"]["definition"]
        self.assertEqual(definition, "א test value")
        self.assertEqual(definition, import_wlc_bdb.normalize_unicode_text(definition))


if __name__ == "__main__":
    unittest.main()
