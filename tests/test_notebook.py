import json
import unittest
from pathlib import Path

from harness.notebook import NotebookError, format_citation, load_path, parse_citation, parse_text

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "cash_application_engine.py"


class DatabricksSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = load_path(SAMPLE)

    def test_cell_count_and_format(self):
        self.assertEqual(self.nb.fmt, "databricks-python")
        self.assertEqual(len(self.nb.cells), 47)
        self.assertEqual(self.nb.n_file_lines, 551)

    def test_addresses_match_databricks_ui(self):
        # line numbers restart per cell and skip the export's blank padding
        self.assertEqual(self.nb.line_text(4, 1), "TOL = 0.5")
        self.assertEqual(self.nb.line_text(29, 4), "TOL = 5.0")
        self.assertEqual(self.nb.cell(29).lines[3].file_line, 361)
        self.assertEqual(self.nb.line_text(26, 35), "    ds = sorted(ds, key=lambda d: -bal[d])")
        self.assertEqual(len(self.nb.cell(26).lines), 56)

    def test_magic_cells(self):
        self.assertEqual(self.nb.cell(1).lang, "md")
        self.assertEqual(self.nb.line_text(1, 1), "%md")            # MAGIC prefix stripped
        self.assertEqual(self.nb.cell(5).lang, "run")
        self.assertEqual(self.nb.cell(5).magic, "%run ../shared/fin_utils")
        self.assertEqual(self.nb.cell(45).lang, "sql")
        self.assertTrue(self.nb.line_text(45, 2).startswith("select typ"))

    def test_listing_has_addresses(self):
        listing = self.nb.listing()
        self.assertIn("[C29:L4] TOL = 5.0", listing)
        self.assertIn("=== Cell 45 · sql ===", listing)

    def test_crlf_and_bom_do_not_change_addresses(self):
        text = SAMPLE.read_text()
        nb = parse_text("﻿" + text.replace("\n", "\r\n"), "x.py")
        self.assertEqual(nb.line_text(29, 4), "TOL = 5.0")
        self.assertEqual(len(nb.cells), 47)


class OtherFormatTests(unittest.TestCase):
    def test_sql_notebook(self):
        nb = load_path(ROOT / "samples" / "sql_example.sql")
        self.assertEqual(nb.fmt, "databricks-sql")
        self.assertEqual([c.lang for c in nb.cells], ["md", "sql", "python", "sql"])
        self.assertEqual(nb.cell(2).title, "Build daily revenue")
        self.assertTrue(nb.line_text(2, 1).startswith("CREATE OR REPLACE TABLE"))
        self.assertEqual(nb.line_text(3, 2), "threshold = 1000")

    def test_ipynb_ignores_outputs(self):
        nb = load_path(ROOT / "samples" / "tiny_example.ipynb")
        self.assertEqual(nb.fmt, "ipynb")
        self.assertEqual([c.lang for c in nb.cells], ["md", "python", "sql", "python"])
        self.assertEqual(nb.cell(2).title, "Load events")
        self.assertNotIn("jane@example.com", nb.listing())
        self.assertIsNone(nb.cell(2).lines[0].file_line)

    def test_percent_script(self):
        nb = parse_text("# %%\nx = 1\n\n# %% [markdown]\n# Title\n# %%\ny = x + 1\n", "s.py")
        self.assertEqual(nb.fmt, "percent")
        self.assertEqual([c.lang for c in nb.cells], ["python", "md", "python"])
        self.assertEqual(nb.line_text(3, 1), "y = x + 1")

    def test_plain_script_is_one_cell(self):
        nb = parse_text("a = 1\nb = 2\n", "plain.py")
        self.assertEqual(len(nb.cells), 1)
        self.assertEqual(nb.line_text(1, 2), "b = 2")

    def test_archive_is_rejected_with_advice(self):
        from harness.notebook import parse_bytes
        with self.assertRaises(NotebookError) as ctx:
            parse_bytes(b"PK\x03\x04rest", "export.dbc")
        self.assertIn("Source file", str(ctx.exception))

    def test_bad_json(self):
        with self.assertRaises(NotebookError):
            parse_text("{not json", "broken.ipynb")


class CitationTests(unittest.TestCase):
    def test_parse_variants(self):
        self.assertEqual(parse_citation("C29:L4"), (29, 4, 4))
        self.assertEqual(parse_citation("[C26:L31-43]"), (26, 31, 43))
        self.assertEqual(parse_citation("C26:L31-L43"), (26, 31, 43))
        self.assertEqual(parse_citation("c26:l31–43"), (26, 31, 43))
        self.assertEqual(parse_citation({"cell": 4, "start": 1, "end": 2}), (4, 1, 2))
        self.assertIsNone(parse_citation("cell 4 line 1"))
        self.assertEqual(format_citation(4, 1), "C4:L1")
        self.assertEqual(format_citation(26, 31, 43), "C26:L31-43")


if __name__ == "__main__":
    unittest.main()
