from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.eval.focus071_tartanair_mono_paper_eval import (  # noqa: E402
    PAPER_DROID_VO,
    PAPER_OURS_DEFAULT,
    PAPER_OURS_FAST,
    TEST_SPLIT,
    _archive_is_valid,
    _google_drive_form,
    _looks_like_html_bytes,
    _mean_ate_from_rows,
    _median_per_sequence,
    _training_process_lines_from_ps_output,
)


class Focus071TartanAirMonoEvalTests(unittest.TestCase):
    def test_paper_rows_cover_full_test_split(self) -> None:
        self.assertEqual(set(PAPER_OURS_DEFAULT), set(TEST_SPLIT))
        self.assertEqual(set(PAPER_OURS_FAST), set(TEST_SPLIT))
        self.assertEqual(set(PAPER_DROID_VO), set(TEST_SPLIT))

    def test_mean_ate_from_rows(self) -> None:
        rows = [
            {"sequence": "ME000", "ate_rmse": "0.10"},
            {"sequence": "ME001", "ate_rmse": "0.20"},
            {"sequence": "ME002", "ate_rmse": "0.30"},
        ]
        self.assertAlmostEqual(_mean_ate_from_rows(rows), 0.20)

    def test_median_per_sequence(self) -> None:
        repeat_rows = [
            [
                {"sequence": "ME000", "ate_rmse": "0.10"},
                {"sequence": "ME001", "ate_rmse": "0.20"},
            ],
            [
                {"sequence": "ME000", "ate_rmse": "0.30"},
                {"sequence": "ME001", "ate_rmse": "0.10"},
            ],
            [
                {"sequence": "ME000", "ate_rmse": "0.20"},
                {"sequence": "ME001", "ate_rmse": "0.40"},
            ],
        ]
        medians = _median_per_sequence(repeat_rows)
        median_map = {row["sequence"]: row["median_ate_rmse"] for row in medians}
        self.assertAlmostEqual(float(median_map["ME000"]), 0.20)
        self.assertAlmostEqual(float(median_map["ME001"]), 0.20)

    def test_google_drive_form_parses_warning_page(self) -> None:
        html = """
        <html><body>
        <form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
          <input type="hidden" name="id" value="abc123"/>
          <input type="hidden" name="export" value="download"/>
          <input type="hidden" name="confirm" value="t"/>
          <input type="hidden" name="uuid" value="uuid-1"/>
        </form>
        </body></html>
        """
        form = _google_drive_form(html)
        self.assertIsNotNone(form)
        action, params = form or ("", {})
        self.assertEqual(action, "https://drive.usercontent.google.com/download")
        self.assertEqual(params["id"], "abc123")
        self.assertEqual(params["confirm"], "t")

    def test_html_bytes_detection(self) -> None:
        self.assertTrue(_looks_like_html_bytes(b"<!DOCTYPE html><html><head><title>Google Drive - Virus scan warning</title>"))
        self.assertFalse(_looks_like_html_bytes(b"\x1f\x8b\x08\x00binary"))

    def test_archive_is_valid_rejects_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "images.tar.gz"
            tar_path.write_text("<!DOCTYPE html><html>bad</html>", encoding="utf-8")
            self.assertFalse(_archive_is_valid(tar_path, "tar.gz"))

    def test_archive_is_valid_accepts_magic_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "images.tar.gz"
            zip_path = Path(tmpdir) / "groundtruth.zip"
            tar_path.write_bytes(b"\x1f\x8b\x08\x00rest")
            zip_path.write_bytes(b"PK\x03\x04rest")
            self.assertTrue(_archive_is_valid(tar_path, "tar.gz"))
            self.assertTrue(_archive_is_valid(zip_path, "zip"))

    def test_training_process_lines_filters_real_training_only(self) -> None:
        ps_output = "\n".join(
            [
                "100 1 bash -lc ps -eo pid,ppid,cmd | rg 'run_assoc9_sweep|train_dino_dpvo_frontend' || true",
                "101 100 rg run_assoc9_sweep|train_dino_dpvo_frontend",
                "200 1 /env/bin/python -m refocus_vo.sweeps.run_assoc9_sweep --manifest x.yaml",
                "201 200 bash /repo/refocus_vo/scripts/train_dino_dpvo_frontend.sh",
                "202 201 /env/bin/python -m refocus_vo.train_dino_dpvo_frontend --config y.yaml",
            ]
        )
        lines = _training_process_lines_from_ps_output(ps_output, exclude_pid=999)
        self.assertEqual(len(lines), 3)
        self.assertTrue(any("run_assoc9_sweep" in line for line in lines))
        self.assertTrue(any("scripts/train_dino_dpvo_frontend.sh" in line for line in lines))
        self.assertTrue(any("refocus_vo.train_dino_dpvo_frontend" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
