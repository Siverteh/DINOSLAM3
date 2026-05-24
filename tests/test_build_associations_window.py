from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
import sys


class BuildAssociationsWindowTests(unittest.TestCase):
    def test_ratio_window_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "associations.txt"
            out = root / "subset.txt"
            lines = ["# header"]
            for i in range(10):
                lines.append(f"{i}.0 rgb/{i}.png {i}.0 depth/{i}.png")
            inp.write_text("\n".join(lines) + "\n", encoding="utf-8")

            script = Path(__file__).resolve().parents[1] / "scripts" / "build_associations_window.py"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(inp),
                    "--output",
                    str(out),
                    "--start-ratio",
                    "0.2",
                    "--end-ratio",
                    "0.5",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertTrue(out.exists())
            out_lines = [x.strip() for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
            # 10 lines -> [2,5) gives 3 lines: 2,3,4
            self.assertEqual(len(out_lines), 3)
            self.assertTrue(out_lines[0].startswith("2.0 "))
            self.assertTrue(out_lines[-1].startswith("4.0 "))

    def test_index_window_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "associations.txt"
            out = root / "subset_idx.txt"
            lines = ["# header"]
            for i in range(8):
                lines.append(f"{i}.0 rgb/{i}.png {i}.0 depth/{i}.png")
            inp.write_text("\n".join(lines) + "\n", encoding="utf-8")

            script = Path(__file__).resolve().parents[1] / "scripts" / "build_associations_window.py"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(inp),
                    "--output",
                    str(out),
                    "--start-idx",
                    "2",
                    "--end-idx",
                    "5",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            out_lines = [x.strip() for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual(len(out_lines), 3)
            self.assertTrue(out_lines[0].startswith("2.0 "))
            self.assertTrue(out_lines[-1].startswith("4.0 "))


if __name__ == "__main__":
    unittest.main()
