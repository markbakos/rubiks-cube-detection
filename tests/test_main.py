from pathlib import Path
import subprocess
import sys
import unittest


class EntryPointTests(unittest.TestCase):
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("single-face recognition", result.stdout)


if __name__ == "__main__":
    unittest.main()
