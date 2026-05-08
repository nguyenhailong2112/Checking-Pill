from __future__ import annotations

import unittest
from pathlib import Path

from edgevision.demo import run_demo
from edgevision.schemas import QualityStatus


class DemoSmokeTest(unittest.TestCase):
    def test_demo_runs_full_contract(self) -> None:
        output_dir = Path("runs") / "tests" / "demo_smoke"
        report = run_demo(output_dir)

        self.assertEqual(report.total_count, 3)
        self.assertEqual(report.count_by_type, {"BlueCaplet": 1, "RoundCream": 2})
        self.assertEqual(report.image_status, QualityStatus.NG)
        self.assertEqual([item.quality.status for item in report.items].count(QualityStatus.NG), 1)
        self.assertTrue((output_dir / "inference" / "report.json").exists())
        self.assertTrue((output_dir / "inference" / "annotated.jpg").exists())


if __name__ == "__main__":
    unittest.main()
