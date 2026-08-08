import unittest
from pathlib import Path

import numpy as np


class FigureDataTests(unittest.TestCase):
    def test_sensitivity_contract(self):
        root = Path(__file__).resolve().parents[2]
        for target in ("et", "sm"):
            for horizon in (7, 30, 90):
                with np.load(root / "figures" / "data" / f"{target}_{horizon}d_analysis.npz") as data:
                    self.assertEqual(str(data["sensitivity_start"]), "2020-08-13")
                    self.assertEqual(str(data["sensitivity_end"]), "2020-12-31")
                    self.assertEqual(int(data["sensitivity_date_count"]), 141)
                    self.assertTrue(np.isfinite(data["sensitivity_rmse_gain"]).any())
                    self.assertTrue(np.isfinite(data["sensitivity_kge_gain"]).any())


if __name__ == "__main__":
    unittest.main()
