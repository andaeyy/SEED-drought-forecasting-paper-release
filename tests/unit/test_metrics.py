import unittest

import numpy as np

from seed.evaluation.metrics import kge, kge_gain, rmse, rmse_gain


class MetricTests(unittest.TestCase):
    def test_perfect_rmse_and_kge(self):
        values = np.array([1.0, 2.0, 3.0])
        rmse_value, count = rmse(values, values, minimum_count=3)
        kge_value, correlation, alpha, beta, kge_count = kge(values, values, minimum_count=3)
        self.assertEqual(float(rmse_value), 0.0)
        self.assertAlmostEqual(float(kge_value), 1.0)
        self.assertAlmostEqual(float(correlation), 1.0)
        self.assertAlmostEqual(float(alpha), 1.0)
        self.assertAlmostEqual(float(beta), 1.0)
        self.assertEqual(int(count), int(kge_count))

    def test_positive_gain_favors_seed(self):
        self.assertEqual(float(rmse_gain(1.0, 2.0)), 1.0)
        self.assertAlmostEqual(float(kge_gain(0.8, 0.2)), 0.6)

    def test_paired_missing_values(self):
        observed = np.array([1.0, np.nan, 3.0])
        predicted = np.array([1.0, 2.0, 5.0])
        value, count = rmse(observed, predicted)
        self.assertAlmostEqual(float(value), np.sqrt(2.0))
        self.assertEqual(int(count), 2)


if __name__ == "__main__":
    unittest.main()
