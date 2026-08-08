import unittest

import numpy as np

from seed.dryness.msdi import circular_climatology, drought_score, dryness_category


class DrynessTests(unittest.TestCase):
    def test_circular_climatology_shape(self):
        values = np.arange(4 * 365, dtype=np.float32).reshape(4 * 365, 1, 1, 1)
        mean, scale, count = circular_climatology(values)
        self.assertEqual(mean.shape, (365, 1, 1, 1))
        self.assertEqual(scale.shape, mean.shape)
        self.assertEqual(count.shape, mean.shape)
        self.assertTrue(np.all(count == 124))

    def test_low_joint_state_is_drier(self):
        low = float(drought_score(np.array([-2.0]), np.array([-2.0]))[0])
        high = float(drought_score(np.array([2.0]), np.array([2.0]))[0])
        self.assertGreater(low, high)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(low, 1.0)

    def test_category_thresholds(self):
        scores = np.array([0.0, 0.70, 0.80, 0.90, 0.95, 0.98])
        np.testing.assert_array_equal(dryness_category(scores), np.arange(6, dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
