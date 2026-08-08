import unittest

import numpy as np

from seed.data.elm import et_from_components, soil_moisture_layer
from seed.data.nldas import aggregate_daily


class DataProcessingTests(unittest.TestCase):
    def test_precipitation_is_summed(self):
        hourly = np.arange(48, dtype=np.float32).reshape(48, 1, 1)
        daily = aggregate_daily(hourly, "PRECTmms")
        np.testing.assert_array_equal(daily[:, 0, 0], [sum(range(24)), sum(range(24, 48))])

    def test_other_forcings_are_averaged(self):
        hourly = np.arange(24, dtype=np.float32).reshape(24, 1, 1)
        self.assertEqual(float(aggregate_daily(hourly, "TBOT")[0, 0, 0]), 11.5)

    def test_et_conversion_and_clipping(self):
        result = et_from_components(np.array([-1.0, 1.0]), np.zeros(2), np.zeros(2))
        np.testing.assert_array_equal(result, [0.0, 86400.0])

    def test_soil_layer_index_two(self):
        values = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
        np.testing.assert_array_equal(soil_moisture_layer(values), values[:, 2])


if __name__ == "__main__":
    unittest.main()
