import unittest
import pandas as pd
import numpy as np
from ab_testing import calc_stats, get_table_sample_size


class UtilsTestCase(unittest.TestCase):

    def test_calc_stats(self):
        stratum1 = [1, 2, 3, 4, 5]
        stratum2 = [6, 7, 8, 9, 10]
        strata = [stratum1, stratum2]
        sample_size = 50
        n_iter = 1000

        # Test case 1: Default parameters (without stratified variance)
        expected_mean = 5.5
        expected_var = 2.9175
        result_mean, result_var = calc_stats(strata, sample_size, n_iter, False)
        self.assertAlmostEqual(result_mean, expected_mean, places=3)
        self.assertAlmostEqual(result_var, expected_var, places=3)

        # Test case 2: With stratified variance
        expected_mean = 5.5
        expected_var = 2.904
        result_mean, result_var = calc_stats(strata, sample_size, n_iter, True)
        self.assertAlmostEqual(result_mean, expected_mean, places=3)
        self.assertAlmostEqual(result_var, expected_var, places=3)

        # Test case 3: With stratified sampling disabled
        expected_mean = 5.5
        expected_var = 2.875
        result_mean, result_var = calc_stats(strata, sample_size, n_iter, False, False)
        self.assertAlmostEqual(result_mean, expected_mean, places=3)
        self.assertAlmostEqual(result_var, expected_var, places=3)

    def test_get_table_sample_size(self):
        mu = 0.5
        std_1 = 1.0
        std_2 = 1.2
        effects = np.array([0.2, 0.5, 0.8])
        errors = np.array([0.05, 0.1, 0.2])

        expected_columns = pd.MultiIndex.from_tuples([(0.05,), (0.1,), (0.2,)], names=["errors"])
        expected_index = pd.MultiIndex.from_tuples([("20.0%",), ("50.0%",), ("80.0%",)], names=["effects"])
        expected_values = np.array([[81, 31, 14], [32, 12, 6], [13, 5, 3]])
        expected_df = pd.DataFrame(expected_values, index=expected_index, columns=expected_columns)

        result_df = get_table_sample_size(mu, std_1, std_2, effects, errors)

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(result_df.shape, expected_df.shape)
        pd.testing.assert_frame_equal(result_df, expected_df)
