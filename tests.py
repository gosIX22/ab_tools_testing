import unittest
import pandas as pd
import numpy as np
from ab_testing import get_table_sample_size


class UtilsTestCase(unittest.TestCase):

    def test_get_table_sample_size(self):
        # Needs refactor
        mu = 10
        std_1 = 2
        std_2 = 2
        effects = np.linspace(1.01, 1.1, 10)
        errors = [0.05, 0.1, 0.15, 0.2]

        expected_columns = pd.MultiIndex.from_tuples([(0.05,), (0.1,), (0.15,), (0.2,)], names=["errors"])
        expected_index = pd.MultiIndex.from_tuples([("1.0%",), ("2.0%",), ("3.0%",), ("4.0%",), ("5.0%",),
                                                    ("6.0%",), ("7.0%",), ("8.0%",), ("9.0%",), ("10.0%",)],
                                                   names=["effects"])
        expected_values = np.array([[10396, 6852, 4905, 3607], [2599, 1713, 1227, 902], [1156, 762, 545, 401],
                                    [650, 429, 307, 226], [416, 275, 197, 145], [289, 191, 137, 101],
                                    [213, 140, 101, 74], [163, 108, 77, 57], [129, 85, 61, 45], [104, 69, 50, 37]])
        expected_df = pd.DataFrame(expected_values, index=expected_index, columns=expected_columns, )

        result_df = get_table_sample_size(mu, std_1, std_2, effects, errors)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(result_df.shape, expected_df.shape)
        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == '__main__':
    unittest.main()
