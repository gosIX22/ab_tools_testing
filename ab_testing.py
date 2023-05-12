from scipy import stats
import numpy as np
import hashlib
from typing import Optional, List


class ABTesting:
    def __init__(self, alpha: float = 0.05, beta: float = 0.2):
        """
        :param alpha:
        :param beta:
        """
        self.ALPHA = alpha
        self.BETA = beta

    def calc_mde(self) -> float:
        """
        :return:
        """
        pass

    @staticmethod
    def t_test(a: np.array, b: np.array) -> float:
        """
        :param a:
        :param b:
        :return:
        """
        _, pvalue = stats.ttest_ind(a, b)
        return pvalue

    @staticmethod
    def mannwhitneyu(a: np.array, b: np.array) -> float:
        """
        :param a:
        :param b:
        :return:
        """
        _, pvalue = stats.mannwhitneyu(a, b, alternative='two-sided')
        return pvalue

    def bootstrap(self, a: np.array, b: np.array, func=np.mean, n: int = 1000) -> float:
        """
        :param a:
        :param b:
        :param func:
        :param n:
        :param alpha:
        :return:
        """
        a_bootstrap = np.random.choice(a, size=(len(a), n))
        b_bootstrap = np.random.choice(b, size=(len(b), n))
        list_diff = func(a_bootstrap, axis=0) - func(b_bootstrap, axis=0)
        left_bound = np.quantile(list_diff, self.ALPHA / 2)
        right_bound = np.quantile(list_diff, 1 - self.ALPHA / 2)
        return 1 if (left_bound > 0) or (right_bound < 0) else 0


def ab_group_split(salt: str = ''):
    """
    :param salt:
    :return:
    """
    pass
