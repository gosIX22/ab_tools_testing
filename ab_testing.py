from scipy import stats
import hashlib
from typing import Optional, List


class ABTesting:
    def __init__(self):
        self.ALPHA = 0.05
        self.BETA = 0.20
        pass

    def calc_mde(self):
        pass

    def t_test(self, a, b):
        _, pvalue = stats.ttest_ind(a, b)
        return pvalue

    def mannwhitneyu(self, a, b):
        _, pvalue = stats.mannwhitneyu(a, b, alternative='two-sided')
        return pvalue

    def bootstrap(self):
        pass


def ab_group_split(salt=''):
    pass
