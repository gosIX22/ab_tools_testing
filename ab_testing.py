from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from typing import Optional, List, Tuple


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
        :param a: a group of samples
        :param b: b group of samples
        :return: p_value number
        """
        _, pvalue = stats.ttest_ind(a, b)
        return pvalue

    @staticmethod
    def mannwhitneyu(a: np.array, b: np.array, hypothesis: str = "two-sided") -> float:
        """
        :param a: a group of samples
        :param b: b group of samples
        :param hypothesis: Defines the alternative hypothesis. {'two-sided', 'less', 'greater'}
        :return: p_value number
        """
        _, pvalue = stats.mannwhitneyu(a, b, alternative=hypothesis)
        return pvalue

    def bootstrap(self, a: np.array, b: np.array, func=np.mean, n: int = 1000) -> float:
        """
        :param a:
        :param b:
        :param func:
        :param n:
        :return:
        """
        a_bootstrap = np.random.choice(a, size=(len(a), n))
        b_bootstrap = np.random.choice(b, size=(len(b), n))
        list_diff = func(a_bootstrap, axis=0) - func(b_bootstrap, axis=0)
        left_bound = np.quantile(list_diff, self.ALPHA / 2)
        right_bound = np.quantile(list_diff, 1 - self.ALPHA / 2)
        return 1 if (left_bound > 0) or (right_bound < 0) else 0


def ab_group_split(salt: str = ""):
    """
    :param salt:
    :return:
    """
    pass


def aa_test(a_prev: np.array, b_prev: np.array, size: int = 100, stat_test=ABTesting.t_test, n: int = 1000):
    # Needs rework
    p_vals = []
    for _ in range(n):
        a = np.random.choice(a_prev, replace=False, size=size)
        b = np.random.choice(b_prev, replace=False, size=size)
        p_val = stat_test(a, b)
        p_vals.append(p_val)
    plt.hist(p_vals, cumulative=True, density=True, bins=50)
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), c="r")


def get_sample_size_abs(epsilon: float, std_1: float, std_2: float, alpha: float = 0.05, beta: float = 0.2) -> int:
    """
    :param epsilon:
    :param std_1:
    :param std_2:
    :param alpha:
    :param beta:
    :return:
    """
    t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)
    z_scores_sum_squared = (t_alpha + t_beta) ** 2
    sample_size = int(np.ceil(z_scores_sum_squared * (std_1**2 + std_2**2) / (epsilon**2)))
    return sample_size


def get_sample_size_arb(
    mu: float, std_1: float, std_2: float, eff: float = 1.01, alpha: float = 0.05, beta: float = 0.2
) -> int:
    """
    :param mu:
    :param std_1:
    :param std_2:
    :param eff:
    :param alpha:
    :param beta:
    :return:
    """
    epsilon = (eff - 1) * mu

    return get_sample_size_abs(epsilon, std_1=std_1, std_2=std_2, alpha=alpha, beta=beta)


def get_minimal_determinable_effect(
    std_1: float, std_2: float, sample_size: int, alpha: float = 0.05, beta: float = 0.2
) -> float:
    """
    :param std_1:
    :param std_2:
    :param sample_size:
    :param alpha:
    :param beta:
    :return:
    """
    t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)
    disp_sum_sqrt = (std_1**2 + std_2**2) ** 0.5
    mde = (t_alpha + t_beta) * disp_sum_sqrt / np.sqrt(sample_size)
    return mde


def get_table_sample_size(mu: float, std_1: float, std_2: float, effects: np.array, errors: np.array) -> pd.DataFrame:
    """
    :param mu:
    :param std_1:
    :param std_2:
    :param effects:
    :param errors:
    :return:
    """
    results = []
    for eff in effects:
        results_eff = []
        for err in errors:
            results_eff.append(get_sample_size_arb(mu, std_1, std_2, eff=eff, alpha=err, beta=err))
        results.append(results_eff)

    df_results = pd.DataFrame(results)
    df_results.index = pd.MultiIndex(
        levels=[[f"{np.round((x - 1) * 100, 1)}%" for x in effects]], codes=[np.arange(len(effects))], names=["effects"]
    )
    df_results.columns = pd.MultiIndex.from_tuples([(err,) for err in errors], names=["errors"])
    return df_results


def _get_sample_mean(data, size):
    return np.random.choice(data, size, False).mean()


def calc_stats(
    strata: list,
    sample_size: int = 100,
    n_iter: int = 1000,
    is_stratified_var: bool = False,
    is_stratified_sampling: bool = True,
) -> Tuple[float, float]:
    if is_stratified_var:
        data = np.concatenate(strata)
        means = [_get_sample_mean(data, sample_size) for _ in range(n_iter)]

        return np.mean(means), np.var(means)
    else:
        strata_sizes = [len(stratum) for stratum in strata]
        full_size = np.sum(strata_sizes)
        weights = np.array(strata_sizes) / full_size

        sample_sizes = np.zeros(shape=len(strata))
        if is_stratified_sampling:
            sample_sizes = (weights * sample_size + 0.5).astype(int)
        else:
            while (np.array(sample_sizes)).min() == 0:
                sample_sizes = np.random.default_rng().multinomial(sample_size, weights)

        means = []
        for _ in range(n_iter):
            strata_means = [_get_sample_mean(stratum, size) for stratum, size in zip(strata, sample_sizes)]
            means.append((weights * np.array(strata_means)).sum())
        return np.mean(means), np.var(means)
