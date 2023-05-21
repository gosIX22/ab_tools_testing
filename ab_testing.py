from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import hashlib
from typing import Tuple


class ABTesting:
    """
    A class for performing hypothesis testing using different statistical tests and methods.

    Parameters:
        alpha (float): The significance level for the test (default: 0.05).
        beta (float): The type II error rate for the test (default: 0.2).

    Methods:
        calc_mde(): Calculates the Minimum Detectable Effect (MDE).
        t_test(a, b): Performs a t-test to compare two groups of samples and returns the p-value.
        mannwhitneyu(a, b, hypothesis): Performs a Mann-Whitney U test to compare two groups of samples and returns the p-value.
        bootstrap(a, b, func, n): Performs bootstrap hypothesis testing by resampling the data and comparing the distribution of statistics.

    """
    def __init__(self, alpha: float = 0.05, beta: float = 0.2):
        """
        Initializes the HypothesisTesting class with specified significance level and type II error rate.

        Parameters:
            alpha (float): The significance level for the test (default: 0.05).
            beta (float): The type II error rate for the test (default: 0.2).
        """
        self.ALPHA = alpha
        self.BETA = beta

    @staticmethod
    def t_test(a: np.array, b: np.array) -> float:
        """
        Performs a t-test to compare two groups of samples and returns the p-value.

        Parameters:
            a (np.array): Group A samples.
            b (np.array): Group B samples.

        Returns:
            float: The calculated p-value from the t-test.
        """
        _, pvalue = stats.ttest_ind(a, b)
        return pvalue

    @staticmethod
    def mannwhitneyu(a: np.array, b: np.array, hypothesis: str = "two-sided") -> float:
        """
        Performs a Mann-Whitney U test to compare two groups of samples and returns the p-value.

        Parameters:
            a (np.array): Group A samples.
            b (np.array): Group B samples.
            hypothesis (str): Defines the alternative hypothesis. Options: 'two-sided', 'less', 'greater'. (default: 'two-sided')

        Returns:
            float: The calculated p-value from the Mann-Whitney U test.
        """
        _, pvalue = stats.mannwhitneyu(a, b, alternative=hypothesis)
        return pvalue

    def bootstrap(self, a: np.array, b: np.array, func=np.mean, n: int = 1000) -> float:
        """
        Performs bootstrap hypothesis testing by resampling the data and comparing the distribution of statistics.

        Parameters:
            a (np.array): Group A samples.
            b (np.array): Group B samples.
            func (function): The function used to compute the statistic. (default: np.mean)
            n (int): The number of bootstrap iterations. (default: 1000)

        Returns:
            float: The result of the hypothesis test (1 or 0) based on the bootstrap results.
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
    """
    Perform an A/A test by randomly sampling from two groups and plotting the cumulative histogram of p-values.

    Parameters:
        a_prev (np.array): Array representing the samples of group A.
        b_prev (np.array): Array representing the samples of group B.
        size (int): Size of the random samples to be drawn from each group (default: 100).
        stat_test (function): Statistical test function to compare the two groups (default: ABTesting.t_test).
        n (int): Number of iterations to perform the test (default: 1000).

    Returns:
        None

    """
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
    Calculate the sample size required for a two-sample comparison of means with a given absolute difference.

    Parameters:
        epsilon (float): The desired absolute difference in means.
        std_1 (float): The standard deviation of the first group.
        std_2 (float): The standard deviation of the second group.
        alpha (float): The significance level (default: 0.05).
        beta (float): The desired power of the test (default: 0.2).

    Returns:
        int: The calculated sample size required for the comparison.

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
    Calculate the sample size required for a two-sample comparison of means with a given effect size.

    Parameters:
        mu (float): The assumed common mean of the two groups.
        std_1 (float): The standard deviation of the first group.
        std_2 (float): The standard deviation of the second group.
        eff (float): The desired effect size, defined as the ratio of the difference in means to the common standard deviation (default: 1.01).
        alpha (float): The significance level (default: 0.05).
        beta (float): The desired power of the test (default: 0.2).

    Returns:
        int: The calculated sample size required for the comparison.

    """
    epsilon = (eff - 1) * mu

    return get_sample_size_abs(epsilon, std_1=std_1, std_2=std_2, alpha=alpha, beta=beta)


def get_minimal_determinable_effect(
    std_1: float, std_2: float, sample_size: int, alpha: float = 0.05, beta: float = 0.2
) -> float:
    """
    Calculate the minimal determinable effect (MDE) for a two-sample comparison of means.

    Parameters:
        std_1 (float): The standard deviation of the first group.
        std_2 (float): The standard deviation of the second group.
        sample_size (int): The sample size for each group.
        alpha (float): The significance level (default: 0.05).
        beta (float): The desired power of the test (default: 0.2).

    Returns:
        float: The minimal determinable effect (MDE) for the comparison.

    """
    t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)
    disp_sum_sqrt = (std_1**2 + std_2**2) ** 0.5
    mde = (t_alpha + t_beta) * disp_sum_sqrt / np.sqrt(sample_size)
    return mde


def get_table_sample_size(mu: float, std_1: float, std_2: float, effects: np.array, errors: np.array) -> pd.DataFrame:
    """
    Generates a table of sample sizes based on specified parameters.

    Parameters:
        mu (float): The desired effect size.
        std_1 (float): The standard deviation of the first group.
        std_2 (float): The standard deviation of the second group.
        effects (np.array): An array of effect sizes to calculate sample sizes for.
        errors (np.array): An array of error rates (alpha and beta) to calculate sample sizes for.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated sample sizes.

    Example:
        mu = 0.5
        std_1 = 1.0
        std_2 = 1.2
        effects = np.array([0.2, 0.5, 0.8])
        errors = np.array([0.05, 0.1, 0.2])
        result = get_table_sample_size(mu, std_1, std_2, effects, errors)
        print(result)
        # Output: DataFrame with the calculated sample sizes for different effect sizes and error rates.

    Note:
        The function calculates the sample sizes required to detect the specified effect sizes with the given error rates.
        It uses the `get_sample_size_arb` function to calculate the sample sizes for each combination of effect size and error rate.
        The resulting table is returned as a DataFrame, with effect sizes as the index and error rates as the columns.
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
    """
    Calculates statistics (mean and variance) for a given set of strata.

    Parameters:
        strata (list): A list containing the data in each stratum.
        sample_size (int): The size of the sample to be drawn from the strata (default: 100).
        n_iter (int): The number of iterations to perform (default: 1000).
        is_stratified_var (bool): Flag indicating whether to calculate the mean and variance using stratified variance (default: False).
        is_stratified_sampling (bool): Flag indicating whether to perform stratified sampling (default: True).

    Returns:
        Tuple[float, float]: A tuple containing the calculated mean and variance.

    Example:
        stratum1 = [1, 2, 3, 4, 5]
        stratum2 = [6, 7, 8, 9, 10]
        strata = [stratum1, stratum2]
        result = calc_stats(strata, sample_size=50, n_iter=1000)
        print(result)
        # Output: Tuple containing the calculated mean and variance

    Note:
        The function uses random sampling techniques to estimate the mean and variance of the given strata.
        If `is_stratified_var` is True, the function calculates the mean and variance using stratified variance.
        If `is_stratified_sampling` is True, the function performs stratified sampling using the provided sample size and weights.

    """
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


def calculate_theta(y_control: np.array, y_pilot: np.array, y_control_cov: np.array, y_pilot_cov: np.array) -> float:
    """
    Calculates the value of theta, which represents the covariance-to-variance ratio, given the control and pilot datasets.

    Parameters:
        y_control (np.array): Array representing the control dataset.
        y_pilot (np.array): Array representing the pilot dataset.
        y_control_cov (np.array): Array representing the covariance values for the control dataset.
        y_pilot_cov (np.array): Array representing the covariance values for the pilot dataset.

    Returns:
        float: The calculated value of theta (covariance-to-variance ratio).

    Example:
        control_data = np.array([1, 2, 3])
        pilot_data = np.array([4, 5, 6])
        control_covariance = np.array([0.1, 0.2, 0.3])
        pilot_covariance = np.array([0.4, 0.5, 0.6])
        result = calculate_theta(control_data, pilot_data, control_covariance, pilot_covariance)
        print(result)
        # Output: The calculated value of theta
    """
    y = np.hstack([y_control, y_pilot])
    y_cov = np.hstack([y_control_cov, y_pilot_cov])
    covariance = np.cov(y_cov, y)[0, 1]
    variance = y_cov.var()
    theta = covariance / variance
    return theta
