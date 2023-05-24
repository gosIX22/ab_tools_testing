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

    @staticmethod
    def shapiro_test(a: np.array) -> float:
        """
        Perform the Shapiro-Wilk test for normality on a given array.

        The Shapiro-Wilk test is used to determine whether a given sample follows a normal distribution.
        It calculates a test statistic and p-value based on the observed data.

        Parameters:
            a (np.array): Array of sample data.

        Returns:
            float: p-value from the Shapiro-Wilk test.

        Example:
            data = np.array([1.2, 1.5, 1.8, 2.1, 2.4])
            p_value = shapiro(data)
            print("p-value:", p_value)
            # Output: p-value: 0.8936132788658142

        Note:
            The Shapiro-Wilk test assumes that the sample size should be between 3 and 5000.
            If the sample size is outside this range, the result may not be accurate.
        """
        _, pvalue = stats.shapiro(a)
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


def set_plot_params():
    """
    Set the parameters for matplotlib plot styling.

    This function sets various parameters for matplotlib plot styling to customize the appearance
    of plots generated using matplotlib.

    Note:
        This function modifies the default plot parameters for the current session.
        Any plots created after calling this function will reflect the updated styling.
    """
    titlesize = 24
    labelsize = 22
    legendsize = 22
    xticksize = 18
    yticksize = xticksize

    plt.rcParams["legend.markerscale"] = 1.5
    plt.rcParams["legend.handletextpad"] = 0.5
    plt.rcParams["legend.labelspacing"] = 0.4
    plt.rcParams["legend.borderpad"] = 0.5
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["axes.labelsize"] = labelsize
    plt.rcParams["axes.titlesize"] = titlesize
    plt.rcParams["figure.figsize"] = (10, 6)

    plt.rc("xtick", labelsize=xticksize)
    plt.rc("ytick", labelsize=yticksize)
    plt.rc("legend", fontsize=legendsize)


def bootstrap_ratio(a: np.array, b: np.array, n: int = 1000) -> Tuple[float, float]:
    """
    Perform a bootstrap test to compare the ratio of two groups.

    This function compares the ratio between two groups (a and b) using a bootstrap test.
    It calculates the p-value and the difference in ratios between the groups.

    Parameters:
        a (array-like): Data of group A.
        b (array-like): Data of group B.
        n (int): Number of bootstrap iterations (default: 1000).

    Returns:
        Tuple[float, float]: A tuple containing the p-value and the difference in ratios.

    Example:
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8, 9, 10]
        p_value, delta = bootstrap_ratio(a, b)
        print("p-value:", p_value)
        print("difference in ratios:", delta)

    Note:
        The function performs a bootstrap test by resampling with replacement from the original groups.
        It calculates the difference in ratios between the resampled groups and compares it to the original difference.
        The p-value is calculated using a two-tailed test based on the normal distribution.
    """
    len_a = len(a)
    len_b = len(b)
    a_sum_count = np.zeros((len_a, 2))
    a_sum_count[:, 0] = np.array([np.sum(row) for row in a])
    a_sum_count[:, 1] = np.array([len(row) for row in a])
    b_sum_count = np.zeros((len_b, 2))
    b_sum_count[:, 0] = np.array([np.sum(row) for row in b])
    b_sum_count[:, 1] = np.array([len(row) for row in b])

    list_diff = []
    for _ in range(n):
        a_bootstrap_index = np.random.choice(np.arange(len_a), len_a)
        b_bootstrap_index = np.random.choice(np.arange(len_b), len_b)
        a_bootstrap = a_sum_count[a_bootstrap_index]
        b_bootstrap = b_sum_count[b_bootstrap_index]
        a_metric = a_bootstrap[:, 0].sum() / a_bootstrap[:, 1].sum()
        b_metric = b_bootstrap[:, 0].sum() / b_bootstrap[:, 1].sum()
        list_diff.append(b_metric - a_metric)
    delta = (
            b_sum_count[:, 0].sum() / b_sum_count[:, 1].sum()
            - a_sum_count[:, 0].sum() / a_sum_count[:, 1].sum()
    )
    std = np.std(list_diff)
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(delta / std)))
    return pvalue, delta


def delta_method(a: np.array, b: np.array) -> Tuple[float, float]:
    """
    Perform a statistical test using the Delta method to compare two groups.

    This function compares the means between two groups (a and b) using the Delta method.
    It calculates the p-value and the difference in means between the groups.

    Parameters:
        a (array-like): Data of group A.
        b (array-like): Data of group B.

    Returns:
        Tuple[float, float]: A tuple containing the p-value and the difference in means.

    Example:
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8, 9], [10, 11, 12]]
        p_value, delta = delta_method(a, b)
        print("p-value:", p_value)
        print("difference in means:", delta)

    Note:
        The function calculates the mean and standard deviation for each group.
        It uses the Delta method to estimate the variance of the difference in means.
        The test statistic is computed as the difference in means divided by the standard error.
        The p-value is calculated using a two-tailed test based on the normal distribution.
    """
    dict_stats = {'a': {'data': a}, 'b': {'data': b}}
    for key, dict_ in dict_stats.items():
        data = dict_['data']
        dict_['x'] = np.array([np.sum(row) for row in data])
        dict_['y'] = np.array([len(row) for row in data])
        dict_['metric'] = np.sum(dict_['x']) / np.sum(dict_['y'])
        dict_['len'] = len(data)
        dict_['mean_x'] = np.mean(dict_['x'])
        dict_['mean_y'] = np.mean(dict_['y'])
        dict_['std_x'] = np.std(dict_['x'])
        dict_['std_y'] = np.std(dict_['y'])
        dict_['cov_xy'] = np.cov(dict_['x'], dict_['y'])[0, 1]
        dict_['var_metric'] = (
                                      (dict_['std_x'] ** 2) / (dict_['mean_y'] ** 2)
                                      + (dict_['mean_x'] ** 2) / (dict_['mean_y'] ** 4) * (dict_['std_y'] ** 2)
                                      - 2 * dict_['mean_x'] / (dict_['mean_y'] ** 3) * dict_['cov_xy']
                              ) / dict_['len']
    var = dict_stats['b']['var_metric'] + dict_stats['a']['var_metric']
    delta = dict_stats['b']['metric'] - dict_stats['a']['metric']
    statistic = delta / np.sqrt(var)
    pvalue = (1 - stats.norm.cdf(np.abs(statistic))) * 2
    return pvalue, delta


def method_linearization(a: np.array, b: np.array) -> Tuple[float, float]:
    """
    Perform a statistical test using the method of linearization to compare two groups.

    This function compares the means between two groups (a and b) using the method of linearization.
    It calculates the p-value and the difference in means between the groups.

    Parameters:
        a (np.array): Data of group A.
        b (np.array): Data of group B.

    Returns:
        Tuple[float, float]: A tuple containing the p-value and the difference in means.

    Example:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9], [10, 11, 12]])
        p_value, delta = method_linearization(a, b)
        print("p-value:", p_value)
        print("difference in means:", delta)

    Note:
        The function calculates the sum and count for each group.
        It estimates the coefficient using the ratio of the sum of x values to the sum of y values in group A.
        The linearized values are obtained by subtracting the estimated linear component from the original values.
        The p-value is calculated using a two-sample independent t-test on the linearized values.
        The difference in means is computed as the mean of the linearized values in group B minus the mean in group A.
    """
    a_x = np.array([np.sum(row) for row in a])
    a_y = np.array([len(row) for row in a])
    b_x = np.array([np.sum(row) for row in b])
    b_y = np.array([len(row) for row in b])
    coef = np.sum(a_x) / np.sum(a_y)
    a_lin = a_x - coef * a_y
    b_lin = b_x - coef * b_y
    _, pvalue = stats.ttest_ind(a_lin, b_lin)
    delta = np.mean(b_lin) - np.mean(a_lin)
    return pvalue, delta
