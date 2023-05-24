# ab_testing

ab_testing is a Python module that provides functionality for performing hypothesis testing and statistical analysis of A/B tests. It offers various statistical tests, such as t-tests, Mann-Whitney U tests, and bootstrap hypothesis testing, to compare two groups of samples and evaluate the significance of observed differences.

[//]: # (## Installation)

[//]: # ()
[//]: # (You can install the ABTesting module using pip:)

[//]: # ()
[//]: # (```)

[//]: # (pip install ABTesting)

[//]: # (```)

## Usage

```python
from ab_testing import ABTesting
import numpy as np

# Create an instance of the ABTesting class
ab_test = ABTesting(alpha=0.05, beta=0.2)

# Perform a t-test
a = [1, 2, 3, 4, 5]
b = [2, 4, 6, 8, 10]
pvalue = ab_test.t_test(a, b)
print("t-test p-value:", pvalue)

# Perform a Mann-Whitney U test
pvalue = ab_test.mannwhitneyu(a, b, hypothesis='two-sided')
print("Mann-Whitney U test p-value:", pvalue)

# Perform a Shapiro-Wilk test
pvalue = ab_test.shapiro_test(a)
print("Shapiro-Wilk test p-value:", pvalue)

# Perform bootstrap hypothesis testing
result = ab_test.bootstrap(a, b, func=np.mean, n=1000)
print("Bootstrap test result:", result)
```

## Class Methods

- `t_test(a, b)`: Performs a t-test to compare two groups of samples and returns the p-value.
- `mannwhitneyu(a, b, hypothesis='two-sided')`: Performs a Mann-Whitney U test to compare two groups of samples and returns the p-value.
- `shapiro_test(a)`: Performs the Shapiro-Wilk test for normality on a given array and returns the p-value.
- `bootstrap(a, b, func=np.mean, n=1000)`: Performs bootstrap hypothesis testing by resampling the data and comparing the distribution of statistics.

## Parameters

- `alpha`: The significance level for the test. Default: 0.05.
- `beta`: The type II error rate for the test. Default: 0.2.

## Contributing

Contributions to the ABTesting module are welcome! If you find any issues or have suggestions for improvement, please create an issue on the GitHub repository.
