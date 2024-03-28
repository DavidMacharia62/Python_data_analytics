#!/usr/bin/python3

import numpy as np
from scipy.stats import ttest_ind
import statsmodels.api as sm

# Define sample data
data1 = [1, 2, 3, 4, 5]
data2 = [2, 3, 4, 5, 6]

# Define your independent variable(s) here
X = np.array([1, 2, 3, 4, 5])

# Define your dependent variable (y) here
y = np.array([2, 3, 4, 5, 6])

# Add a constant to the independent variable(s) for the regression model
X = sm.add_constant(X)

# Continue with your regression analysis...

# Perform a t-test
t_stat, p_value = ttest_ind(data1, data2)
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Perform linear regression
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

