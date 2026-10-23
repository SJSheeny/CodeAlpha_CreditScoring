"""
Creates credit_data.csv - a SYNTHETIC (computer-generated) credit dataset.
It copies the column layout of the Kaggle "Give Me Some Credit" file
(cs-training.csv), so real data can be swapped in later without changing
the model code. The people in this file are NOT real.

How to run:   python generate_data.py
Output:       credit_data.csv  (in the same folder)
"""

import os
import numpy as np
import pandas as pd

SEED = 42                    # same seed = same file every time
N_ROWS = 30000               # number of fake applicants
TARGET_DEFAULT_RATE = 0.067  # about 6-7% default, similar to the real Kaggle data

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
rng = np.random.default_rng(SEED)
n = N_ROWS

#Step 1: create the applicants' basic details
age = np.clip(rng.normal(52, 14, n), 21, 95).round().astype(int)
monthly_income = rng.lognormal(mean=8.6, sigma=0.6, size=n).round()      # typical ~ 5,000 / month
utilization = rng.beta(1.2, 3.0, n)                                      # share of credit limit used
outliers = rng.random(n) < 0.02                                          # a few extreme values, like real data
utilization[outliers] *= rng.uniform(1.0, 2.5, outliers.sum())
utilization = utilization.round(4)
debt_ratio = np.clip(rng.lognormal(mean=-1.0, sigma=0.8, size=n), 0, 10).round(4)
open_lines = rng.poisson(8, n)
real_estate_loans = rng.poisson(1.0, n)
dependents = rng.poisson(0.8, n).astype(float)

#Step 2: late payments (people who pay late once tend to pay late again)
propensity = rng.exponential(1.0, n)                 # hidden "tends to pay late" level
scale = propensity * (1 + utilization)
late_30_59 = rng.poisson(0.12 * scale)
late_60_89 = rng.poisson(0.05 * scale)
late_90 = rng.poisson(0.06 * scale)

#Step 3: decide who defaults (the hidden rule the model must discover) 
raw = (2.2 * np.minimum(utilization, 1.5)
       + 0.9 * late_30_59 + 1.4 * late_60_89 + 1.6 * late_90
       - 0.05 * (age - 50)
       + 0.5 * np.minimum(debt_ratio, 3)
       - 0.4 * (np.log(monthly_income) - 8.6)
       + rng.normal(0, 0.3, n))                      # randomness = real life

# Find the intercept that gives the default rate we want (simple bisection search)
low, high = -15.0, 5.0
for _ in range(60):
    mid = (low + high) / 2
    mean_prob = (1 / (1 + np.exp(-(raw + mid)))).mean()
    low, high = (low, mid) if mean_prob > TARGET_DEFAULT_RATE else (mid, high)
prob_default = 1 / (1 + np.exp(-(raw + (low + high) / 2)))
default = (rng.random(n) < prob_default).astype(int)

#Step 4: build the table using the Kaggle column names
df = pd.DataFrame({
    "SeriousDlqin2yrs": default,                       # 1 = defaulted / seriously late within 2 years
    "RevolvingUtilizationOfUnsecuredLines": utilization,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": monthly_income,
    "NumberOfOpenCreditLinesAndLoans": open_lines,
    "NumberOfTimes90DaysLate": late_90,
    "NumberRealEstateLoansOrLines": real_estate_loans,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
    "NumberOfDependents": dependents,
})

# Real data has gaps: hide ~20% of incomes and ~2.6% of dependents
df.loc[rng.random(n) < 0.20, "MonthlyIncome"] = np.nan
df.loc[rng.random(n) < 0.026, "NumberOfDependents"] = np.nan

out_path = os.path.join(BASE_DIR, "credit_data.csv")
df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows, columns:", df.shape)
print("Default rate:", round(df["SeriousDlqin2yrs"].mean(), 3))
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
