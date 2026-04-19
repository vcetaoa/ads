# Experiment 03: Inferential Statistics
# Import required libraries
import pandas as pd
import numpy as np
from scipy import stats
import sys

# Load Dataset
df = pd.read_csv('titanic.csv')

print("\n--- Dataset Validation ---")
print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nDataset Info:")
df.info()
print("\nFirst 5 Rows:")
print(df.head())
print("--------------------------\n")


num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(exclude=["number", "datetime"]).columns

if len(num_cols) > 0:
    target_num1 = num_cols[0]
else:
    raise ValueError("No numeric columns found for tests")

target_num2 = num_cols[1] if len(num_cols) > 1 else target_num1

binary_cat = None
for col in cat_cols:
    if df[col].nunique() == 2:
        binary_cat = col
        break
if not binary_cat and len(cat_cols) > 0:
    binary_cat = cat_cols[0]

test_var1 = df[target_num1].dropna()

#

# 1. Z-Test
#
print("\n----- Z-Test  ----------  ")


sample_mean = np.mean(test_var1)
pop_mean = test_var1.mean() # using mean as pop mean for generic test
std = np.std(test_var1)
n = len(test_var1)

z = (sample_mean - pop_mean) / (std / np.sqrt(n)) if std > 0 else 0


print("Sample Mean:", sample_mean)
print("Z statistic:", z)


#

# 2. One Sample T-Test
#
print("\n----- One Sample T -Test  ----------  ")


test_var2 = df[target_num2].dropna()


t_stat1, p_value1 = stats.ttest_1samp(test_var2, test_var2.mean())


print("T Statistic:", t_stat1)
print("P Value:", p_value1)


if p_value1 < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


#

# 3. Independent  Sample T-Test
#
print("\n----- Independent Sample T -Test  ----------  ")

if binary_cat and df[binary_cat].nunique() >= 2:
    val1, val2 = df[binary_cat].dropna().unique()[:2]
    group1 = df[df[binary_cat] == val1][target_num2].dropna()
    group2 = df[df[binary_cat] == val2][target_num2].dropna()
else:
    mid = df[target_num2].median()
    group1 = df[df[target_num2] >= mid][target_num2].dropna()
    group2 = df[df[target_num2] < mid][target_num2].dropna()

t_stat2, p_value2 = stats.ttest_ind(group1, group2)


print("T Statistic:", t_stat2)
print("P Value:", p_value2)


if p_value2 < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


#

# 4. Paired Sample T-Test (Example  Data)
#
print("\n----- Paired Sample T -Test  ----------  ")


before = [10, 12, 13, 15, 16]

after = [12, 15, 14, 18, 20]


t_stat3, p_value3 = stats.ttest_rel(before, after)


print("T Statistic:", t_stat3)
print("P Value:", p_value3)


if p_value3 < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


print("\nInferential Statistics Analysis Completed")
