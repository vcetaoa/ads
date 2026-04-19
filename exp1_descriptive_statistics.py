# Experiment 01: Descriptive Statistics
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load Dataset
df = pd.read_csv('exp1_dataset.csv')

print("\n--- Dataset Validation ---")
print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nDataset Info:")
df.info()
print("\nFirst 5 Rows:")
print(df.head())
print("--------------------------\n")

#
# 1. Measure of Central Tendency
#
print("\nMEASURE  OF CENTRAL TENDENCY \n")

print("Mean:\n", df.mean(numeric_only=True))
print("\nMedian: \n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True))

#
# 2. Measure of Spread / Dispersion
#
print("\nMEASURE  OF DISPERSION \n")

print("Variance: \n", df.var(numeric_only=True))
print("\nStandard  Deviation: \n", df.std(numeric_only=True))
print("\nRange:\n", df.max(numeric_only=True) - df.min(numeric_only=True))

# Interquartile Range (IQR)
Q1 = df.quantile(0.25, numeric_only=True)
Q3 = df.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1
print("\nInterquartile  Range (IQR):\n", IQR)
# Boxplot
df.select_dtypes(include="number").plot(kind="box")
plt.title("Boxplot of Numerical Variables")
plt.show()


#
# 3. Measure of Symmetry  / Shape
#
print("\nMEASURE OF SHAPE\n")

# Skewness
print("Skewness: \n", df.skew(numeric_only=True))

# Kurtosis
print("\nKurtosis: \n", df.kurt(numeric_only=True))
