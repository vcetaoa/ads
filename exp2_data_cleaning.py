# Experiment 02: Data Cleaning
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import sys

# Step 1: Load the Dataset
df = pd.read_csv('exp2_dataset.csv')

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

# Step 2: Remove Duplicate  Records
#
df = df.drop_duplicates()

print("\nDataset  after removing  duplicates: \n")
print(df)


#

# Step 3: Handle Missing Values
#
num_cols = df.select_dtypes(include=["number"]).columns
num_imputer = SimpleImputer(strategy="mean")
if len(num_cols) > 0:
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_cols = df.select_dtypes(exclude=["number", "datetime"]).columns
cat_imputer = SimpleImputer(strategy="most_frequent")
if len(cat_cols) > 0:
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

#

# Step 4: Standardize  Date Format
#
date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

#

# Step 5: Detect and Handle Outliers
# (Using IQR Method)
#

if len(num_cols) > 0:
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        ~(
            (df[num_cols] < (Q1 - 1.5 * IQR))
            | (df[num_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)
    ]

#

# Step 6: Encode Categorical  Variable
# (Label Encoding)
#

le = LabelEncoder()
for col in cat_cols:
    if col not in date_cols:
        df[f"{col}_Encoded"] = le.fit_transform(df[col].astype(str))


#

# Final Cleaned Dataset
#
print("\nFinal Cleaned Dataset: \n")
print(df)
