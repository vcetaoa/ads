# Experiment 06: Outlier Detection
# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

# Load Dataset
df = pd.read_csv('Mall_Customers.csv')

print("\n--- Dataset Validation ---")
print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nDataset Info:")
df.info()
print("\nFirst 5 Rows:")
print(df.head())
print("--------------------------\n")


# Drop unnecessary identifier columns dynamically
id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
if id_cols:
    df = df.drop(columns=id_cols)


# Convert categorical column

df = pd.get_dummies(df, drop_first=True)

# Convert boolean columns to integer
df = df.astype(int)


# Boxplot for visualization
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.title("Boxplot  for Outlier Detection")
plt.show()


# Use only numeric columns for IQR

numeric_df = df.select_dtypes(include=np.number)


# IQR method

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1


# Remove outliers

df_clean = df[
    ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
]


print("Original Shape:", df.shape)

print("After Removing Outliers:", df_clean.shape)


# Create target variable dynamically
num_cols_clean = df_clean.select_dtypes(include=np.number).columns
if len(num_cols_clean) > 0:
    target_base = num_cols_clean[-1]
else:
    raise ValueError("No numeric columns found for target")

median_val = df_clean[target_base].median()
df_clean["Target_Binary"] = (df_clean[target_base] > median_val).astype(int)

# Features and target
X = df_clean.drop(columns=[target_base, "Target_Binary"])
y = df_clean["Target_Binary"]


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Train model

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# Prediction

y_pred = model.predict(X_test)


# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))


print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
