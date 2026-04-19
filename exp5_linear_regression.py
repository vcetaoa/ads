# Experiment 05: SMOTE Technique
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import sys

# Step 1: Load Dataset
df = pd.read_csv('fraudTest.csv')

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

# Step 2: Identify Target Column Automatically

#

binary_cols = [col for col in df.columns if df[col].nunique() == 2]
if binary_cols:
    target_col = binary_cols[-1]
else:
    target_col = df.select_dtypes(include=["number"]).columns[-1]
    print(f"No binary column found. Binarizing {target_col} by median.")
    df[target_col] = (df[target_col] > df[target_col].median()).astype(int)


print("Target Column Selected:", target_col)
print("\nOriginal Class Distribution:")
print(df[target_col].value_counts())


#

# Step 3: Encode Categorical  Columns
#
categorical_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()


for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


print("\nCategorical  columns encoded successfully")


#

# Step 4: Visualization  – Before SMOTE
#
plt.figure()

df[target_col].value_counts().plot(kind="bar")
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class (0 = Legitimate,  1 = Fraud)")
plt.ylabel("Count")
plt.show()


#

# Step 5: Split Dataset
#
X = df.drop(target_col, axis=1)
y = df[target_col]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#

# Step 6: Apply SMOTE
#
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


print("\nClass Distribution After SMOTE:")
print(pd.Series(y_train_smote).value_counts())


#

# Step 7: Visualization  – After SMOTE
#
plt.figure()
pd.Series(y_train_smote).value_counts().plot(kind="bar")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


#

# Step 8: Train Model (Fast & Efficient)
#
model = LogisticRegression(max_iter=1000)
model.fit(X_train_smote, y_train_smote)


#

# Step 9: Evaluation


y_pred = model.predict(X_test)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#

# Step 10: Confusion  Matrix Visualization
#
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion  Matrix")
plt.show()
