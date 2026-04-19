# Experiment 04: Data Visualization
# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import sys

# Load Dataset
data = pd.read_csv('Titanic-Dataset.csv')

print("\n--- Dataset Validation ---")
print("Dataset loaded successfully.")
print(f"Shape: {data.shape}")
print("\nColumns:", data.columns.tolist())
print("\nDataset Info:")
data.info()
print("\nFirst 5 Rows:")
print(data.head())
print("--------------------------\n")

num_cols = data.select_dtypes(include="number").columns
cat_cols = data.select_dtypes(exclude=["number", "datetime"]).columns

target_num1 = num_cols[0] if len(num_cols) > 0 else None
target_num2 = num_cols[1] if len(num_cols) > 1 else target_num1
target_cat1 = cat_cols[0] if len(cat_cols) > 0 else None
target_cat2 = cat_cols[1] if len(cat_cols) > 1 else target_cat1

#

# 1. BOX PLOT – Dynamic
#

plt.figure(figsize=(8, 6))
if target_cat1 and target_num1:
    sns.boxplot(x=target_cat1, y=target_num1, data=data)
    plt.title(f"Box Plot of {target_num1} by {target_cat1}")
    plt.xlabel(target_cat1)
    plt.ylabel(target_num1)
elif target_num1:
    sns.boxplot(y=target_num1, data=data)
    plt.title(f"Box Plot of {target_num1}")
plt.tight_layout()
plt.show()


#

# 2. HISTOGRAM  – Dynamic
#
plt.figure(figsize=(8, 6))

if target_num1:
    if target_cat1 and data[target_cat1].nunique() <= 10:
        sns.histplot(data=data, x=target_num1, hue=target_cat1, kde=True, multiple="stack")
    else:
        sns.histplot(data=data, x=target_num1, kde=True)
    plt.title(f"Histogram  of {target_num1} Distribution")
    plt.xlabel(target_num1)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

#

# 3. HEAT MAP – Correlation  Matrix
#
plt.figure(figsize=(10, 8))

corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Heat Map of Correlation Matrix")
plt.tight_layout()
plt.show()


#

# 4. CHARTS – Bar Chart (Dynamic)
#
plt.figure(figsize=(6, 4))
if target_cat1:
    sns.countplot(x=target_cat1, data=data)
    plt.title(f"Bar Chart of {target_cat1} Count")
    plt.xlabel(target_cat1)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()


#

# 5. TREE MAP – Dynamic
#

if target_cat1 and target_cat2:
    fig = px.treemap(
        data,
        path=[target_cat1, target_cat2],  # Hierarchy
        title=f"Tree Map of {target_cat1} and {target_cat2}",
    )
    fig.update_traces(textinfo="label+percent entry")  # Shows names + percentage
    fig.show()
else:
    print("Not enough categorical columns for Treemap")

#

# 6. WORD CLOUD – Dynamic Text Column
#

text_col = None
for col in cat_cols:
    if data[col].nunique() > 10:
        text_col = col
        break
if not text_col and len(cat_cols) > 0:
    text_col = cat_cols[-1]

if text_col:
    plt.figure(figsize=(8, 6))
    text_data = data[text_col].dropna().astype(str)
    
    wc = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(text_data)
    )
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"Word Cloud of {text_col}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
