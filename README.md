# Data Analysis Experiments

## Description of All 7 Experiments

This repository contains heavily standardized and rigorously validated scripts for 7 fundamental Data Science workflows:

1. **Experiment 01: Descriptive Statistics** (`exp1_descriptive_statistics.py`)
   Calculates core statistical foundations including Means, Medians, Outlier Variations, Ranges, and Skewness automatically matching dataset dimensions.
2. **Experiment 02: Data Cleaning** (`exp2_data_cleaning.py`)
   Cleanses bad data dynamically. Injects median figures into null numerical columns, labels textual attributes safely, and mathematically eliminates rows failing IQR boundary tests.
3. **Experiment 03: Inferential Statistics** (`exp3_inferential_statistics.py`)
   Executes rigorous Z-tests and T-tests (One-Sample & Independent), returning exact P-values against hypotheses to determine validation bounds.
4. **Experiment 04: Data Visualization** (`exp4_exploratory_data_analysis.py`)
   Dynamically aggregates dimensions to render vivid visual breakdowns: Boxplots spanning classifications, stacked histograms, correlation heat maps, survival-style bar-charts, complex tree-maps, and high-frequency Word Clouds.
5. **Experiment 05: SMOTE Technique** (`exp5_linear_regression.py`)
   Combats severe class imbalances. The algorithm analyzes categorical distributions naturally isolating the target metric, up-samples the minority rows synthetically (`imbalanced-learn`), bounds them logically, and constructs a robust Logistic predictor.
6. **Experiment 06: Outlier Detection** (`exp6_logistic_regression.py`)
   Drops useless identifiers autonomously, aggressively hunts internal outliers sweeping clean inputs before forging them through predictive Logistic Regression tests tracking absolute precision and F1 recalls.
7. **Experiment 07: Time Series Forecasting** (`exp7_time_series.py`)
   Orchestrates a Deep Learning Long Short-Term Memory Sequence architecture (LSTM) to scale multi-dimensional timelines via Keras mapping RMSE regression graphs plotting predicted paths against reality.

## How the Datasets are Loaded 

Every single script operates using a strictly enforced, fully dynamic load system leveraging `pandas.read_csv()`.

- **Automatic Feature Decoding**: The system checks immediately whether the dataset lacks string headers (such as when raw telemetry data acts as the first line instead of names like "Age").
- **Error Handlers & Fallbacks**: If numerical figures hijack the header list during the `.read_csv()` parsing protocol, a `Warning` trigger catches it natively and instantly re-processes the entire structure enforcing `header=None`. 
- **Validation Blocks**: Prior to any core algorithm activating, scripts rigidly validate and print to the console the new structural framework including exact `.shape`, distinct `.columns`, aggregate `.info()`, and top-level `.head()` rows ensuring no corrupt injections survive.

### 💻 Basic Universal Code
If you want to understand the foundational logic to load any dataset in these experiments, here is the most basic code structure you will need:

```python
import pandas as pd

# 1. Load the dataset
df = pd.read_csv('dataset_name.csv')

# 2. View Dataset Information
print(df.info())

# 3. View the first 5 records
print(df.head())
```

## Instructions to Run Each File

Running standard operations is extremely easy because datasets have been abstracted dynamically. You can either utilize the default parameters built natively into each `.py` file, or feed explicit targets straight down the CLI.

```bash
# General Syntax
python <script_name>.py [optional_dataset.csv]

# 1. Descriptive Statistics
python exp1_descriptive_statistics.py 
# (Optionally test with your own exam data:) -> python exp1_descriptive_statistics.py

# 2. Data Cleaning
python exp2_data_cleaning.py

# 3. Inferential Statistics
python exp3_inferential_statistics.py

# 4. Data Visualization
python exp4_exploratory_data_analysis.py

# 5. SMOTE Technique
python exp5_linear_regression.py

# 6. Outlier Detection
python exp6_logistic_regression.py

# 7. Time Series Forecasting
python exp7_time_series.py 
```

## Required Libraries

Ensure the machine executing the tests operates using the following standard structural modules.

```bash
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn imbalanced-learn tensorflow
```
