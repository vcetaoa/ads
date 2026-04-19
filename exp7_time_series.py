# Experiment 07: Time Series Forecasting
# Import required libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load Dataset
df = pd.read_csv('covid_19.csv')

print("\n--- Dataset Validation ---")
print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nDataset Info:")
df.info()
print("\nFirst 5 Rows:")
print(df.head())
print("--------------------------\n")

# Automatically select the last numeric column for forecasting
num_cols = df.select_dtypes(include='number').columns
if len(num_cols) > 0:
    target_col = num_cols[-1]
else:
    raise ValueError("No numeric columns available to forecast")

data = df[[target_col]].values.astype('float32')

# Step 2: Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Split into train and test sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

# Helper function to create sequences
def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Step 4: Build LSTM Model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Step 5: Train Model
print("\nTraining LSTM Model...")
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

# Step 6: Make Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform([y_test])

# Step 7: Calculate RMSE
train_score = np.sqrt(mean_squared_error(y_train_inv[0], train_predict[:,0]))
print(f'\nTrain RMSE: {train_score:.2f}')
test_score = np.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))
print(f'Test RMSE: {test_score:.2f}')

# Step 8: Visualization
# shift train predictions for plotting
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(look_back*2)+1:len(data)-1, :] = test_predict

plt.figure(figsize=(10,6))
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Data')
plt.plot(train_predict_plot, label='Training Prediction')
plt.plot(test_predict_plot, label='Testing Prediction')
plt.title('Time Series Forecasting using LSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

print("\nTime Series Analysis Completed")
