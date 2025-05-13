import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preprocessing import load_and_preprocess
from evaluate import evaluate_model, plot_predictions
from models.lstm_model import train_lstm, predict_lstm
from sklearn.model_selection import train_test_split

# 1. Load and preprocess the data
file_path = 'data/bitcoin_5min_latest.csv'
X, y, df, scaler, timestamps = load_and_preprocess(file_path, n_lags=5)

# 2. Split into train, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, shuffle=False)

# 3. Train the LSTM model
model = train_lstm(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# 4. Predict
predictions = predict_lstm(model, X_test)

# 5. Evaluate the model manually
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"LSTM RMSE: {rmse:.4f}")

# 6. Plot predictions
plot_predictions(y_test, predictions, title="LSTM: Forecast vs Actual")

# Save the rmse for comparison
with open('Results/LSTM_rmse.txt', 'w') as file:
    file.write(str(rmse))

from sklearn.metrics import r2_score

# 7. Calculate R² score
r2 = r2_score(y_test, predictions)
print(f"LSTM R² score: {r2:.4f}")

# 8. Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - predictions) / y_test)*100)
accuracy = 100 - mape
print(f"LSTM Accuracy: {accuracy:.4f}%")

# 9. Save the R² score and Accuracy into the result file
with open('Results/LSTM.txt', 'w') as file:
    file.write(f"RMSE: {rmse:.4f}\n")
    file.write(f"R²: {r2:.4f}\n")
    file.write(f"Accuracy: {accuracy:.4f}%\n")
