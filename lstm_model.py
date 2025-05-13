from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


def reshape_for_lstm(X):
    return np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)


def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    X_train_lstm = reshape_for_lstm(X_train)
    X_val_lstm = reshape_for_lstm(X_val)

    model = build_lstm(input_shape=(X_train.shape[1], 1))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_lstm, y_train, validation_data=(X_val_lstm, y_val),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    return model


def predict_lstm(model, X_test):
    X_test_lstm = reshape_for_lstm(X_test)
    return model.predict(X_test_lstm).flatten()
