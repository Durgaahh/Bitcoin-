import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(filepath, n_lags=5):
    df = pd.read_csv(filepath)  # ✅ don't use parse_dates here
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')  # ✅ correct parsing

    df = df.sort_values('Timestamp')
    df = df[['Timestamp', 'Close']]  # we'll use just Timestamp + Close

    # Create lag features
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)]

    # Normalize features + target
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols + ['Close']] = scaler.fit_transform(df_scaled[feature_cols + ['Close']])

    X = df_scaled[feature_cols].values
    y = df_scaled['Close'].values
    timestamps = df_scaled['Timestamp'].values  # ✅ this was missing before

    return X, y, df_scaled, scaler, timestamps
