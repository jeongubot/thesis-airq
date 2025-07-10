import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Define possible features (use only those present in the data)
POSSIBLE_FEATURES = ['pm2.5', 'pm10', 'temperature', 'humidity']
TARGET_COL = 'pm2.5'
TIMESTEPS = 10

def get_available_features(df):
    # Use only features present in the dataframe and not the target
    return [col for col in POSSIBLE_FEATURES if col in df.columns and col != TARGET_COL]

def load_and_scale(csv_path, scaler=None, fit=False, feature_cols=None):
    df = pd.read_csv(csv_path)
    # Drop rows with missing values in selected features or target
    df = df.dropna(subset=feature_cols + [TARGET_COL])
    features = df[feature_cols].values
    if scaler is None:
        scaler = MinMaxScaler()
    if fit:
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    return features_scaled, df[TARGET_COL].values, scaler

def create_sequences(data, targets, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(targets[i+timesteps])
    return np.array(X), np.array(y)

def preprocess_day(train_csv, val_csv, test_csv, timesteps=TIMESTEPS, out_prefix=''):
    # Load all splits to determine common features
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    # Find intersection of available features across splits
    features_train = set(get_available_features(train_df))
    features_val = set(get_available_features(val_df))
    features_test = set(get_available_features(test_df))
    feature_cols = list(features_train & features_val & features_test)
    feature_cols.sort()  # for consistency

    if not feature_cols:
        raise ValueError("No common features found across splits!")

    print(f"Using features for {out_prefix}: {feature_cols}")

    # Preprocess each split
    X_train_raw, y_train, scaler = load_and_scale(train_csv, fit=True, feature_cols=feature_cols)
    X_val_raw, y_val, _ = load_and_scale(val_csv, scaler, feature_cols=feature_cols)
    X_test_raw, y_test, _ = load_and_scale(test_csv, scaler, feature_cols=feature_cols)

    X_train, y_train = create_sequences(X_train_raw, y_train, timesteps)
    X_val, y_val = create_sequences(X_val_raw, y_val, timesteps)
    X_test, y_test = create_sequences(X_test_raw, y_test, timesteps)

    # Save arrays and feature info
    np.savez(f'{out_prefix}lstm_preprocessed_data.npz',
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             feature_cols=feature_cols)
    print(f"Saved: {out_prefix}lstm_preprocessed_data.npz")

if __name__ == "__main__":
    # Example for 11_10_data (full features)
    preprocess_day(
        train_csv='dataset/d_data_split/11_10_data/train.csv',
        val_csv='dataset/d_data_split/11_10_data/val.csv',
        test_csv='dataset/d_data_split/11_10_data/test.csv',
        out_prefix='11_10_'
    )
    # Example for 7_24_data (fewer features)
    preprocess_day(
        train_csv='dataset/d_data_split/7_24_data/train.csv',
        val_csv='dataset/d_data_split/7_24_data/val.csv',
        test_csv='dataset/d_data_split/7_24_data/test.csv',
        out_prefix='7_24_'
    )
    # Example for 10_19_data (handle missing location)
    preprocess_day(
        train_csv='dataset/d_data_split/10_19_data/train.csv',
        val_csv='dataset/d_data_split/10_19_data/val.csv',
        test_csv='dataset/d_data_split/10_19_data/test.csv',
        out_prefix='10_19_'
    )