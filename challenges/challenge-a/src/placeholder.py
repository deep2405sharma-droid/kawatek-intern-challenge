"""
Kawatek Internship Challenge A: EMG Signal Classification Pipeline
==================================================================
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(filepath):
    df = pd.read_csv(filepath)
    timestamps = df['timestamp'].values
    signals = df[['ch1', 'ch2', 'ch3', 'ch4']].values
    labels = df['label'].values if 'label' in df.columns else None
    return timestamps, signals, labels


def preprocess(signals, fs=1000):
    nyquist = fs / 2
    low = 20 / nyquist
    high = 450 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')

    filtered = np.zeros_like(signals)
    for ch in range(signals.shape[1]):
        filtered[:, ch] = signal.filtfilt(b, a, signals[:, ch])

    normalized = np.zeros_like(filtered)
    for ch in range(signals.shape[1]):
        mean = filtered[:, ch].mean()
        std = filtered[:, ch].std()
        normalized[:, ch] = (filtered[:, ch] - mean) / std

    return normalized


def segment_windows(signals, labels=None, window_size=200, overlap=50):
    step = window_size - overlap
    n_samples = signals.shape[0]

    windows = []
    window_labels = [] if labels is not None else None

    start = 0
    while start + window_size <= n_samples:
        windows.append(signals[start:start + window_size, :])
        if labels is not None:
            window_slice = labels[start:start + window_size]
            values, counts = np.unique(window_slice, return_counts=True)
            window_labels.append(values[np.argmax(counts)])
        start += step

    windows = np.array(windows)
    if labels is not None:
        window_labels = np.array(window_labels)

    return windows, window_labels


def extract_features(windows):
    n_windows = windows.shape[0]
    n_channels = windows.shape[2]
    features = []

    for i in range(n_windows):
        window_features = []
        for ch in range(n_channels):
            x = windows[i, :, ch]
            rms = np.sqrt(np.mean(x**2))
            mav = np.mean(np.abs(x))
            zcr = np.sum(np.diff(np.sign(x)) != 0)
            wl = np.sum(np.abs(np.diff(x)))
            window_features.extend([rms, mav, zcr, wl])
        features.append(window_features)

    return np.array(features)


def train_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


def predict(model, features):
    return model.predict(features)


if __name__ == "__main__":
    print(" Kawatek EMG Classification Pipeline \n")

    # Step 1: Load training data
    print("Loading training data")
    timestamps, signals, labels = load_data("data/emg_signals.csv")

    # Step 2: Preprocess
    print("Preprocessing signals")
    filtered = preprocess(signals, fs=1000)

    # Step 3: Segment into windows
    print("Segmenting into windows")
    windows, window_labels = segment_windows(filtered, labels, window_size=200, overlap=50)

    # Step 4: Extract features
    print("Extracting features")
    features = extract_features(windows)

    # Step 5: Train classifier
    print("Training classifier")
    model, accuracy = train_classifier(features, window_labels)
    print(f"Training accuracy: {accuracy:.2%}")

    # Step 6: Predict on test data
    print("Predicting test data")
    test_timestamps, test_signals, _ = load_data("data/test_signals.csv")
    test_filtered = preprocess(test_signals, fs=1000)
    test_windows, _ = segment_windows(test_filtered, labels=None, window_size=200, overlap=50)
    test_features = extract_features(test_windows)
    predictions = predict(model, test_features)

    # Step 7: Output predictions
    for i, pred in enumerate(predictions):
        print(f"Window {i:03d}: {pred}")
