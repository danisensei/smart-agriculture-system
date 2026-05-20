import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import FEATURE_NAMES, SOIL_FEATURES, DATA_DIR, MODELS_DIR, CROP_TYPES


def generate_crop_data(output_path, n_per_crop=100):
    np.random.seed(42)
    crop_profiles = {
        "rice":        (80, 10, 48, 8, 40, 5, 23, 2, 82, 5, 6.0, 0.5, 230, 30),
        "maize":       (78, 10, 50, 8, 22, 5, 23, 3, 65, 6, 6.2, 0.4, 90, 15),
        "chickpea":    (40, 10, 65, 8, 80, 8, 18, 2, 17, 3, 7.0, 0.4, 75, 10),
        "kidneybeans": (20, 5, 65, 8, 20, 3, 20, 2, 22, 3, 5.7, 0.3, 110, 15),
        "pigeonpeas":  (20, 5, 65, 5, 20, 3, 28, 3, 50, 5, 6.0, 0.5, 145, 15),
        "mothbeans":   (20, 5, 50, 5, 20, 3, 28, 3, 48, 5, 6.8, 0.5, 50, 10),
        "mungbean":    (20, 5, 45, 5, 20, 3, 28, 3, 85, 5, 6.5, 0.4, 50, 10),
        "blackgram":   (40, 5, 60, 5, 20, 3, 30, 3, 65, 5, 7.0, 0.4, 70, 10),
        "lentil":      (20, 5, 65, 5, 20, 3, 25, 3, 55, 5, 6.8, 0.3, 50, 10),
        "pomegranate": (20, 5, 10, 3, 40, 5, 22, 3, 90, 5, 6.5, 0.5, 110, 15),
        "banana":      (100, 10, 75, 8, 50, 8, 27, 2, 80, 5, 6.0, 0.3, 105, 15),
        "mango":       (20, 5, 25, 5, 30, 5, 32, 3, 50, 8, 5.8, 0.5, 95, 15),
        "grapes":      (20, 5, 125, 10, 200, 10, 24, 3, 82, 5, 6.0, 0.4, 70, 10),
        "watermelon":  (100, 10, 15, 3, 50, 5, 26, 2, 85, 5, 6.5, 0.3, 50, 10),
        "muskmelon":   (100, 10, 15, 3, 50, 5, 29, 2, 92, 3, 6.3, 0.3, 25, 5),
        "apple":       (20, 5, 130, 8, 200, 10, 23, 2, 92, 3, 6.0, 0.3, 110, 15),
        "orange":      (20, 5, 15, 5, 10, 3, 23, 3, 92, 3, 7.0, 0.4, 110, 10),
        "papaya":      (50, 10, 60, 8, 50, 8, 34, 3, 92, 3, 6.7, 0.4, 145, 15),
        "coconut":     (20, 5, 10, 3, 30, 5, 27, 2, 95, 3, 6.0, 0.4, 170, 20),
        "cotton":      (120, 10, 45, 8, 20, 5, 24, 3, 80, 5, 7.0, 0.5, 80, 15),
        "jute":        (80, 10, 45, 8, 60, 8, 25, 2, 80, 5, 6.7, 0.3, 175, 20),
        "coffee":      (100, 10, 20, 5, 30, 5, 26, 2, 58, 5, 6.5, 0.4, 160, 20),
    }

    rows = []
    for crop, p in crop_profiles.items():
        n_m, n_s, p_m, p_s, k_m, k_s, t_m, t_s, h_m, h_s, ph_m, ph_s, r_m, r_s = p
        for _ in range(n_per_crop):
            rows.append({
                "N": max(0, np.random.normal(n_m, n_s)),
                "P": max(5, np.random.normal(p_m, p_s)),
                "K": max(5, np.random.normal(k_m, k_s)),
                "temperature": np.clip(np.random.normal(t_m, t_s), 8, 44),
                "humidity": np.clip(np.random.normal(h_m, h_s), 14, 100),
                "ph": np.clip(np.random.normal(ph_m, ph_s), 3.5, 10),
                "rainfall": max(20, np.random.normal(r_m, r_s)),
                "label": crop,
            })

    df = pd.DataFrame(rows)
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    df[numeric_cols] = df[numeric_cols].round(2)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated crop dataset: {len(df)} rows -> {output_path}")
    return df


def generate_yield_data(output_path, n_samples=10000):
    np.random.seed(123)
    data = {
        "N": np.random.uniform(0, 140, n_samples),
        "P": np.random.uniform(5, 145, n_samples),
        "K": np.random.uniform(5, 205, n_samples),
        "temperature": np.random.uniform(10, 42, n_samples),
        "humidity": np.random.uniform(20, 95, n_samples),
        "ph": np.random.uniform(4.0, 9.0, n_samples),
        "rainfall": np.random.uniform(20, 280, n_samples),
    }
    N, P, K = data["N"], data["P"], data["K"]
    temp, hum, ph, rain = data["temperature"], data["humidity"], data["ph"], data["rainfall"]

    nutrient_score = 0.015 * np.sqrt(N) + 0.012 * np.sqrt(P) + 0.008 * np.sqrt(K)
    temp_score = np.exp(-0.5 * ((temp - 25) / 8) ** 2)
    water_score = 0.3 * (hum / 100) + 0.7 * np.minimum(rain / 200, 1.0)
    ph_score = np.exp(-0.5 * ((ph - 6.5) / 1.5) ** 2)

    base_yield = 2.0 + 3.0 * nutrient_score + 2.0 * temp_score + 1.5 * water_score + 1.0 * ph_score
    yield_values = np.clip(base_yield + np.random.normal(0, 0.4, n_samples), 0.5, 10.0)
    data["yield_tons_per_hectare"] = np.round(yield_values, 2)

    df = pd.DataFrame(data).round(2)
    df.to_csv(output_path, index=False)
    print(f"Generated yield dataset: {len(df)} rows -> {output_path}")
    return df


def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {filepath} — shape: {df.shape}")
    return df


def handle_missing_values(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def treat_outliers(df, columns):
    df = df.copy()
    for col in columns:
        if col not in df.columns or df[col].dtype == "object":
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n = ((df[col] < lower) | (df[col] > upper)).sum()
        if n > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"  Capped {n} outliers in {col}")
    return df


def prepare_classification_data(df):
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    joblib.dump(le, le_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Classification: {X_train.shape[0]} train, {X_test.shape[0]} test, {len(le.classes_)} classes")
    return X_train, X_test, y_train, y_test, le, FEATURE_NAMES


def prepare_clustering_data(df):
    X = df[SOIL_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_path = os.path.join(MODELS_DIR, "cluster_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    print(f"Clustering: {X.shape[0]} samples, features: {SOIL_FEATURES}")
    return X_scaled, scaler, SOIL_FEATURES, X


def prepare_regression_data(df):
    X = df[FEATURE_NAMES].values
    y = df["yield_tons_per_hectare"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_path = os.path.join(MODELS_DIR, "regression_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Regression: {X_train.shape[0]} train, {X_test.shape[0]} test, yield range [{y.min():.2f}, {y.max():.2f}]")
    return X_train, X_test, y_train, y_test, scaler, FEATURE_NAMES


def run_preprocessing():
    os.makedirs(MODELS_DIR, exist_ok=True)

    crop_path = os.path.join(DATA_DIR, "crop_recommendation.csv")
    if not os.path.exists(crop_path):
        generate_crop_data(crop_path)
    crop_df = load_data(crop_path)
    crop_df = handle_missing_values(crop_df)
    crop_df = treat_outliers(crop_df, FEATURE_NAMES)

    yield_path = os.path.join(DATA_DIR, "crop_yield.csv")
    if not os.path.exists(yield_path):
        generate_yield_data(yield_path)
    yield_df = load_data(yield_path)
    yield_df = handle_missing_values(yield_df)
    yield_df = treat_outliers(yield_df, FEATURE_NAMES + ["yield_tons_per_hectare"])

    class_data = prepare_classification_data(crop_df)
    cluster_data = prepare_clustering_data(crop_df)
    reg_data = prepare_regression_data(yield_df)

    print("Preprocessing done.")
    return class_data, cluster_data, reg_data


if __name__ == "__main__":
    run_preprocessing()
