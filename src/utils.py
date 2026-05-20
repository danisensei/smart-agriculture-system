import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

FEATURE_UNITS = {
    "N": "mg/kg", "P": "mg/kg", "K": "mg/kg",
    "temperature": "°C", "humidity": "%", "ph": "", "rainfall": "mm",
}

FEATURE_RANGES = {
    "N": (0, 140), "P": (5, 145), "K": (5, 205),
    "temperature": (8.0, 44.0), "humidity": (14.0, 100.0),
    "ph": (3.5, 10.0), "rainfall": (20.0, 300.0),
}

FEATURE_DEFAULTS = {
    "N": 50, "P": 55, "K": 50,
    "temperature": 25.0, "humidity": 70.0, "ph": 6.5, "rainfall": 100.0,
}

SOIL_FEATURES = ["N", "P", "K", "ph"]

CROP_TYPES = [
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee",
    "cotton", "grapes", "jute", "kidneybeans", "lentil", "maize",
    "mango", "mothbeans", "mungbean", "muskmelon", "orange",
    "papaya", "pigeonpeas", "pomegranate", "rice", "watermelon",
]

CLUSTER_GUIDANCE = {
    0: {
        "name": "Zone A - High-Nutrient Alluvial Soil",
        "color": "#2ecc71",
        "guidance": (
            "Elevated nitrogen and phosphorus with moderate pH. "
            "Ideal for rice, maize, and jute. "
            "Apply balanced NPK fertilizer and monitor waterlogging."
        ),
    },
    1: {
        "name": "Zone B - Balanced Loamy Soil",
        "color": "#3498db",
        "guidance": (
            "Balanced nutrients with neutral pH. Suitable for wheat, lentils, and chickpeas. "
            "Maintain organic matter and rotate legumes for nitrogen fixation."
        ),
    },
    2: {
        "name": "Zone C - Potassium-Rich Clay Soil",
        "color": "#e67e22",
        "guidance": (
            "High potassium with slightly acidic pH. Good for banana, mango, and grapes. "
            "Supplement phosphorus and improve drainage."
        ),
    },
}


def validate_input(feature_name, value_str):
    if not value_str or value_str.strip() == "":
        return False, f"{FEATURE_LABELS[feature_name]} is required."
    try:
        value = float(value_str.strip())
    except ValueError:
        return False, f"{FEATURE_LABELS[feature_name]}: Enter a valid number."

    lo, hi = FEATURE_RANGES[feature_name]
    if value < lo or value > hi:
        return False, f"{FEATURE_LABELS[feature_name]}: Must be between {lo} and {hi}."
    return True, value


def validate_all_inputs(input_dict):
    errors = []
    values = {}
    for feat in FEATURE_NAMES:
        ok, result = validate_input(feat, input_dict.get(feat, ""))
        if ok:
            values[feat] = result
        else:
            errors.append(result)
    if errors:
        return False, errors
    return True, values


def get_cluster_info(cluster_id):
    if cluster_id in CLUSTER_GUIDANCE:
        return CLUSTER_GUIDANCE[cluster_id]
    return {
        "name": f"Zone {cluster_id} - Uncharacterized Soil",
        "color": "#95a5a6",
        "guidance": "Soil zone not characterized. Consult a local agronomist.",
    }


def calculate_confidence_bounds(residual_std, prediction, n_samples=100):
    margin = 1.96 * residual_std * np.sqrt(1 + 1 / n_samples)
    return float(prediction - margin), float(prediction + margin), float(margin)
