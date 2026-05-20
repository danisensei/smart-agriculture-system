import argparse
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import MODELS_DIR, RESULTS_DIR, DATA_DIR


def models_exist():
    files = [
        "decision_tree.joblib", "knn_clustering.joblib", "linear_regression.joblib",
        "label_encoder.joblib", "cluster_scaler.joblib", "cluster_pca.joblib",
        "regression_scaler.joblib", "residual_info.joblib",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files)


def run_training():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from src.preprocessing import run_preprocessing
    class_data, cluster_data, reg_data = run_preprocessing()

    from src.models import train_all_models
    train_all_models(class_data, cluster_data, reg_data)


def run_gui():
    from src.gui import launch_gui
    launch_gui()


def main():
    parser = argparse.ArgumentParser(description="Smart Agriculture Decision Support System")
    parser.add_argument("--train", action="store_true", help="Train all models from scratch")
    args = parser.parse_args()

    if args.train:
        run_training()
    else:
        if not models_exist():
            print("Models not found, running training first...")
            run_training()
        run_gui()


if __name__ == "__main__":
    main()
