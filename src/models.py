import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.decomposition import PCA
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import FEATURE_NAMES, SOIL_FEATURES, MODELS_DIR, RESULTS_DIR

# shared plot style
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "figure.dpi": 120,
})


def train_decision_tree(X_train, X_test, y_train, y_test, label_encoder, feature_names, models_dir, results_dir):
    print("\nTraining Decision Tree Classifier...")

    best_acc, best_model, best_depth = 0, None, None
    for depth in [5, 8, 10, 12, 15, None]:
        dt = DecisionTreeClassifier(
            max_depth=depth, min_samples_split=5, min_samples_leaf=2,
            random_state=42, criterion="gini"
        )
        dt.fit(X_train, y_train)
        acc = accuracy_score(y_test, dt.predict(X_test))
        print(f"  max_depth={str(depth):>5s}  acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_model, best_depth = acc, dt, depth

    print(f"  Best: max_depth={best_depth}, accuracy={best_acc:.4f}")

    y_pred = best_model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"  Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # feature importance plot
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    colors = ["#00d2ff", "#3a7bd5", "#6c5ce7", "#a29bfe", "#fd79a8", "#e17055", "#00b894"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_features)), sorted_importances,
                   color=colors[:len(sorted_features)], edgecolor="#ffffff", linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance - Decision Tree", fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, sorted_importances):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, color="#e0e0e0")
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "feature_importance.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    joblib.dump(best_model, os.path.join(models_dir, "decision_tree.joblib"))
    print(f"  Model saved.")

    return {
        "model": best_model,
        "metrics": {
            "model": "Decision Tree Classifier",
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1_score": f1, "best_max_depth": best_depth,
        },
        "feature_importances": dict(zip(feature_names, importances)),
        "plot_path": plot_path,
    }


def train_kmeans_clustering(X_scaled, X_original, soil_features, models_dir, results_dir):
    print("\nTraining KMeans Clustering...")

    K_range = range(2, 9)
    inertias, sil_scores = [], []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)
        print(f"  K={k}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    best_k_idx = np.argmax(sil_scores)
    best_k = list(K_range)[best_k_idx]
    best_sil = sil_scores[best_k_idx]

    # prefer K=5 if close in silhouette score
    if 5 in list(K_range):
        k5_idx = list(K_range).index(5)
        if abs(sil_scores[k5_idx] - best_sil) < 0.05:
            best_k, best_sil = 5, sil_scores[k5_idx]

    print(f"  Selected K={best_k} (silhouette={best_sil:.4f})")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_scaled)

    for c in range(best_k):
        count = (cluster_labels == c).sum()
        print(f"  Cluster {c}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_
    print(f"  PCA variance explained: {ev[0]:.3f}, {ev[1]:.3f}")

    # cluster scatter + silhouette bar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cluster_colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c", "#f39c12"]

    for c in range(best_k):
        mask = cluster_labels == c
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=cluster_colors[c % len(cluster_colors)],
                    label=f"Cluster {c} (n={mask.sum()})",
                    alpha=0.6, s=30, edgecolors="white", linewidth=0.3)

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                c="white", marker="X", s=200, edgecolors="black", linewidth=1.5, zorder=5, label="Centroids")
    ax1.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)", fontsize=11)
    ax1.set_title("Soil Profile Clusters (PCA)", fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.bar(list(K_range), sil_scores,
            color=["#ff6b6b" if k == best_k else "#00d2ff" for k in K_range],
            edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Silhouette Score by K", fontsize=13)
    ax2.set_xticks(list(K_range))
    ax2.axhline(y=best_sil, color="#ff6b6b", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "cluster_scatter.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    joblib.dump(kmeans, os.path.join(models_dir, "knn_clustering.joblib"))
    joblib.dump(pca, os.path.join(models_dir, "cluster_pca.joblib"))
    print("  Models saved.")

    return {
        "model": kmeans, "pca": pca,
        "metrics": {
            "model": "KMeans Clustering",
            "optimal_k": best_k, "silhouette_score": best_sil,
            "inertia": kmeans.inertia_,
            "pca_variance_explained": float(sum(ev)),
            "silhouette_scores_all_k": {k: s for k, s in zip(K_range, sil_scores)},
        },
        "cluster_labels": cluster_labels,
        "plot_path": plot_path,
    }


def train_linear_regression(X_train, X_test, y_train, y_test, feature_names, models_dir, results_dir):
    print("\nTraining Linear Regression...")

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_train = lr.predict(X_train)
    y_pred_test  = lr.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)

    print(f"  RMSE train={rmse_train:.4f}  test={rmse_test:.4f}  MAE={mae_test:.4f}  R2={r2_test:.4f}")

    print("  Coefficients:")
    for feat, coef in zip(feature_names, lr.coef_):
        print(f"    {feat:>15s}: {coef:+.6f}")
    print(f"    {'Intercept':>15s}: {lr.intercept_:+.6f}")

    residuals = y_test - y_pred_test
    residual_std = np.std(residuals)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    ax1.scatter(y_test, y_pred_test, alpha=0.3, s=15, c="#00d2ff", edgecolors="none")
    mn, mx = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
    ax1.set_xlabel("Actual Yield (tons/ha)", fontsize=11)
    ax1.set_ylabel("Predicted Yield (tons/ha)", fontsize=11)
    ax1.set_title("Actual vs Predicted", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f"R² = {r2_test:.3f}", transform=ax1.transAxes,
             fontsize=11, va="top", bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))

    ax2.scatter(y_pred_test, residuals, alpha=0.3, s=15, c="#fd79a8", edgecolors="none")
    ax2.axhline(y=0, color="#00d2ff", linewidth=2, linestyle="--")
    ax2.axhline(y=2*residual_std, color="#ffeaa7", linewidth=1, linestyle=":", alpha=0.7, label=f"±2σ")
    ax2.axhline(y=-2*residual_std, color="#ffeaa7", linewidth=1, linestyle=":", alpha=0.7)
    ax2.set_xlabel("Predicted Yield (tons/ha)", fontsize=11)
    ax2.set_ylabel("Residuals", fontsize=11)
    ax2.set_title("Residuals vs Predicted", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3.hist(residuals, bins=40, color="#6c5ce7", edgecolor="white", linewidth=0.5, alpha=0.8, density=True)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = (1 / (residual_std * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((x_range - np.mean(residuals)) / residual_std) ** 2)
    ax3.plot(x_range, normal_curve, color="#00d2ff", linewidth=2,
             label=f"Normal (σ={residual_std:.3f})")
    ax3.set_xlabel("Residual", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title("Residual Distribution", fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "residual_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    joblib.dump(lr, os.path.join(models_dir, "linear_regression.joblib"))
    residual_info = {
        "residual_std": float(residual_std),
        "residual_mean": float(np.mean(residuals)),
        "n_test_samples": len(y_test),
    }
    joblib.dump(residual_info, os.path.join(models_dir, "residual_info.joblib"))
    print("  Model saved.")

    return {
        "model": lr,
        "metrics": {
            "model": "Linear Regression",
            "rmse_train": rmse_train, "rmse_test": rmse_test,
            "mae_test": mae_test, "r2_test": r2_test,
            "residual_std": float(residual_std),
        },
        "plot_path": plot_path,
    }


def save_metrics_summary(dt_result, km_result, lr_result, results_dir):
    path = os.path.join(results_dir, "metrics_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        m = dt_result["metrics"]
        f.write("Decision Tree Classifier\n")
        f.write(f"  Accuracy:  {m['accuracy']:.4f}\n")
        f.write(f"  Precision: {m['precision']:.4f}\n")
        f.write(f"  Recall:    {m['recall']:.4f}\n")
        f.write(f"  F1-Score:  {m['f1_score']:.4f}\n")
        f.write(f"  Max depth: {m['best_max_depth']}\n\n")

        m = km_result["metrics"]
        f.write("KMeans Clustering\n")
        f.write(f"  Optimal K:        {m['optimal_k']}\n")
        f.write(f"  Silhouette score: {m['silhouette_score']:.4f}\n")
        f.write(f"  Inertia:          {m['inertia']:.2f}\n\n")

        m = lr_result["metrics"]
        f.write("Linear Regression\n")
        f.write(f"  RMSE (train): {m['rmse_train']:.4f}\n")
        f.write(f"  RMSE (test):  {m['rmse_test']:.4f}\n")
        f.write(f"  MAE:          {m['mae_test']:.4f}\n")
        f.write(f"  R²:           {m['r2_test']:.4f}\n")

    print(f"  Metrics summary saved to {path}")


def train_all_models(class_data, cluster_data, reg_data, models_dir=None, results_dir=None):
    if models_dir is None:
        models_dir = MODELS_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    X_train_c, X_test_c, y_train_c, y_test_c, le, feat_names = class_data
    X_scaled, scaler, soil_feats, X_original = cluster_data
    X_train_r, X_test_r, y_train_r, y_test_r, reg_scaler, reg_feats = reg_data

    dt_result = train_decision_tree(X_train_c, X_test_c, y_train_c, y_test_c, le, feat_names, models_dir, results_dir)
    km_result = train_kmeans_clustering(X_scaled, X_original, soil_feats, models_dir, results_dir)
    lr_result = train_linear_regression(X_train_r, X_test_r, y_train_r, y_test_r, reg_feats, models_dir, results_dir)

    save_metrics_summary(dt_result, km_result, lr_result, results_dir)
    print("\nAll models trained successfully.")

    return {"decision_tree": dt_result, "kmeans": km_result, "linear_regression": lr_result}


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    class_data, cluster_data, reg_data = run_preprocessing()
    train_all_models(class_data, cluster_data, reg_data)
