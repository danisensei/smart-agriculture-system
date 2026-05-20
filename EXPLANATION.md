# Smart Agriculture Decision Support System — Full Explanation

## What This Project Does

This is an AI-powered decision support tool for farmers and agronomists. You enter
measurements about your soil and local climate, and the system runs three machine
learning models back-to-back to answer three questions:

1. **What crop should I plant?** (Decision Tree Classifier)
2. **What soil zone/type do I have?** (KMeans Clustering)
3. **How much yield can I expect?** (Linear Regression)

Everything runs locally — no internet required. The models are trained once, saved to
disk, and then loaded by the GUI for fast inference.

---

## Project Structure

```
smart-agriculture-system/
│
├── main.py                  # Entry point — runs training or launches GUI
│
├── src/
│   ├── utils.py             # Constants, feature definitions, validation helpers
│   ├── preprocessing.py     # Data loading, cleaning, and preparation
│   ├── models.py            # Training logic for all three models
│   └── gui.py               # Tkinter GUI — the user-facing application
│
├── data/                    # CSV datasets (auto-generated if missing)
│   ├── crop_recommendation.csv
│   └── crop_yield.csv
│
├── models/                  # Serialized trained models (saved by joblib)
│   ├── decision_tree.joblib
│   ├── label_encoder.joblib
│   ├── knn_clustering.joblib
│   ├── cluster_scaler.joblib
│   ├── cluster_pca.joblib
│   ├── linear_regression.joblib
│   ├── regression_scaler.joblib
│   └── residual_info.joblib
│
└── results/                 # Evaluation plots saved as PNG
    ├── feature_importance.png
    ├── cluster_scatter.png
    ├── residual_plot.png
    └── metrics_summary.txt
```

---

## Input Features

All three models share the same 7 input features:

| Feature     | Description                    | Unit   | Valid Range  |
|-------------|--------------------------------|--------|--------------|
| N           | Nitrogen content in soil       | mg/kg  | 0 – 140      |
| P           | Phosphorus content in soil     | mg/kg  | 5 – 145      |
| K           | Potassium content in soil      | mg/kg  | 5 – 205      |
| temperature | Ambient temperature            | °C     | 8 – 44       |
| humidity    | Relative humidity              | %      | 14 – 100     |
| ph          | Soil pH                        | —      | 3.5 – 10.0   |
| rainfall    | Annual rainfall                | mm     | 20 – 300     |

These are the same features used in the popular **Kaggle Crop Recommendation Dataset**
by Atharva Ingle. Since we don't bundle that dataset, the system auto-generates
synthetic data using agronomic profiles based on FAO crop production guidelines.

---

## Step 1 — Data Generation (src/preprocessing.py)

### Why synthetic data?

Real agricultural datasets require licensing. The synthetic data is generated using
known crop profiles: each crop has a characteristic mean and standard deviation for
each feature (e.g., rice needs high humidity ~82% and high rainfall ~230mm).
This means the data is realistic and the trained models actually work.

### Crop Recommendation Dataset

- **22 crop types**: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans,
  mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon,
  muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee
- **2200 samples** (100 per crop)
- Each sample is drawn from a Normal distribution centered on that crop's profile
- Values are clipped to realistic physical bounds (e.g., humidity can't exceed 100%)

### Yield Dataset

- **10,000 samples** with random features across the valid ranges
- The yield target (`yield_tons_per_hectare`) is computed from a formula:

```
nutrient_score = 0.015*sqrt(N) + 0.012*sqrt(P) + 0.008*sqrt(K)
temp_score     = exp(-0.5 * ((temp - 25) / 8)^2)       # bell curve, optimal at 25°C
water_score    = 0.3*(humidity/100) + 0.7*min(rain/200, 1.0)
ph_score       = exp(-0.5 * ((ph - 6.5) / 1.5)^2)      # optimal pH ~6.5

yield = 2.0 + 3.0*nutrient_score + 2.0*temp_score + 1.5*water_score + 1.0*ph_score
      + Normal(0, 0.4)   # random noise
```

This gives yields in roughly the 3.5 – 8 tons/ha range, clipped to [0.5, 10].

---

## Step 2 — Preprocessing (src/preprocessing.py)

Before training, each dataset goes through a cleaning pipeline:

### Missing Value Imputation

```
handle_missing_values(df)
```

- **Numeric columns** → filled with the column **median** (robust to outliers)
- **Categorical columns** → filled with the column **mode** (most frequent value)

In practice the synthetic data has no missing values, but this makes the pipeline
robust if a real dataset is substituted.

### Outlier Treatment — IQR Capping (Winsorization)

```
treat_outliers(df, columns)
```

Uses the **Interquartile Range (IQR)** method:

```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

Any value below `lower_bound` is set to `lower_bound`, and any value above
`upper_bound` is set to `upper_bound`. This is called **capping** or **winsorization**.
It keeps all rows (unlike dropping) but removes the distorting effect of extreme values.

### Feature Scaling — StandardScaler

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

StandardScaler transforms each feature to have **mean = 0** and **standard deviation = 1**:

```
X_scaled = (X - mean) / std
```

This is critical for KMeans (which uses Euclidean distance — unscaled features with
larger ranges would dominate) and for Linear Regression (makes coefficients comparable).
The Decision Tree does NOT need scaling because it splits on thresholds, not distances.

The scaler is `fit` on training data only, then `transform` is applied to both train
and test sets. The fitted scaler is saved to disk so the GUI can scale new inputs
the same way.

### Label Encoding

```python
le = LabelEncoder()
y_encoded = le.fit_transform(y)   # "rice" -> 21, "apple" -> 0, etc.
```

The Decision Tree requires numeric class labels. `LabelEncoder` maps each crop name
to an integer alphabetically. The encoder is saved so predictions can be decoded back
to crop names (e.g., 21 → "rice").

### Train/Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_encoded)
```

- **80% training, 20% testing**
- `stratify=y_encoded` ensures every crop class is proportionally represented in
  both train and test sets (otherwise a small class might end up entirely in train)
- `random_state=42` makes it reproducible

---

## Model 1 — Decision Tree Classifier (Crop Recommendation)

### What is a Decision Tree?

A Decision Tree learns a hierarchy of yes/no questions about the features.
At each internal node it picks the feature and threshold that best splits the data.
At each leaf node it assigns a class label.

Example path for rice prediction:
```
rainfall > 180?
  YES → humidity > 75?
         YES → N > 60? → YES → RICE
```

### How it splits (Gini Impurity)

We use `criterion="gini"`. The Gini impurity of a node measures how mixed its
classes are:

```
Gini = 1 - sum(p_i^2)
```

where `p_i` is the proportion of class `i` in the node. A pure node (all one class)
has Gini = 0. The algorithm tries every possible split and picks the one that
minimizes the weighted Gini of the two child nodes.

### Hyperparameter Tuning (manual grid search)

We train 6 trees with different `max_depth` values and pick the best by test accuracy:

```python
for depth in [5, 8, 10, 12, 15, None]:
    dt = DecisionTreeClassifier(max_depth=depth, ...)
    dt.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt.predict(X_test))
```

Other fixed hyperparameters:
- `min_samples_split=5` — a node must have at least 5 samples to be split
- `min_samples_leaf=2` — each leaf must have at least 2 samples
- These prevent overfitting to noise in the training data

### Results

The best model typically achieves **~96% accuracy** on the test set with `max_depth=12`.

### Evaluation Metrics

| Metric    | Formula                        | What it measures                    |
|-----------|--------------------------------|-------------------------------------|
| Accuracy  | correct / total                | Overall fraction correct            |
| Precision | TP / (TP + FP)                 | Of predicted class X, how many right|
| Recall    | TP / (TP + FN)                 | Of actual class X, how many caught  |
| F1-Score  | 2 * P * R / (P + R)            | Harmonic mean of precision & recall |

We use **weighted averaging** for multi-class (weighted by class support size).

### Feature Importance

After training, `dt.feature_importances_` gives each feature a score indicating
how much it reduced Gini impurity across all splits. Typically:
- **rainfall** is most important (~32%) — separates water-loving crops (rice, coconut)
  from drought-resistant ones (mothbeans, mungbean)
- **P, K, N** follow — soil nutrients differentiate crop families
- **ph** is least important (~0.7%) — most crops tolerate a wide pH range

---

## Model 2 — KMeans Clustering (Soil Zone Classification)

### What is KMeans?

KMeans is an **unsupervised** algorithm — it finds groups (clusters) in data without
using any labels. It's used here to segment the soil into zones based on nutrient
profile (N, P, K, pH only — climate features are excluded).

The algorithm:
1. Randomly initialize K cluster centroids
2. Assign each sample to its nearest centroid (Euclidean distance)
3. Recompute each centroid as the mean of its assigned samples
4. Repeat steps 2–3 until assignments don't change (convergence)

### Choosing the right K — Silhouette Score

We evaluate K from 2 to 8 using the **silhouette score**:

```
silhouette(i) = (b - a) / max(a, b)

where:
  a = mean distance from point i to all other points in its cluster
  b = mean distance from point i to all points in the nearest other cluster
```

Score ranges from -1 to 1:
- **+1** → point is well inside its own cluster, far from others (good)
- **0** → point is on the boundary between two clusters
- **-1** → point is probably in the wrong cluster (bad)

The K with the highest average silhouette score is selected.

### Why K=2 is selected

With synthetic data generated from 22 known crop profiles, the soil nutrient features
naturally form 2 broad zones: high-nutrient soils (rice, banana, maize) vs. the rest.
A higher K would capture finer differences but at the cost of silhouette score.

### PCA for Visualization

Since the clustering happens in 4D space (N, P, K, pH), we can't directly visualize
it. **Principal Component Analysis (PCA)** projects the 4D data onto 2D while
preserving as much variance as possible.

```
PCA finds the directions (principal components) of maximum variance.
PC1 explains ~41.7% of variance, PC2 explains ~25.7% → 67.4% total captured.
```

The 2D scatter plot in the GUI shows the clusters in this PCA projection. The PCA
transformer is also saved so new user inputs can be projected into the same space.

---

## Model 3 — Linear Regression (Yield Prediction)

### What is Linear Regression?

Linear Regression fits a straight-line (hyperplane in multiple dimensions) through
the data by finding coefficients that minimize the sum of squared errors:

```
yield = b0 + b1*N + b2*P + b3*K + b4*temperature + b5*humidity + b6*ph + b7*rainfall
```

The **Ordinary Least Squares (OLS)** solution finds the exact optimal coefficients
analytically (no iteration needed):

```
β = (X^T X)^{-1} X^T y
```

### Interpreting the Coefficients

After training on scaled features, the coefficients indicate direction and relative
magnitude of each feature's effect on yield:

- **rainfall: +0.30** — strongest positive effect (water availability drives yield)
- **temperature: -0.10** — negative (high temperatures above 25°C hurt yield)
- **N: +0.12, P: +0.08, K: +0.08** — all nutrients help, nitrogen matters most
- **ph: +0.004** — very small effect (yield is relatively insensitive to pH)

### Why R² is low (~0.20)

An R² of 0.20 means the linear model explains only 20% of variance in yield. This is
expected because:
1. The yield formula used to generate the data involves nonlinear terms (sqrt, exp)
2. Linear Regression can't capture these nonlinearities
3. Random noise was added (std=0.4 tons/ha)

This is intentional — it demonstrates Linear Regression's real-world limitations
when the true relationship is nonlinear. The model still gives a useful ballpark
prediction with a confidence interval.

### Confidence Interval

The GUI displays a **95% prediction interval** around the yield estimate:

```
margin = 1.96 * residual_std * sqrt(1 + 1/n)

lower = prediction - margin
upper = prediction + margin
```

- `1.96` is the z-score for 95% confidence
- `residual_std` is the standard deviation of residuals on the test set
- The `sqrt(1 + 1/n)` term accounts for uncertainty in the mean estimate

### Evaluation Metrics

| Metric          | Formula                          | Meaning                              |
|-----------------|----------------------------------|--------------------------------------|
| RMSE            | sqrt(mean((y - y_pred)^2))       | Average error in same units as yield |
| MAE             | mean(abs(y - y_pred))            | Average absolute error               |
| R² (R-squared)  | 1 - SS_res/SS_tot                | Fraction of variance explained       |

---

## The Residual Plots (results/residual_plot.png)

Three subplots are generated after training Linear Regression:

**1. Actual vs Predicted**
Plots true yield on X-axis vs predicted yield on Y-axis. A perfect model would have
all points on the diagonal red line. Scatter around it shows prediction error.

**2. Residuals vs Predicted**
Plots residuals (actual − predicted) on Y-axis vs predicted values on X-axis.
A good model shows residuals randomly scattered around 0 with no pattern.
A funnel shape (heteroscedasticity) would indicate the model's errors are
larger for certain prediction ranges.

**3. Residual Distribution**
Histogram of residuals with a fitted Normal curve overlaid. Linear Regression
assumes residuals are normally distributed. If the histogram roughly matches
the curve, the assumption holds.

---

## How the GUI Works (src/gui.py)

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Header (title + status bar)                            │
├──────────────┬──────────────────────────────────────────┤
│  Input Panel │  Result Cards (Crop | Cluster | Yield)   │
│  (7 fields)  ├──────────────────────────────────────────┤
│  [Analyze]   │  Tabs: Feature Importance | Clusters |   │
│  [Reset]     │        Residual Analysis                  │
│  [Defaults]  │                                          │
└──────────────┴──────────────────────────────────────────┘
```

### Inference Flow (when you click "Analyze & Predict")

```
User inputs (7 numbers as strings)
    ↓
validate_all_inputs()        # type check + range check
    ↓
feature_vector = [N, P, K, temp, humidity, ph, rainfall]   # shape (1, 7)
    ↓
┌─────────────────────────────────────────────────────────┐
│ Decision Tree                                           │
│   dt_model.predict(feature_vector)    → crop index      │
│   label_encoder.inverse_transform()   → crop name       │
│   dt_model.predict_proba()            → confidence %    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ KMeans Clustering                                       │
│   soil_vals = [N, P, K, ph]   (4 features only)        │
│   cluster_scaler.transform()  → scaled 4D vector        │
│   km_model.predict()          → cluster ID (0 or 1)     │
│   get_cluster_info()          → name + color + guidance  │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Linear Regression                                       │
│   regression_scaler.transform(feature_vector)           │
│   lr_model.predict()          → yield (tons/ha)         │
│   calculate_confidence_bounds() → [lower, upper]        │
└─────────────────────────────────────────────────────────┘
    ↓
Update result cards + refresh cluster plot tab
```

### Important: Two separate scalers

The clustering scaler was fit on 4 features (N, P, K, pH).
The regression scaler was fit on all 7 features.
They are kept separate because combining them would mix distributions fitted
on different data shapes and produce incorrect scaled values.

---

## How Models Are Saved and Loaded (joblib)

`joblib.dump(model, path)` serializes any Python object to a binary file.
`joblib.load(path)` deserializes it back.

This means training happens once (`python main.py --train`) and the GUI just loads
the pre-trained objects — no re-training on every launch.

Saved objects:
- `decision_tree.joblib` — the trained `DecisionTreeClassifier` object
- `label_encoder.joblib` — the fitted `LabelEncoder` (crop name ↔ integer)
- `knn_clustering.joblib` — the trained `KMeans` object
- `cluster_scaler.joblib` — the fitted `StandardScaler` for 4D soil features
- `cluster_pca.joblib` — the fitted `PCA` transformer (for visualization)
- `linear_regression.joblib` — the trained `LinearRegression` object
- `regression_scaler.joblib` — the fitted `StandardScaler` for all 7 features
- `residual_info.joblib` — dict with `residual_std` and `n_test_samples`

---

## Libraries Used

| Library      | Purpose                                                      |
|--------------|--------------------------------------------------------------|
| numpy        | Numerical arrays, math operations                            |
| pandas       | CSV loading and DataFrame manipulation                       |
| scikit-learn | All ML models, scalers, encoders, metrics, train/test split  |
| joblib       | Saving and loading trained model objects                     |
| matplotlib   | Generating training evaluation plots                         |
| tkinter      | Native Python GUI framework (included with Python)           |

---

## Running the Project

```bash
# First run — generates data, trains models, saves everything, then opens GUI
python main.py

# Force re-training (even if models already exist)
python main.py --train

# After training is done, just open the GUI
python main.py
```

---

## Known Limitations

1. **Synthetic data** — the models are trained on generated data, not real field
   measurements. Predictions are illustrative, not agronomically certified.

2. **Linear Regression for yield** — yield depends on many nonlinear factors.
   R² ≈ 0.20 means 80% of variance is unexplained. A Random Forest or XGBoost
   would perform much better but is outside the scope of this project.

3. **KMeans selects K=2** — with this dataset the algorithm finds 2 broad zones.
   More granular soil classification would need more diverse training data or a
   different feature set (texture, organic matter, etc.).

4. **No cross-validation** — we use a single train/test split. k-fold cross-validation
   would give more reliable metric estimates but is omitted for simplicity.
