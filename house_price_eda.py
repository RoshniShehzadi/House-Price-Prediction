import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone
import joblib

# ===============================
# 1) Load dataset
# ===============================
df = pd.read_csv("kc_house_data.csv")

print(df.shape)
print(df.head())
print("\nInfo:")
df.info()

# ===============================
# 2) Date handling (NO dropping before correlation)
# ===============================
df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")

# Create numeric date features
df["year_sold"] = df["date"].dt.year
df["month_sold"] = df["date"].dt.month
df["day_sold"] = df["date"].dt.day

# Numeric representation of date for correlation
df["date_numeric"] = df["date"].astype("int64")   # nanoseconds since epoch

target = "price"

# ===============================
# 3) Correlation analysis
# ===============================
corr_matrix = df.corr(numeric_only=True)

print("\n=== Correlation with price (sorted) ===")
target_corr = corr_matrix[target].sort_values(ascending=False)
print(target_corr)

# ===============================
# 4) Select features strongly related to price
#    (close to -1 or +1)
# ===============================
# You can change this to 0.6 / 0.7 / 0.8 if you want stricter selection
strong_threshold = 0.5   # |corr| >= 0.5

strong_features = [
    col for col in target_corr.index
    if col != target and abs(target_corr[col]) >= strong_threshold
]

print(f"\nUsing correlation threshold |corr| >= {strong_threshold}")
print("Selected strongly correlated features:")
print(strong_features)

# X / y with only strongly correlated features
X = df[strong_features]
y = df[target]

n_samples = len(X)
print("\nTotal samples:", n_samples)
print("X shape:", X.shape, " | y shape:", y.shape)

# ===============================
# 5) Define K under constraints
#    - k <= 10
#    - each fold >= 30 instances
# ===============================
max_splits_by_size = n_samples // 30   # each fold has at least 30 samples
if max_splits_by_size < 2:
    raise ValueError(
        f"Not enough samples ({n_samples}) to have even 2 folds "
        f"with >= 30 instances each."
    )

k = min(10, max_splits_by_size)
print(f"\nUsing K-Fold with k = {k} folds (each fold >= 30 instances)")

kf = KFold(n_splits=k, shuffle=True, random_state=42)

# ===============================
# 6) Gradient Boosting model
# ===============================
base_model = GradientBoostingRegressor(random_state=42)

fold_mae = []
fold_mse = []
fold_rmse = []

fold_num = 1

# ===============================
# 7) K-Fold Cross-Validation
# ===============================
for train_idx, test_idx in kf.split(X):
    print(f"\n===== Fold {fold_num} / {k} =====")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = clone(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    fold_mae.append(mae)
    fold_mse.append(mse)
    fold_rmse.append(rmse)

    print(f"MAE  (fold {fold_num}): {mae:.4f}")
    print(f"MSE  (fold {fold_num}): {mse:.4f}")
    print(f"RMSE (fold {fold_num}): {rmse:.4f}")

    fold_num += 1

# ===============================
# 8) Overall K-Fold performance
# ===============================
mae_mean = np.mean(fold_mae)
mae_std  = np.std(fold_mae)
mse_mean = np.mean(fold_mse)
mse_std  = np.std(fold_mse)
rmse_mean = np.mean(fold_rmse)
rmse_std  = np.std(fold_rmse)

print("\n===== FINAL GRADIENT BOOSTING PERFORMANCE (K-FOLD AVERAGES) =====")
print(f"MAE  mean ± std : {mae_mean:.4f} ± {mae_std:.4f}")
print(f"MSE  mean ± std : {mse_mean:.4f} ± {mse_std:.4f}")
print(f"RMSE mean ± std : {rmse_mean:.4f} ± {rmse_std:.4f}")

# ===============================
# 9) Train FINAL model on ALL data
# ===============================
final_model = clone(base_model)
final_model.fit(X, y)

# Optional: feature importance
feature_importances = pd.Series(
    final_model.feature_importances_,
    index=strong_features
).sort_values(ascending=False)

print("\n=== Gradient Boosting Feature Importances (final model) ===")
print(feature_importances)

# ===============================
# 10) Save final model
# ===============================
artifact = {
    "model": final_model,
    "features": strong_features,
    "k_folds": k,
    "corr_threshold": strong_threshold,
    "cv_metrics": {
        "MAE_mean": mae_mean,
        "MAE_std": mae_std,
        "MSE_mean": mse_mean,
        "MSE_std": mse_std,
        "RMSE_mean": rmse_mean,
        "RMSE_std": rmse_std
    }
}

joblib.dump(artifact, "house_price_gradient_boosting_kfold.pkl")
print("\nFinal Gradient Boosting model saved to 'house_price_gradient_boosting_kfold.pkl'")
