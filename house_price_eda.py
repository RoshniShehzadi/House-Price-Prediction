import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("kc_house_data.csv")

# ---- Quick EDA (optional but useful) ----
print(df.shape)      # rows, columns
print(df.head())     # first 5 rows

print("\nData types and non-null counts:")
df.info()

print("\nSummary statistics (numerical columns):")
print(df.describe())

# 2. Convert 'date' to datetime and create numeric date features
df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
df["year_sold"] = df["date"].dt.year
df["month_sold"] = df["date"].dt.month

# 3. Drop raw 'date' and 'id' (not useful as feature)
df = df.drop(columns=["date", "id"])

# 4. Define features (X) and target (y)
target = "price"
feature_cols = [col for col in df.columns if col != target]

X = df[feature_cols]
y = df[target]

print("\nFeature columns:")
print(feature_cols)
print("\nX shape:", X.shape, " | y shape:", y.shape)

# 5. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Define and train Random Forest model
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# 7. Predict on test set
y_pred = rf.predict(X_test)

# 8. Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== RandomForestRegressor performance ===")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# 9. Feature importance
feature_importances = pd.Series(
    rf.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nTop 10 important features according to RandomForest:")
print(feature_importances.head(10))

import joblib 
# 9. Save model to .pkl
model_artifact = {
    "model": rf,
    "features": feature_cols
}

joblib.dump(model_artifact, "house_price_rf_model.pkl")
print("\nModel saved to 'house_price_rf_model.pkl'")
