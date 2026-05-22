import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error
)

# =========================
# LOAD DATASET
# =========================

print("\nLoading Dataset...\n")

data = pd.read_excel("Jee_data.xlsx")

print(data.head())
print(data.tail())
print(data.describe())

# =========================
# DATA VISUALIZATION
# =========================

plt.figure(figsize=(12,8))

# Plot 1
plt.subplot(2,2,1)
plt.scatter(data["Marks"], data["Percentile"])
plt.title("Marks vs Percentile")
plt.xlabel("Marks")
plt.ylabel("Percentile")

# Plot 2
plt.subplot(2,2,2)
plt.scatter(data["Marks"], data["Rank"])
plt.title("Marks vs Rank")
plt.xlabel("Marks")
plt.ylabel("Rank")

# Plot 3
plt.subplot(2,2,3)
plt.scatter(data["Percentile"], data["Rank"])
plt.title("Percentile vs Rank")
plt.xlabel("Percentile")
plt.ylabel("Rank")

# Plot 4
plt.subplot(2,2,4)
plt.hist(data["Marks"])
plt.title("Marks Distribution")
plt.xlabel("Marks")

plt.tight_layout()
plt.show()

# =========================
# FEATURES AND LABELS
# =========================

X = data.drop("Rank", axis=1)
y = data["Rank"].copy()

# =========================
# TRAIN TEST SPLIT
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("\nTrain Set Shape:")
print(x_train.shape)

print("\nTest Set Shape:")
print(x_test.shape)

# =========================
# PIPELINE
# =========================

num_attributes = x_train.columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes)
])

# =========================
# PREPARE DATA
# =========================

print("\nPreparing Data...\n")

x_train_prepared = full_pipeline.fit_transform(x_train)
x_test_prepared = full_pipeline.transform(x_test)

# =========================
# LINEAR REGRESSION
# =========================

print("\nTraining Linear Regression Model...\n")

for i in tqdm(range(50)):
    time.sleep(0.02)

linear_model = LinearRegression()

linear_model.fit(x_train_prepared, y_train)

print("\nLinear Regression Training Completed!\n")

# Predictions
lin_predictions = linear_model.predict(x_test_prepared)

# =========================
# LINEAR REGRESSION RESULTS
# =========================

print("\n===== Linear Regression Results =====\n")

print("RMSE:",
      root_mean_squared_error(y_test, lin_predictions))

print("R2 Score:",
      r2_score(y_test, lin_predictions))

print("MSE:",
      mean_squared_error(y_test, lin_predictions))

print("MAE:",
      mean_absolute_error(y_test, lin_predictions))

print("\nSample Predictions:")
print(np.round(lin_predictions[:10]).astype(int))

# =========================
# RANDOM FOREST
# =========================

print("\nTraining Random Forest Model...\n")

for epoch in tqdm(range(100)):
    time.sleep(0.02)

model_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model_regressor.fit(x_train_prepared, y_train)

print("\nRandom Forest Training Completed!\n")

# Predictions
reg_predictions = model_regressor.predict(x_test_prepared)

# =========================
# RANDOM FOREST RESULTS
# =========================

print("\nEvaluating Model...\n")

for i in tqdm(range(50)):
    time.sleep(0.02)

print("\n===== Random Forest Results =====\n")

print("RMSE:",
      root_mean_squared_error(y_test, reg_predictions))

print("R2 Score:",
      r2_score(y_test, reg_predictions))

print("MSE:",
      mean_squared_error(y_test, reg_predictions))

print("MAE:",
      mean_absolute_error(y_test, reg_predictions))

# Convert predictions to integer
reg_predictions = np.round(reg_predictions).astype(int)

print("\nSample Predictions:")
print(reg_predictions[:10])

# =========================
# EXPORT LINEAR REGRESSION
# =========================

linear_output = pd.DataFrame({
    "Actual Rank": y_test.reset_index(drop=True),
    "Predicted Rank":
        np.round(lin_predictions).astype(int)
})

linear_output["Error"] = (
    linear_output["Actual Rank"]
    - linear_output["Predicted Rank"]
)

linear_output.to_csv(
    "linear_regression_predictions.csv",
    index=False
)

print("\nLinear Regression predictions exported!")

# =========================
# EXPORT RANDOM FOREST
# =========================

random_forest_output = pd.DataFrame({
    "Actual Rank": y_test.reset_index(drop=True),
    "Predicted Rank": reg_predictions
})

random_forest_output["Error"] = (
    random_forest_output["Actual Rank"]
    - random_forest_output["Predicted Rank"]
)

random_forest_output["Absolute Error"] = (
    random_forest_output["Error"].abs()
)

random_forest_output.to_csv(
    "random_forest_predictions.csv",
    index=False
)

print("\nRandom Forest predictions exported!")

# =========================
# SAVE MODEL
# =========================

import joblib

joblib.dump(
    model_regressor,
    "jee_rank_predictor.pkl"
)

print("\nModel saved successfully!")

print("\n===== PROJECT COMPLETED =====")