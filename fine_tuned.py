import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error
)

# =========================
# Load Dataset
# =========================

data = pd.read_excel("Jee_data.xlsx")

# =========================
# Features and Labels
# =========================

X = data.drop("Rank", axis=1)
y = data["Rank"].copy()

# =========================
# Train Test Split
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# =========================
# Numerical Columns
# =========================

num_attributes = x_train.columns.tolist()

# =========================
# Numerical Pipeline
# =========================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# =========================
# Full Pipeline
# =========================

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes)
])

# =========================
# Prepare Data
# =========================

x_train_prepared = full_pipeline.fit_transform(x_train)
x_test_prepared = full_pipeline.transform(x_test)

# Hyperparameter combinations to try
parameter_grid = {
    # Number of trees in the Random Forest
    # GridSearchCV will try:
    # 50 trees
    # 100 trees
    # 200 trees
    "n_estimators": [50, 100, 200],
    # Maximum depth of each tree
    # None means unlimited depth
    # 10 and 20 restrict tree growth
    "max_depth": [None, 10, 20],
    # Minimum samples required to split a node
    # Smaller value = more splits
    # Larger value = more controlled tree
    "min_samples_split": [2, 5, 10],
    # Minimum samples required in a leaf node
    # Helps reduce overfitting
    "min_samples_leaf": [1, 2, 4]
}

# =========================
# Grid Search
# =========================

grid_search = GridSearchCV(

    # Base model to optimize
    estimator=RandomForestRegressor(random_state=42),
    # Parameter combinations to test
    param_grid=parameter_grid,
    # 5-Fold Cross Validation
    # Dataset split into 5 parts
    # Model trains/tests 5 times for each combination
    cv=5,
    # Evaluation metric
    # Higher R2 score = better model
    scoring="r2",
    # Use all CPU cores for faster training
    n_jobs=-1,
    # Show training progress
    verbose=2
)

# =========================
# Train Fine Tuned Model
# =========================

grid_search.fit(x_train_prepared, y_train)

# =========================
# Best Parameters
# =========================

print("Best Parameters:")
print(grid_search.best_params_)

# =========================
# Best Model
# =========================

best_model = grid_search.best_estimator_

# =========================
# Predictions
# =========================

predictions = best_model.predict(x_test_prepared)

# Convert to integer ranks
predictions = np.round(predictions).astype(int)

# =========================
# Evaluation
# =========================

print("\n===== Fine Tuned Model Results =====")

print("R2 Score:", r2_score(y_test, predictions))

print("RMSE:", root_mean_squared_error(y_test, predictions))

print("MSE:", mean_squared_error(y_test, predictions))

print("MAE:", mean_absolute_error(y_test, predictions))

# =========================
# Prediction Comparison
# =========================

results = pd.DataFrame({
    "Actual Rank": y_test.reset_index(drop=True),
    "Predicted Rank": predictions
})

results["Error"] = (
    results["Actual Rank"]
    - results["Predicted Rank"]
)

results["Absolute Error"] = results["Error"].abs()

print("\nPrediction Samples:")
print(results.head(10))

# =========================
# Export Results
# =========================

results.to_csv("fine_tuned_predictions.csv", index=False)

print("\nPredictions exported successfully!")

# =========================
# Save Model
# =========================

import joblib

joblib.dump(best_model, "fine_tuned_jee_rank_model.pkl")

print("Model saved successfully!")