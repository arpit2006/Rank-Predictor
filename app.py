import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# =========================
# Load Dataset
# =========================

data = pd.read_excel("Jee_data.xlsx")

# =========================
# Features and Labels
# =========================

X = data.drop("Rank", axis=1)
y = data["Rank"]

# =========================
# Preprocessing Pipeline
# =========================

num_attributes = X.columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes)
])

# =========================
# Prepare Data
# =========================

X_prepared = full_pipeline.fit_transform(X)

# =========================
# Train Model
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_prepared, y)

# =========================
# Streamlit UI
# =========================

st.title("🎯 JEE Rank Predictor")

st.write("Enter student details to predict JEE Rank")

# =========================
# User Inputs
# =========================

year = st.selectbox(
    "Select Year",
    [2022, 2023, 2024, 2025]
)

marks = st.number_input(
    "Enter Marks",
    min_value=0.0,
    max_value=300.0
)

percentile = st.number_input(
    "Enter Percentile",
    min_value=0.0,
    max_value=100.0
)

total_candidates = st.number_input(
    "Enter Total Candidates",
    min_value=1
)

# =========================
# Prediction Button
# =========================

if st.button("Predict Rank"):

    input_data = pd.DataFrame({
        "Year": [year],
        "Marks": [marks],
        "Percentile": [percentile],
        "Total_Candidates": [total_candidates]
    })

    input_prepared = full_pipeline.transform(input_data)

    prediction = model.predict(input_prepared)

    prediction = int(np.round(prediction[0]))

    st.success(f"🎯 Predicted JEE Rank: {prediction}")