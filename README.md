# 🎯 Rank Predictor

 Machine Learning web application that predicts a student's expected **JEE Rank** using historical exam-related data such as marks, percentile, year, and total number of candidates.

The project demonstrates a complete **end-to-end Machine Learning workflow** including data preprocessing, model training, evaluation, visualization, prediction export, and deployment using **Streamlit**.

---

# 📌 Overview

Rank Predictor uses regression-based Machine Learning models to estimate a student's probable JEE rank based on exam statistics.

The project was built to:
- understand regression algorithms
- Data visualization
- practice preprocessing pipelines
- implement real-world ML workflows
- deploy an ML model using Streamlit

The application currently uses:
- **Linear Regression**
- **Random Forest Regressor**

where Random Forest provides significantly better accuracy and prediction performance.

---

# 🚀 Features

✅ JEE Rank Prediction  
✅ Machine Learning Regression Models  
✅ Data Preprocessing Pipeline  
✅ Feature Scaling & Missing Value Handling  
✅ Data Visualization using Matplotlib  
✅ CSV Export of Predictions  
✅ Streamlit Interactive UI  
✅ Real-time Rank Prediction  
✅ Model Evaluation Metrics  
✅ End-to-End ML Workflow  

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Handling |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Web Application UI |

---

# 📂 Project Structure

```bash
Rank-Predictor/
│
├── app.py
├── model.py
├── Jee_data.xlsx
├── linear_regression_predictions.csv
├── random_forest_predictions.csv
├── requirements.txt
├── README.md
└── images/
```

---

# ⚙️ Machine Learning Workflow

```text
Data Collection
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Preprocessing Pipeline
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction Export
        ↓
Streamlit Deployment
```

---

# 📊 Dataset Features

The dataset includes:

| Feature | Description |
|---|---|
| Year | Examination Year |
| Marks | Student Marks |
| Percentile | JEE Percentile |
| Total_Candidates | Total Students Appeared |
| Rank | Target Variable |

---

# 📈 Data Visualization

The project includes multiple visualizations:

- Marks vs Percentile
- Marks vs Rank
- Percentile vs Rank
- Marks Distribution

These plots help understand:
- feature relationships
- rank trends
- score distribution
- dataset patterns

---

# 🤖 Machine Learning Models

## 1️⃣ Linear Regression

Linear Regression was used as a baseline regression model.

### Advantages
- Simple
- Fast
- Easy to understand

### Limitations
- Performs poorly on nonlinear relationships
- Can predict negative values

---

## 2️⃣ Random Forest Regressor

Random Forest Regressor is the primary model used in the project.

### Advantages
- Handles nonlinear relationships
- Better prediction accuracy
- Reduces overfitting
- More stable predictions

### Performance
Random Forest significantly outperformed Linear Regression in:
- RMSE
- MAE
- R² Score

---

# 📏 Evaluation Metrics

The models are evaluated using:

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error |
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| R² Score | Model Accuracy Score |

---

# 🧹 Data Preprocessing

A complete preprocessing pipeline was implemented using:

- `SimpleImputer`
- `StandardScaler`
- `ColumnTransformer`
- `Pipeline`

This automates:
- missing value handling
- feature scaling
- preprocessing workflow

---

# 🌐 Streamlit Web App

The project includes an interactive Streamlit web application where users can:

✅ Select Exam Year  
✅ Enter Marks  
✅ Enter Percentile  
✅ Enter Total Candidates  
✅ Predict Expected JEE Rank  

---

# 🖥️ Streamlit Preview

```text
🎯 JEE Rank Predictor

Select Year: 2025
Enter Marks: 250
Enter Percentile: 99
Enter Total Candidates: 1400000

🎯 Predicted JEE Rank: 10366
```

---

# ▶️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Rank-Predictor.git
```

---

## 2️⃣ Navigate to Project Directory

```bash
cd Rank-Predictor
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📦 Generate Requirements File

```bash
pip freeze > requirements.txt
```

---

# 📤 Prediction Export

The project exports prediction results into CSV files:

- `linear_regression_predictions.csv`
- `random_forest_predictions.csv`

The exported files include:
- actual rank
- predicted rank
- prediction error

---

# 📈 Example Prediction

| Year | Marks | Percentile | Total Candidates | Predicted Rank |
|---|---|---|---|---|
| 2025 | 250 | 99 | 1400000 | ~10366 |

---

# 🔥 Future Improvements

- Hyperparameter Tuning using GridSearchCV
- XGBoost Integration
- CatBoost Regressor
- Deep Learning Models
- Cloud Deployment
- Live Rank Analysis
- User Authentication
- Advanced Analytics Dashboard
- Model Persistence using Joblib
- Docker Deployment

---

# 💡 Key Learning Outcomes

This project helped in learning:

✅ Regression Models  
✅ Machine Learning Pipelines  
✅ Data Visualization  
✅ Feature Engineering  
✅ Model Evaluation  
✅ Streamlit Deployment  
✅ End-to-End ML Workflow  
✅ Real-world ML Project Structure  

---

# 📚 Concepts Used

- Supervised Learning
- Regression
- Ensemble Learning
- Feature Scaling
- Data Preprocessing
- Model Evaluation
- Machine Learning Deployment

---

# 📜 License

This project is open-source and available under the MIT License.

---

# 👨‍💻 Author

**Arpit Shirbhate**

- Data Science Enthusiast
- Machine Learning Learner
- Open Source Contributor

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository  
🍴 Fork the project  
🛠️ Contribute improvements  
🚀 Share with others
