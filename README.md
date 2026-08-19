# CardioPredict — Heart Disease Prediction

## Overview

CardioPredict is a machine learning project designed to predict the risk of heart disease based on patient clinical data.

The project analyzes clinical features, performs exploratory data analysis and preprocessing, compares multiple machine learning models, and deploys the final model through a FastAPI web application.

The system is designed as a decision-support tool to help prioritize patients based on their predicted risk level. It does not replace professional medical diagnosis.

---

## Project Features

* Analysis of 270,000 patient records
* 12 clinical input features
* Exploratory Data Analysis (EDA)
* Data preprocessing and feature scaling
* Data leakage detection and removal
* Stratified train/test split
* Comparison of three machine learning models:

  * Gradient Boosting
  * Random Forest
  * Logistic Regression
* Hyperparameter optimization using `RandomizedSearchCV`
* Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * ROC-AUC
* FastAPI web application for real-time predictions
* Patient risk classification:

  * Low Risk
  * Moderate Risk
  * High Risk

---

## Dataset

The project uses clinical patient data containing demographic and cardiovascular-related features.

Example features include:

| Feature         | Description                         |
| --------------- | ----------------------------------- |
| Age             | Patient age                         |
| Sex             | Patient gender                      |
| Chest pain type | Type of chest pain                  |
| BP              | Resting blood pressure              |
| Cholesterol     | Serum cholesterol level             |
| FBS over 120    | Fasting blood sugar above 120 mg/dl |
| Max HR          | Maximum heart rate during exercise  |
| Exercise angina | Chest pain during exercise          |
| ST depression   | ST segment depression               |
| Vessels fluro   | Number of major vessels detected    |

The target variable is binary:

* `0` — Normal
* `1` — Disease

A potential data leakage feature, `Thallium`, was removed from the final model features after target creation.

---

## Data Preprocessing

The preprocessing pipeline includes:

1. Creating the target variable
2. Removing the leakage feature
3. Checking for missing values
4. Splitting the dataset into training and testing sets
5. Using a stratified 80/20 train-test split
6. Scaling features using `StandardScaler`
7. Saving the scaler and feature column order for consistent predictions in the web application

Example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Machine Learning Models

Three models were evaluated:

### 1. Gradient Boosting

A sequential ensemble model that improves predictions by correcting errors from previous models.

### 2. Random Forest

A tree-based ensemble model that provides a robust baseline and feature importance.

### 3. Logistic Regression

A simple and interpretable linear classification model used as a baseline.

Hyperparameters were optimized using:

```python
RandomizedSearchCV
```

The optimization used stratified cross-validation and recall as the scoring metric.

---

## Model Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Gradient Boosting   |   76.54% |    72.44% | 68.76% |   70.55% |  82.52% |
| Random Forest       |   75.93% |    68.82% | 75.13% |   71.83% |  82.33% |
| Logistic Regression |   75.85% |    69.35% | 73.27% |   71.26% |  82.06% |

### Final Model

**Gradient Boosting** was selected as the final model because it achieved the best overall balance across the evaluation metrics.

* Accuracy: **76.54%**
* Precision: **72.44%**
* F1-Score: **70.55%**
* ROC-AUC: **82.52%**

Random Forest achieved the highest recall, while Gradient Boosting provided the strongest overall performance.

---

## Web Application

The trained model is deployed using **FastAPI**.

### Prediction Flow

1. User enters patient clinical information.
2. The application creates a DataFrame using the correct feature order.
3. The saved scaler transforms the input data.
4. The trained model calculates the probability of disease.
5. The system classifies the patient into a risk category.

Risk categories:

* **Low Risk:** probability below 0.35
* **Moderate Risk:** probability between 0.35 and 0.60
* **High Risk:** probability above 0.60

---

## Project Structure

```text
CardioPredict/
│
├── data/
│   └── data.csv
│
├── processed/
│   ├── scaler.pkl
│   └── columns.pkl
│
├── models/
│   └── model.pkl
│
├── web_site/
│   ├── main.py
│   ├── requirements.txt
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       └── style.css
│
├── eda.py
├── preprocessing.py
├── train_model.py
└── README.md
```

---

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* FastAPI
* Uvicorn
* Jinja2
* HTML
* CSS
* pickle

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/CardioPredict.git
```

Move into the project directory:

```bash
cd CardioPredict
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Navigate to the web application folder:

```bash
cd web_site
```

Run the FastAPI application:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

## Important Note

This project is intended for educational and research purposes. The predictions generated by the model should not be used as a substitute for professional medical diagnosis or treatment.

---

## Author

**Dilsora Ibodullayeva**

Computer Science Student
Interested in Artificial Intelligence, Machine Learning, and Data Science.

---

⭐ If you find this project interesting, feel free to star the repository!

