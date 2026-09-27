# Student Performance AI

An interactive machine learning web application built with **Python**, **scikit-learn**, and **Streamlit** that estimates a student's final academic grade (**G3**) on a 0-20 scale using academic, demographic, family, and behavioral information.

---

## Live Demo

Try the deployed application:

**[Student Grade Prediction AI](https://student-grade-prediction-ai.streamlit.app/)**

---

## Project Overview

This project implements an end-to-end machine learning workflow for predicting student final performance using the **UCI Student Performance Dataset**.

The final application uses a scikit-learn `Pipeline` that combines:

- **Numerical preprocessing:** `StandardScaler`
- **Categorical preprocessing:** `OneHotEncoder`
- **Regression model:** `GradientBoostingRegressor`

The trained pipeline is saved using `joblib` and loaded by the Streamlit application for real-time predictions.

---

## Prediction Target

The model predicts:

**G3 - Final Grade**

The original dataset represents the final grade on a **0-20 scale**.

---

## Dataset

This project uses the **UCI Student Performance Dataset**, specifically the mathematics student dataset (`student-mat.csv`).

The dataset contains information related to:

- Previous academic grades
- Study habits
- Absences
- Family background
- Demographic characteristics
- School-related information
- Social and behavioral factors

The raw CSV dataset is used locally for model development and training and is excluded from the Git repository through `.gitignore`.

---

## Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   |
   v
Data Understanding & Analysis
   |
   v
Feature Preparation
   |
   v
Train/Test Split
   |
   v
Preprocessing Pipeline
   |
   v
Model Comparison
   |
   v
Cross-Validation
   |
   v
Hyperparameter Tuning
   |
   v
Final Gradient Boosting Model
   |
   v
Model Evaluation
   |
   v
Model Serialization
   |
   v
Streamlit Application

Final Model

The final model is a:

Gradient Boosting Regressor

The complete preprocessing and prediction pipeline is saved as:

models/gradient_boosting_model.pkl

This allows the Streamlit application to load the trained pipeline directly without retraining the model every time the application starts.

Repository Structure
student-performance-ai/
|
├── models/
│   └── gradient_boosting_model.pkl
|
├── notebooks/
│   └── 01_data_analysis_and_modeling.ipynb
|
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

The raw dataset is intentionally excluded from the repository through .gitignore.

Streamlit Application

The application provides an interactive interface where users can enter student information and receive an estimated final grade.

The interface organizes the inputs into sections such as:

Academic Performance
Student & Family
School & Activities
Advanced Information

Some advanced fields contain pre-filled values. If these values are not changed, they are used as assumptions when generating the prediction.

Run Locally

1. Clone the repository
git clone https://github.com/muhammad-habeeb/student-performance-ai.git
cd student-performance-ai
2. Create a virtual environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run the application
streamlit run app.py

The application will then open in your browser.

Technologies Used

Python
Pandas
Scikit-learn
Joblib
Streamlit
Jupyter Notebook

Limitations

This project is a machine learning portfolio application based on the UCI Student Performance Dataset.

The predictions should be treated as estimates, not as official academic assessments.

Because the model was trained on a specific dataset, its performance may not generalize directly to students from different schools, educational systems, or populations.

The dataset represents students from specific Portuguese secondary schools, so broader generalization would require evaluation using additional and more diverse data.

Future Improvements

Potential improvements include:

Evaluating the model on data from additional schools
Testing generalization on external datasets
Improving feature selection
Adding model explainability
Adding prediction uncertainty
Improving the user experience
Monitoring model performance after deployment

Project Links
Live Demo: Student Grade Prediction AI
GitHub Repository: Student Performance AI

Project

Student Performance AI

Machine Learning Portfolio Project