Income Prediction using Machine Learning

📌 Project Overview

This project predicts whether an individual’s income exceeds \$50K/year based on demographic and employment attributes. The dataset used is the Adult Income Dataset (Census Income Dataset).

The goal is to build and evaluate multiple machine learning models, compare their performance, and determine the most accurate model for income classification.

---

📊 Dataset

* Source: UCI Machine Learning Repository – Adult Income Dataset
* Records: 32,561
* Features: 14 (both categorical and numerical)
* Target Variable: `income` (<=50K or >50K)

 Features include:

* Age
* Workclass
* Education
* Occupation
* Marital Status
* Relationship
* Race
* Gender
* Hours per week
* Native country
* …and more

---

 ⚙️ Steps Performed

1. Data Cleaning & Preprocessing

   * Handled missing values (`?` entries).
   * Encoded categorical features using **OneHotEncoding/LabelEncoding**.
   * Scaled numerical features.

2.Exploratory Data Analysis (EDA)

   * Visualized distributions of income across age, education, occupation, and hours-per-week.
   * Analyzed feature correlations with income.

3. Model Building
   Implemented and compared:

   * Logistic Regression
   * Decision Tree
   * Random Forest
   * XGBoost

4.Evaluation Metrics

   * Accuracy
   * Precision, Recall, F1-score
   * ROC-AUC Curve



🚀 Results

* Best Model: XGBoost
* Accuracy: 87.6%
* ROC AUC: 0.81
* Outperformed traditional classifiers in handling both categorical and numerical features efficiently.

---

🛠️ Tech Stack

Language: Python
Libraries:

  * pandas, numpy (data handling)
  * matplotlib, seaborn (visualization)
  * scikit-learn (ML models, preprocessing, metrics)
  * xgboost (advanced boosting model)


 📂 Project Structure

```
income-prediction/
│-- data/                # Dataset files
│-- notebooks/           # Jupyter notebooks for analysis & training
│-- results/             # Evaluation results & plots
│-- program.py # Main script for training & prediction
│-- README.md            # Project documentation
```

---

 ▶️ How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run Jupyter Notebook or Python script:

   ```bash
   jupyter notebook notebooks/program.ipynb
   ```

   or

   ```bash
   python program.py
   ```


 📈 Future Improvements

* Implement deep learning models (ANN).
* Perform hyperparameter tuning with GridSearchCV / RandomizedSearchCV.
* Deploy as a web app using Flask / Streamlit.
* Handle class imbalance with SMOTE or other techniques.

👨‍💻 Author

Ravi Kumar Chittiboyina**
B.Tech (CSE) | Aspiring Data Scientist
