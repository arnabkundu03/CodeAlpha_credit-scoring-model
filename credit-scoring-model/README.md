# 🏦 Credit Scoring Model - Machine Learning Project

A simple yet powerful machine learning project that predicts whether a person is a good or bad credit risk based on their financial and demographic data.

🔗 **GitHub Repository**: [Click Here](https://github.com/arnabkundu03/CodeAlpha_credit-scoring-model/tree/master/credit-scoring-model)

---


## 🧠 What This Project Does

- Cleans and preprocesses the dataset
- Encodes categorical variables
- Scales the features
- Splits the data into training and test sets
- Trains a **Random Forest Classifier**
- Evaluates the model with a **confusion matrix** and **classification report**
- Shows a bar chart of the **most important features** used in prediction

---

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/arnabkundu03/CodeAlpha_credit-scoring-model.git
cd CodeAlpha_credit-scoring-model/credit-scoring-model
```

### 2. Install Required Libraries

Make sure Python is installed, then run:
```pip install -r requirements.txt```

3. Run the Python Script

Ensure that ```credit_data.csv``` is in the same directory and execute:
python ```"python credit_scoring_model.py"```

### 📈 Expected Output

You will see:

1. First 5 rows of the dataset printed

2.Dataset info (column types and non-null counts)

3. Confusion matrix and classification metrics (accuracy, precision, etc.)

4. A bar graph showing feature importance

### 📊 Dataset Format

The dataset should contain:

1. Several input features (e.g., age, job, income)

2. A target column:

     1 → Good credit risk

     0 → Bad credit risk

All missing values are dropped, and text fields are encoded using Label Encoding.

### 📦 Requirements

Listed in ```requirements.txt```

Install them with:
```pip install -r requirements.txt```

### 🧪 ML Model Used

RandomForestClassifier from ```scikit-learn```
Chosen for its robustness and accuracy in classification problems.
