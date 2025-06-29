# credit_scoring_model.py

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load dataset (replace with your dataset path or name)
# For now, let's simulate loading a dataset. Replace with: pd.read_csv("credit_data.csv")
data = pd.read_csv("D:/Credit_Score/credit-scoring-model/credit_data.csv")


# Step 3: Initial exploration
print("First 5 rows of data:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 4: Preprocessing
# Drop rows with missing values (you may also consider imputing them)
data.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(data.drop("target", axis=1))
y = data["target"]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Feature importance plot
importances = model.feature_importances_
features = data.drop("target", axis=1).columns
plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
