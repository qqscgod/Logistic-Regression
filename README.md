# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Step 2: Load the dataset (using Iris dataset as an example)
data = load_iris()

# For simplicity, we'll focus on just two classes (setosa and versicolor)
X = data.data[data.target != 2]  # Features (exclude Virginica)
y = data.target[data.target != 2]  # Target variable (exclude Virginica)

# Step 3: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Logistic Regression model
model = LogisticRegression(max_iter=200)  # max_iter=200 is used to ensure convergence
model.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 6: Evaluate the model's performance
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optionally, you can also look at predicted probabilities
y_prob = model.predict_proba(X_test)  # Probabilities for each class
print("Predicted Probabilities:")
print(y_prob)
