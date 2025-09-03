import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "pass_exam": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})

# Features and labels
X = data[["hours"]]   # input (hours studied)
y = data["pass_exam"]  # output (0 or 1)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probability of passing for 4.5 hours
# Create a DataFrame for prediction with the same feature name 'hours'
hours_to_predict = pd.DataFrame([[4.5]], columns=["hours"])
prob = model.predict_proba(hours_to_predict)[0][1]  # probability of class 1
prediction = model.predict(hours_to_predict)

print(f"Probability of passing: {prob:.2f}")
print(f"Predicted class: {prediction[0]}")
