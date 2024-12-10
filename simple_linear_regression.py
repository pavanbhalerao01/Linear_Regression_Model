# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('student_scores.csv')  # Ensure this file is in the same directory

# Display the first 5 rows of the dataset
print("Dataset Preview:")
print(data.head())

# Separate the features (X) and target (y)
X = data[['Hours']]  # Features
y = data['Scores']    # Target

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title("Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.show()

# Save model coefficients
print(f"Model Coefficient (b1): {model.coef_[0]:.2f}")
print(f"Model Intercept (b0): {model.intercept_:.2f}")

# Predict a specific value (e.g., for 6 hours of study)
hours_studied = 6
predicted_score = model.predict([[hours_studied]])
print(f"Predicted score for {hours_studied} hours of study: {predicted_score[0]:.2f}")
