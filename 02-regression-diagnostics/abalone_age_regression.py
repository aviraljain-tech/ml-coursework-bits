import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset_path = "abalone.data"

column_names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", 
                "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
abalone = pd.read_csv(dataset_path, names=column_names)

# The 'Rings' column is the age of the abalone (add 1.5 to get the age in years)
abalone['Age'] = abalone['Rings'] + 1.5

# Print the first few rows to understand the data
print("First 5 rows of the dataset:")
print(abalone.head())

# Print some basic statistics
print("\nDataset information:")
print(abalone.info())
print("\nSummary statistics:")
print(abalone.describe())

# Check for missing values
print(f"\nMissing values in the dataset: {abalone.isnull().sum().sum()}")

# Prepare data for regression (ignoring gender)
# Select all numerical features except 'Sex', 'Rings', and 'Age'
X = abalone.drop(['Sex', 'Rings', 'Age'], axis=1)
# Target variable (age)
y = abalone['Age']

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print model coefficients and metrics
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

# Plot residuals against fitted values
plt.figure(figsize=(10, 6))
residuals = y - y_pred

plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Abalone Age Prediction')
plt.grid(True, alpha=0.3)

# Add a trend line to check for patterns in residuals
z = np.polyfit(y_pred, residuals, 1)
p = np.poly1d(z)
plt.plot(y_pred, p(y_pred), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig('abalone_residual_plot.png')
plt.show()
