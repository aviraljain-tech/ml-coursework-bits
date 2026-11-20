import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import math

# Define file path where you'll save the dataset
dataset_path = "abalone.data"

# Load the dataset from the local file
column_names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", 
                "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
abalone = pd.read_csv(dataset_path, names=column_names)

# The 'Rings' column is the age of the abalone (add 1.5 to get the age in years)
abalone['Age'] = abalone['Rings'] + 1.5

# Calculate log of age (for parts c and d)
abalone['Log_Age'] = np.log(abalone['Age'])

# Create numeric encoding for gender
sex_map = {'M': 1, 'F': 0, 'I': -1}
abalone['Sex_Numeric'] = abalone['Sex'].map(sex_map)

# Print the first few rows to understand the data
print("First 5 rows of the dataset with log age:")
print(abalone.head())

# Print some basic statistics
print("\nDataset information:")
print(abalone.info())

# Check for missing values
print(f"\nMissing values in the dataset: {abalone.isnull().sum().sum()}")


print("\n\n--- PART C: LOG AGE WITHOUT GENDER ---")

# Prepare data for regression (ignoring gender)
X_c = abalone.drop(['Sex', 'Sex_Numeric', 'Rings', 'Age', 'Log_Age'], axis=1)
y_c = abalone['Log_Age']

# Create and train the linear regression model
model_c = LinearRegression()
model_c.fit(X_c, y_c)

# Make predictions
y_pred_c = model_c.predict(X_c)

# Calculate metrics
mse_c = mean_squared_error(y_c, y_pred_c)
r2_c = r2_score(y_c, y_pred_c)

# Print model coefficients and metrics
print("\nModel Coefficients:")
for feature, coef in zip(X_c.columns, model_c.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model_c.intercept_:.4f}")
print(f"Mean Squared Error: {mse_c:.4f}")
print(f"R² Score: {r2_c:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse_c):.4f}")

# Plot residuals against fitted values
plt.figure(figsize=(10, 6))
residuals_c = y_c - y_pred_c

plt.scatter(y_pred_c, residuals_c, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Fitted Values (Log Age)')
plt.ylabel('Residuals')
plt.title('Residual Plot for Log Abalone Age Prediction (Without Gender)')
plt.grid(True, alpha=0.3)

# Add a trend line to check for patterns in residuals
z = np.polyfit(y_pred_c, residuals_c, 1)
p = np.poly1d(z)
plt.plot(y_pred_c, p(y_pred_c), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig('abalone_log_residual_plot_without_gender.png')
plt.show()


print("\n\n--- PART D: LOG AGE WITH GENDER ---")

# Prepare data for regression (including gender)
X_d = abalone.drop(['Sex', 'Rings', 'Age', 'Log_Age'], axis=1)  # Keep Sex_Numeric
y_d = abalone['Log_Age']

# Create and train the linear regression model
model_d = LinearRegression()
model_d.fit(X_d, y_d)

# Make predictions
y_pred_d = model_d.predict(X_d)

# Calculate metrics
mse_d = mean_squared_error(y_d, y_pred_d)
r2_d = r2_score(y_d, y_pred_d)

# Print model coefficients and metrics
print("\nModel Coefficients:")
for feature, coef in zip(X_d.columns, model_d.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model_d.intercept_:.4f}")
print(f"Mean Squared Error: {mse_d:.4f}")
print(f"R² Score: {r2_d:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse_d):.4f}")

# Plot residuals against fitted values
plt.figure(figsize=(10, 6))
residuals_d = y_d - y_pred_d

plt.scatter(y_pred_d, residuals_d, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Fitted Values (Log Age)')
plt.ylabel('Residuals')
plt.title('Residual Plot for Log Abalone Age Prediction (With Gender)')
plt.grid(True, alpha=0.3)

# Add a trend line to check for patterns in residuals
z = np.polyfit(y_pred_d, residuals_d, 1)
p = np.poly1d(z)
plt.plot(y_pred_d, p(y_pred_d), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig('abalone_log_residual_plot_with_gender.png')
plt.show()



print("\n\n--- PART F: ELASTICNET REGULARIZATION ---")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_d,  # Using the dataset with gender
    y_d,  # Using log age as target
    test_size=0.2,
    random_state=42
)

# Define parameter grid for ElasticNet
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    ElasticNet(max_iter=10000, random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
print(f"Best ElasticNet Parameters: {best_params}")

# Train the ElasticNet model with the best parameters
elasticnet_model = ElasticNet(
    alpha=best_params['alpha'],
    l1_ratio=best_params['l1_ratio'],
    max_iter=10000,
    random_state=42
)
elasticnet_model.fit(X_train, y_train)

# Make predictions
y_pred_train_en = elasticnet_model.predict(X_train)
y_pred_test_en = elasticnet_model.predict(X_test)

# Calculate metrics for training and test sets
train_mse = mean_squared_error(y_train, y_pred_train_en)
train_r2 = r2_score(y_train, y_pred_train_en)
test_mse = mean_squared_error(y_test, y_pred_test_en)
test_r2 = r2_score(y_test, y_pred_test_en)

print("\nElasticNet Model Coefficients:")
for feature, coef in zip(X_d.columns, elasticnet_model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {elasticnet_model.intercept_:.4f}")

print(f"\nTraining Mean Squared Error: {train_mse:.4f}")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Training Root Mean Squared Error: {np.sqrt(train_mse):.4f}")

print(f"\nTest Mean Squared Error: {test_mse:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Test Root Mean Squared Error: {np.sqrt(test_mse):.4f}")

# Plot residuals against fitted values for the ElasticNet model
plt.figure(figsize=(10, 6))
residuals_test_en = y_test - y_pred_test_en

plt.scatter(y_pred_test_en, residuals_test_en, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Fitted Values (Log Age)')
plt.ylabel('Residuals')
plt.title('Residual Plot for ElasticNet Model (Test Set)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('abalone_elasticnet_residual_plot.png')
plt.show()

# Compare all models
print("\n--- MODEL COMPARISON ---")
models = {
    "Linear (Age)": f"R²: {r2_c:.4f}, RMSE: {np.sqrt(mse_c):.4f}",
    "Linear (Age) with Gender": f"R²: {r2_d:.4f}, RMSE: {np.sqrt(mse_d):.4f}",
    "ElasticNet (Log Age) with Gender": f"R²: {test_r2:.4f}, RMSE: {np.sqrt(test_mse):.4f}"
}

for model_name, metrics in models.items():
    print(f"{model_name}: {metrics}")

