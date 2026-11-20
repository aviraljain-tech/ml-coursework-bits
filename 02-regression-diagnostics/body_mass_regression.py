import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from urllib.request import urlopen
from io import StringIO

# Download the data
url = "http://www.statsci.org/data/oz/physical.txt"
data_str = urlopen(url).read().decode('utf-8')
data = pd.read_csv(StringIO(data_str), sep='\t')

# Clean up column names and display data overview
data.columns = data.columns.str.strip()
print(data.head())
print(data.info())

# Prepare data for regression
# Use the actual column names from the dataset
predictors = ['Fore', 'Bicep', 'Chest', 'Neck', 'Shoulder', 'Waist', 'Thigh', 'Calf', 'Height']
X = data[predictors]
y = data['Mass']

# (a) Linear regression of mass against all measurements
model_mass = LinearRegression()
model_mass.fit(X, y)
y_pred_mass = model_mass.predict(X)
residuals_mass = y - y_pred_mass

# Print regression coefficients
print("\nMass Regression Coefficients:")
print(f"Intercept: {model_mass.intercept_:.4f}")
for name, coef in zip(X.columns, model_mass.coef_):
    print(f"{name}: {coef:.4f}")
print(f"R^2 Score: {model_mass.score(X, y):.4f}")

# Plot residuals vs fitted values for mass regression
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_mass, residuals_mass)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('(a) Residuals vs. Fitted Values - Mass Regression')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mass_residuals.png')
plt.close()

# (b) Regression with cube root of mass
y_cbrt = np.cbrt(y)
model_cbrt = LinearRegression()
model_cbrt.fit(X, y_cbrt)
y_pred_cbrt = model_cbrt.predict(X)
residuals_cbrt = y_cbrt - y_pred_cbrt

# Convert predictions back to original scale for comparison
y_pred_cbrt_orig = y_pred_cbrt**3
residuals_cbrt_orig = y - y_pred_cbrt_orig

# Print regression coefficients for cube root model
print("\nCube Root Mass Regression Coefficients:")
print(f"Intercept: {model_cbrt.intercept_:.4f}")
for name, coef in zip(X.columns, model_cbrt.coef_):
    print(f"{name}: {coef:.4f}")
print(f"R^2 Score: {model_cbrt.score(X, y_cbrt):.4f}")

# Plot residuals vs fitted values for cube root regression (in cube root scale)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_cbrt, residuals_cbrt)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values (Cube Root Scale)')
plt.ylabel('Residuals (Cube Root Scale)')
plt.title('(b) Residuals vs. Fitted Values - Cube Root Scale')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cbrt_residuals.png')
plt.close()

# Plot residuals vs fitted values for cube root regression (in original scale)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_cbrt_orig, residuals_cbrt_orig)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values (Original Scale)')
plt.ylabel('Residuals (Original Scale)')
plt.title('(b) Residuals vs. Fitted Values - Original Scale from Cube Root Model')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cbrt_orig_residuals.png')
plt.close()

# Additional analysis and comparison
# Compare the models using RMSE
rmse_mass = np.sqrt(np.mean(residuals_mass**2))
rmse_cbrt_orig = np.sqrt(np.mean(residuals_cbrt_orig**2))

print("\nModel Comparison:")
print(f"RMSE for direct mass regression: {rmse_mass:.4f}")
print(f"RMSE for cube root regression (in original scale): {rmse_cbrt_orig:.4f}")

# Create a comprehensive comparison figure with all three residual plots
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# Plot 1: Direct mass regression residuals
axes[0].scatter(y_pred_mass, residuals_mass)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('(a) Mass Regression Residuals')
axes[0].grid(True, alpha=0.3)

# Plot 2: Cube root regression residuals in cube root scale
axes[1].scatter(y_pred_cbrt, residuals_cbrt)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Fitted Values (Cube Root Scale)')
axes[1].set_ylabel('Residuals (Cube Root Scale)')
axes[1].set_title('(b) Cube Root Regression Residuals - Cube Root Scale')
axes[1].grid(True, alpha=0.3)

# Plot 3: Cube root regression residuals in original scale
axes[2].scatter(y_pred_cbrt_orig, residuals_cbrt_orig)
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_xlabel('Fitted Values (Original Scale)')
axes[2].set_ylabel('Residuals (Original Scale)')
axes[2].set_title('(b) Cube Root Regression Residuals - Original Scale')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('all_residual_plots.png')
plt.show()

# Generate histograms of residuals to examine their distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(residuals_mass, bins=10, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribution of Residuals - Mass Model')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(residuals_cbrt_orig, bins=10, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribution of Residuals - Cube Root Model')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('residual_distributions.png')
plt.show()

# Generate Q-Q plots to check for normality of residuals
from scipy import stats

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
stats.probplot(residuals_mass, plot=plt)
plt.title('Q-Q Plot - Mass Regression Residuals')

plt.subplot(1, 2, 2)
stats.probplot(residuals_cbrt_orig, plot=plt)
plt.title('Q-Q Plot - Cube Root Regression Residuals')

plt.tight_layout()
plt.savefig('qq_plots.png')
plt.show()

# Check for heteroscedasticity with simple scatter plots instead of regplot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_mass, np.abs(residuals_mass), alpha=0.5)
plt.title('Heteroscedasticity Check - Mass Model')
plt.xlabel('Fitted Values')
plt.ylabel('Absolute Residuals')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_pred_cbrt_orig, np.abs(residuals_cbrt_orig), alpha=0.5)
plt.title('Heteroscedasticity Check - Cube Root Model')
plt.xlabel('Fitted Values')
plt.ylabel('Absolute Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heteroscedasticity_check.png')
plt.show()