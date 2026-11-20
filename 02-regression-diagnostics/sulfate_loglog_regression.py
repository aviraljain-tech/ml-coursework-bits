import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Data
data = {
    'Hours': [2.0, 4.00, 6.00, 8.00, 10.00, 15.00, 20, 25, 30, 40, 50, 60, 70, 80, 90, 110, 130, 150, 160, 170, 180],
    'Sulfate': [15.11, 11.36, 9.77, 9.09, 8.48,7.69, 7.33, 7.06, 6.7, 6.43, 6.16, 5.99, 5.77, 5.64, 5.39, 5.09, 4.87, 4.6, 4.5, 4.36, 4.27]
}

df = pd.DataFrame(data)

# Displaying the first few rows of the DataFrame
print(df.head())
df['log_Hours'] = np.log(df['Hours'])
df['log_Sulfate'] = np.log(df['Sulfate'])
X = df[['log_Hours']]
y = df['log_Sulfate']

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept: {intercept}")
print(f"Slope: {slope}")

# Predicting values using the regression model
df['log_Sulfate_pred'] = model.predict(X)


plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_Hours', y='log_Sulfate', data=df, label='Data points')
sns.lineplot(x='log_Hours', y='log_Sulfate_pred', data=df, color='red', label='Regression line')
plt.xlabel('Log of Hours')
plt.ylabel('Log of Sulfate Concentration')
plt.title('Log-Log Plot of Sulfate Concentration vs. Time')
plt.legend()
plt.show()

# Exponentiate the predicted log values to get back to original scale
df['Sulfate_pred'] = np.exp(df['log_Sulfate_pred'])


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours', y='Sulfate', data=df, label='Data points')
sns.lineplot(x='Hours', y='Sulfate_pred', data=df, color='red', label='Regression curve')
plt.xlabel('Hours')
plt.ylabel('Sulfate Concentration')
plt.title('Sulfate Concentration vs. Time')
plt.legend()
plt.show()

# Calculate residuals
df['Residuals_log'] = df['log_Sulfate'] - df['log_Sulfate_pred']
df['Residuals'] = df['Sulfate'] - df['Sulfate_pred']

# Plot residuals in log-log coordinates
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_Sulfate_pred', y='Residuals_log', data=df)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Log Sulfate Concentration')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values in Log-Log Coordinates')
plt.show()

# Plot residuals in original coordinates
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sulfate_pred', y='Residuals', data=df)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Sulfate Concentration')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values in Original Coordinates')
plt.show()
