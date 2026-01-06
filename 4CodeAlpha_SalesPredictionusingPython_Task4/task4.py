# TASK 4: Sales Prediction using Python
# ------------------------------------

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load the dataset
df = pd.read_csv(r"E:\CodeAlpha\Data_Science\4CodeAlpha_SalesPredictionusingPython_Task4\advertising.csv")   # change path if required


# 3. Data Cleaning
# Remove unwanted index column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Check missing values
print("Missing values:\n", df.isnull().sum())


# 4. Exploratory Data Analysis
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between Advertising Spend and Sales")
plt.show()


# 5. Feature Selection
X = df[['TV', 'Radio', 'Newspaper']]   # advertising platforms
y = df['Sales']                        # target variable


# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)


# 8. Predict Sales
y_pred = model.predict(X_test)


# 9. Model Evaluation
print("\nModel Performance:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


# 10. Advertising Impact Analysis
impact = pd.DataFrame({
    "Advertising Medium": X.columns,
    "Impact Coefficient": model.coef_
})

print("\nImpact of Advertising on Sales:")
print(impact)


# 11. Predict Future Sales
future_ads = pd.DataFrame({
    'TV': [200],
    'Radio': [40],
    'Newspaper': [20]
})

future_sales = model.predict(future_ads)
print("\nPredicted Future Sales:", future_sales[0])


# 12. Visualization: Actual vs Predicted Sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
