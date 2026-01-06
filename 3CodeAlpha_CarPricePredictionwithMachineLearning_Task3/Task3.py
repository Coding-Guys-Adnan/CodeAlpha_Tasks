# ---------------------------------------------
# CAR PRICE PREDICTION USING MACHINE LEARNING
# ---------------------------------------------

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load Dataset
df = pd.read_csv(r"E:\CodeAlpha\Data_Science\3CodeAlpha_CarPricePredictionwithMachineLearning_Task3/car data.csv")



print("First 5 rows:")
print(df.head())


# 3. Dataset Information
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# 4. Feature Engineering
# Convert Year to Car Age
df['Car_Age'] = 2024 - df['Year']
df.drop('Year', axis=1, inplace=True)

# Drop Car Name (not useful for prediction)
df.drop('Car_Name', axis=1, inplace=True)


# 5. Encode Categorical Variables
df = pd.get_dummies(df, drop_first=True)

print("\nProcessed Data:")
print(df.head())


# 6. Split Features & Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 8. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)


# 9. Prediction
y_pred = model.predict(X_test)


# 10. Model Evaluation
print("\nModel Performance:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


# 11. Visualization
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()


# 12. Predict Price for a Sample Car
sample_car = X.iloc[0:1]
predicted_price = model.predict(sample_car)

print("\nPredicted Price for Sample Car:", predicted_price[0])