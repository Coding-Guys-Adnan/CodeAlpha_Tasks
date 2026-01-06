# ==========================================
# TASK 2: Unemployment Analysis with Python
# Using TWO CSV files
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. Load BOTH datasets
# ------------------------------------------

df1 = pd.read_csv(r"E:\CodeAlpha\Data_Science\Task2\Unemployment_in_India.csv")
df2 =pd.read_csv(r"E:\CodeAlpha\Data_Science\Task2\Unemployment_Rate_upto_11_2020.csv")

print("Dataset 1 Preview:")
print(df1.head())

print("\nDataset 2 Preview:")
print(df2.head())

# ------------------------------------------
# 2. Data Cleaning (BOTH datasets)
# ------------------------------------------

# Clean column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Rename unemployment rate column
df1.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate'
}, inplace=True)

df2.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate'
}, inplace=True)

# Convert Date column to datetime
df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])

# Drop missing values
df1.dropna(inplace=True)
df2.dropna(inplace=True)

# ------------------------------------------
# 3. Combine BOTH datasets
# ------------------------------------------

# Concatenate datasets
df = pd.concat([df1, df2], ignore_index=True)

# Sort by date
df = df.sort_values('Date')

print("\nCombined Dataset Info:")
print(df.info())

# ------------------------------------------
# 4. Overall Unemployment Trend
# ------------------------------------------

plt.figure()
plt.plot(df['Date'], df['Unemployment Rate'])
plt.title("Overall Unemployment Rate Trend (India)")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ------------------------------------------
# 5. Covid-19 Impact Analysis
# ------------------------------------------

covid_start = "2020-03-01"
covid_end = "2021-12-31"

df['Period'] = "Pre-Covid"
df.loc[(df['Date'] >= covid_start) & (df['Date'] <= covid_end), 'Period'] = "Covid"
df.loc[df['Date'] > covid_end, 'Period'] = "Post-Covid"

# Average unemployment by period
covid_analysis = df.groupby('Period')['Unemployment Rate'].mean()

print("\nAverage Unemployment Rate by Period:")
print(covid_analysis)

# Covid impact visualization
plt.figure()
sns.boxplot(x='Period', y='Unemployment Rate', data=df)
plt.title("Impact of Covid-19 on Unemployment Rate")
plt.show()

# ------------------------------------------
# 6. Seasonal Trend Analysis
# ------------------------------------------

df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby('Month')['Unemployment Rate'].mean()

plt.figure()
plt.plot(monthly_avg.index, monthly_avg.values)
plt.title("Seasonal Trend of Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ------------------------------------------
# 7. Key Insights
# ------------------------------------------

print("\nKEY INSIGHTS:")
print("1. Unemployment rose sharply during the Covid-19 period.")
print("2. Combining both datasets shows a clear pre and post Covid comparison.")
print("3. Monthly unemployment trends indicate seasonal employment patterns.")