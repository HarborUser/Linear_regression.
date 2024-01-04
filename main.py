
#Analyze the distribution of Clicks and Impressions over time.
#Calculate and visualize the Click-Through Rate (CTR) trend.
#Identify any patterns or trends in CTR based on day-of-week and other features.
#Create relevant features for CTR analysis.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

#Build a forecasting model to predict future CTR values.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("whitegrid")

df = pd.read_csv("ctr.csv")

data_stats = df.describe()
print(data_stats)

cols = ['Date', 'Clicks', 'Impressions']


sns.relplot( x='Clicks', y='Impressions',data=df)

plt.title("Click Rate")
ctr = df['Clicks']/df['Impressions']
df.insert(3,column="CTR",value=ctr)
features = ["Clicks","Impressions"]
target = 'CTR'




X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

model = LinearRegression()



# Training the model
model.fit(X_train, y_train)

y_pred_lr = model.predict(X_test)

# Visualization: Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicton')
plt.title('Actual CTR vs. Predicted CTR')
plt.show()


plt.show()


