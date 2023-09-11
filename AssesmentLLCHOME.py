#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[92]:


# Load the data (you need to have a dataset with relevant features and the S&P Case-Shiller Home Price Index)
data = pd.read_csv('LLChome.csv')  


# In[93]:


print(data.dtypes)


# In[94]:


# Calculate the mean of a numerical column
numerical_mean = data['CSUSHPISA'].mean()
numerical_mean


# In[95]:


data['DATE'] = pd.to_datetime(data['DATE'], format='%d-%m-%Y')
data['Year'] = data['DATE'].dt.year


# In[96]:


print(f'Numerical Mean: {numerical_mean}')


# In[97]:


average_value = data['DATE'].mean()
print(f'Average Value: {average_value}')


# In[ ]:





# In[98]:


data


# In[99]:


# Data preprocessing (handle missing values, feature engineering, etc.)

# Split data into features (X) and target variable (y)
X = data[['Year']]
y = data['CSUSHPISA']


# In[100]:



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[101]:


# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[102]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[103]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[104]:


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')


# In[105]:


r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')


# In[106]:


# Plot actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Home Prices')
plt.ylabel('Predicted Home Prices')
plt.title('Actual vs. Predicted Home Prices')
plt.show()

