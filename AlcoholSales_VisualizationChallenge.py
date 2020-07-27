# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:34:22 2020

@author: anushree
"""

# Project Team Name: Insights
# Participant Name: Anushree Chopde
# Project: Visualization challenge
# The visualization consists of Alcohol Sales Dataset
# Data Source: Kaggle.com (https://www.kaggle.com/bulentsiyah/for-simple-exercises-time-series-forecasting)
# I am going to visualize and analyse the dataset and draw my insights

# 1) Import libraries
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import datetime

# 2) Import data
Alcohol_Sales = pd.read_csv("Alcohol_Sales.csv", index_col="DATE", parse_dates=True)

Alcohol_Sales.index.freq = "MS"
print(Alcohol_Sales.head())


# 3) Vizualizations

# Vizualization 1

Alcohol_Sales.columns = ['Sales']

viz1 = Alcohol_Sales.plot(figsize=(12,8))

# Visualization 2
from statsmodels.tsa.seasonal import seasonal_decompose

viz2 = seasonal_decompose(Alcohol_Sales['Sales'])
viz2.observed.plot(figsize=(12,2))

viz2.trend.plot(figsize=(12,2))

viz2.seasonal.plot(figsize=(12,2))

viz2.resid.plot(figsize=(12,2))

# Train-Test Split for forecasting

print("len(Alcohol_sales)",len(Alcohol_Sales))
train = Alcohol_Sales.iloc[:310]
test = Alcohol_Sales.iloc[310:]
print("length of train data",len(train))
print("length of test data",len(test))

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# 4) Time Series generator
from keras.preprocessing.sequence import TimeseriesGenerator
scaled_train[0]

# define generator
n_input = 2
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

print('length of scaled_train data',len(scaled_train))
print('len(generator)',len(generator))

# First batch
x,y = generator[0]
print(f'Given the array: {x.flatten()} \nPredict y: {y}')

# Redefining n_input to 12 to predict the next month sales by going 12 months back
n_input = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# First batch
x,y = generator[0]
print(f'Given the array: {x.flatten()} \nPredict y: {y}')

# 5) Creating the model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Defining Model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='Adam', loss='mse')
print(model.summary())

model.fit_generator(generator, epochs=55)

model.history.history.keys()
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

first_batch_eval = scaled_train[-12:]
print(first_batch_eval)

first_batch_eval = first_batch_eval.reshape((1,n_input,n_features))
model.predict(first_batch_eval)

scaled_test[0]

predict_test = []
first_batch_eval = scaled_train[-n_input:]
current_batch = first_batch_eval.reshape((1,n_input,n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    predict_test.append(current_pred)
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1) # update batch to now include prediction and drop first value
    
print(predict_test)

print(scaled_test)

actual_predictions = scaler.inverse_transform(predict_test)
print(actual_predictions)

test['Predictions'] = actual_predictions
print(test)

test.plot(figsize=(12,8))

