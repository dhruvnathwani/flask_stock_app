import pandas as pd
import requests
import json
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import time

def get_current_price(stock):

  url = 'https://www.alphavantage.co/query?'

  function = 'TIME_SERIES_INTRADAY'

  interval = '1min'

  symbol = str(stock)

  key = '2UZM8DILNESN9BB1'

  request = requests.get(url +
                               'function=' + function +
                               '&symbol=' + symbol +
                               '&interval=' + interval +
                               '&apikey='+ key)



  # need to reformat the function in order to comnbine it to have an index to parse the json tree with
  function_reformatted = function.split("_")

  function_reformatted = str(function_reformatted[0]) + " " + str(function_reformatted[1])

  function_reformatted = function_reformatted.title()

  function_reformatted = function_reformatted + " (" + str(interval) + ")"

  request = request.json()

  request = json.dumps(request,sort_keys = True, indent = 4)

  request = json.loads(request)

  top_node = request[function_reformatted]

  # get a list of all the indexes

  indexer = []

  for x in top_node:
    indexer.append(x)

  # Use the first indexer value to access the most recent price
  current_stock_price = top_node[indexer[0]]['1. open']


  return current_stock_price

def get_previous_prices(stock):

  url = 'https://www.alphavantage.co/query?'

  function = 'TIME_SERIES_INTRADAY'

  interval = '1min'

  symbol = str(stock)

  key = '2UZM8DILNESN9BB1'

  request = requests.get(url +
                               'function=' + function +
                               '&symbol=' + symbol +
                               '&interval=' + interval +
                               '&apikey='+ key)



  # need to reformat the function in order to comnbine it to have an index to parse the json tree with
  function_reformatted = function.split("_")

  function_reformatted = str(function_reformatted[0]) + " " + str(function_reformatted[1])

  function_reformatted = function_reformatted.title()

  function_reformatted = function_reformatted + " (" + str(interval) + ")"

  request = request.json()

  request = json.dumps(request,sort_keys = True, indent = 4)

  request = json.loads(request)

  top_node = request[function_reformatted]

  # get a list of all the indexes

  indexer = []

  for x in top_node:
    indexer.append(x)

  # Use the first indexer value to access the most recent price

  recent_prices = []

  for x in indexer:
    price = top_node[x]['1. open']

    recent_prices.append(price)


  return recent_prices

def predict_next_price(stock):

  recent_prices = get_previous_prices(stock)

  shape = len(recent_prices)

  df = np.array(recent_prices)

  df = np.flip(df)

  df = pd.DataFrame(df.reshape(shape), columns = ['Prices'])

  df['Predictions'] = df['Prices'].shift(-1)

  # Now we need to create the x variable

  x = np.array(df.drop(['Predictions'],1))

  x = x[:-1]

  # Now we need to create the y variable

  y = np.array(df['Predictions'])

  y = y[:-1]

  # Split the data into training and testing (80/20)

  x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
  # Create and train the models we will be using

  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  svr_rbf.fit(x_train,y_train)
  svm_confidence = float(svr_rbf.score(x_test,y_test))


  lr = LinearRegression()
  lr.fit(x_train,y_train)
  lr_confidence = float(lr.score(x_test,y_test))

  svr_ridge = SVR(kernel = 'linear', C = 1e3, gamma = 0.1)
  svr_ridge.fit(x_train, y_train)
  svr_ridge_confidence = float(svr_ridge.score(x_test,y_test))


  # Choose the correct model to use by deciding which has the higher r^2

  optimal = float(max(svr_ridge_confidence, lr_confidence, svm_confidence))

  if optimal == svm_confidence:
    method = svr_rbf
    method_used = 'SVR RBF'
  elif optimal == lr_confidence:
    method = lr
    method_used = 'Linear Regression'
  elif optimal == svr_ridge_confidence:
    method = svr_ridge
    method_used = 'SVR Ridge'

  #print("Method Used: " + str(method_used))

  # set x_forecast equal to the last 'n' forecast rows of the original data set from Prices column

  # In simpler terms, find the next value we want to forecast, as this will be the only real value without a prediction next to it as the algorithm can not give predictions for values that have not occurred

  x_forecast = np.array(df.drop(['Predictions'],1))[-1:]

  current_price = x_forecast

  # Create prediction for the next stock price

  predicted_price = method.predict(x_forecast)

  #results = {
  #    'currentPrice':current_price,
  #    'predictedPrice': predicted_price,
  #    'methodUsed': method_used
  #}

  return current_price, predicted_price, method_used