import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import mplfinance as mpf

# Get the stock data
#df = quandl.get("WIKI/FB")
df = pd.read_csv (r'dataset/FB.csv')
# Take a look at data
#print(df.head())

df2 = pd.read_csv (r'dataset/FB.csv')
#print(df2.head())

# Get the Adjusted Close
df = df[['Adj Close']]
# Take a look at new data
#print(df.head())

# A variable for predicting 'forecast_out' days out in future
forecast_out = 1

# Create another column (dependent variable) shifted 'forecast_out' units up
df['Prediction'] = df[["Adj Close"]].shift(-forecast_out)
#print(df.tail())

# Create the independent dataset
# Convert the dataframe to numpy array
X = np.array(df.drop(['Prediction'],1))
# Remove the last 'forecast_out' rows
X = X[:-forecast_out]
#print(X)

# Create the dependent dataset
# Convert the dataframe to numpy array (All of the values including the NaNs)
y = np.array(df['Prediction'])
# Get all of the y values except the last 'forecast_out' rows
y = y[:-forecast_out]
#print(y)

z = np.array(df2)
z = z[:-forecast_out]
print(z)

# Split the data into 80% training and 20% testing
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# train_set = x_train + y_train
# test_set = x_test + y_test

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Create and train the Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
train_scores = cross_val_score(svr_rbf, x_train, y_train, scoring='accuracy',cv=5)
test_scores = cross_val_score(svr_rbf, x_test, y_test, scoring='accuracy',cv=5)



#Set x_forecast equal to the last forecast_out rows of the original dat set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
#print(x_forecast)

# Print the SVR prediction for the next forecast_out days
svm_prediction = svr_rbf.predict(x_forecast)
print(f'The next {forecast_out} days by the SVR model: \n{svm_prediction}')




# Testing Model: Score returns the coefficient of determination R^2 of the prediction
# Best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svr confidence: ", svm_confidence)



## Creating predicted values for both models
y_pred_svr = svr_rbf.predict(x_test)


# Mean Squared Error for SVR
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f'The MSE for the SVR algorithm is: {mse_svr}')




