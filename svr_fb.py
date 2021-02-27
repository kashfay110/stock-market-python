import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Get the stock data
#df = quandl.get("WIKI/FB")
df = pd.read_csv (r'dataset/FB.csv')
# Take a look at data
#print(df.head())

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

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# train_set = x_train + y_train
# test_set = x_test + y_test

# Create and train the Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Create and train the Linear Regression model
lr = LinearRegression()
# Train model
lr.fit(x_train, y_train)

#Set x_forecast equal to the last forecast_out rows of the original dat set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
#print(x_forecast)

# Print the SVR prediction for the next forecast_out days
svm_prediction = svr_rbf.predict(x_forecast)
print(f'The next {forecast_out} days by the SVR model: \n{svm_prediction}')

# Print the Linear Regression prediction for the next forecast_out days
lr_prediction = lr.predict(x_forecast)
print(f'The next {forecast_out} days by the LR model: \n{lr_prediction}')

# Testing Model: Score returns the coefficient of determination R^2 of the prediction
# Best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svr confidence: ", svm_confidence)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction
# It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model
# Best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

## Creating predicted values for both models
y_pred_svr = svr_rbf.predict(x_test)
y_pred_lr = lr.predict(x_test)

# Mean Squared Error for SVR
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f'The MSE for the SVR algorithm is: {mse_svr}')

# Mean Squared Error for LR
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'The MSE for the LR algorithm is: {mse_lr}')

# Root Mean Squared Error for SVR
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
print(f'The RMSE for the SVR algorithm is: {rmse_svr}')

# Root Mean Squared Error for LR
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
print(f'The RMSE for the LR algorithm is: {rmse_lr}')

# Mean Absolute Error for SVR
mae_svr = mean_absolute_error(y_test, y_pred_svr)
print(f'The MAE for the SVR algorithm is: {mae_svr}')

# Mean Absolute Error for LR
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f'The MAE for the LR algorithm is: {mae_lr}')

# Mean Absolute Percentage Error for SVR
mape_svr = mean_absolute_percentage_error(y_test, y_pred_svr)
print(f'The MAPE for the SVR algorithm is: {mape_svr}')

# Mean Absolute Percentage Error for LR
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
print(f'The MAPE for the LR algorithm is: {mape_lr}')

##plot the models on a graph to see which has the best fit to original data
##correct missing values
# plt.figure(figsize=(16,8))
# plt.scatter(##, y_test, color='red', label='Data')
# plt.plot(##, svr_rbf.predict(##), color='green', label='SVR RBF Model')
# plt.legend()
# plt.show()