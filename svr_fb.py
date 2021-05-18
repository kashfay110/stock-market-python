import numpy
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import mplfinance as mpf
import os

# Get the stock data
#df = quandl.get("WIKI/FB")
pathname = r'dataset/AML.L.csv'
df = pd.read_csv (pathname)

# Grab the filename from the path
filename = os.path.basename(pathname)
filename_grab = os.path.splitext(filename)[0]

# Get the Adjusted Close
df = df[['Adj Close']]

# A variable for predicting 'forecast_out' days out in future
forecast_out = 5

# Create another column (dependent variable) shifted 'forecast_out' units up
df['Prediction'] = df[["Adj Close"]].shift(-forecast_out)

# Create the independent dataset
# Convert the dataframe to numpy array
X = np.array(df.drop(['Prediction'],1))
# Remove the last 'forecast_out' rows
X = X[:-forecast_out]


# Create the dependent dataset
# Convert the dataframe to numpy array (All of the values including the NaNs)
y = np.array(df['Prediction'])
# Get all of the y values except the last 'forecast_out' rows
y = y[:-forecast_out]

print('\n')

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#K-Fold
cv_kfold = KFold(n_splits=5, random_state=1, shuffle=True)

# Create and train the Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
cv_scores_svr = cross_val_score(svr_rbf, X, y,cv=cv_kfold)
kfold_final_svr = cv_scores_svr.mean()
print(f'Final KFold Average for SVR: {kfold_final_svr}')

# Create and train the Linear Regression model
lr = LinearRegression()
# Train model
lr.fit(x_train, y_train)
cv_scores_lr = cross_val_score(lr, X, y,cv=cv_kfold)
kfold_final_lr = cv_scores_lr.mean()
print(f'Final KFold Average for LR: {kfold_final_lr}')

print('\n')

#Set x_forecast equal to the last forecast_out rows of the original dat set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]

# Print the SVR prediction for the next forecast_out days
svm_prediction = svr_rbf.predict(x_forecast)
print(f'Prediction of the Adj. Close price for the next {forecast_out} day(s) by the SVR model: \n{svm_prediction}')

# Print the Linear Regression prediction for the next forecast_out days
lr_prediction = lr.predict(x_forecast)
print(f'Prediction of the Adj. Close price for the next {forecast_out} day(s) by the LR model: \n{lr_prediction}')
print('\n')

# Testing Model: Score returns the coefficient of determination R^2 of the prediction
# Best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svr confidence: ", svm_confidence)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction
# It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model
# Best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

print('\n')

## Creating predicted values for both models
y_pred_svr = svr_rbf.predict(x_test)
y_pred_lr = lr.predict(x_test)

# Mean Squared Error for SVR
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f'The MSE for the SVR algorithm is: {mse_svr}')

# Mean Squared Error for LR
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'The MSE for the LR algorithm is: {mse_lr}')
print('\n')

# Root Mean Squared Error for SVR
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
print(f'The RMSE for the SVR algorithm is: {rmse_svr}')

# Root Mean Squared Error for LR
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
print(f'The RMSE for the LR algorithm is: {rmse_lr}')
print('\n')

# Mean Absolute Error for SVR
mae_svr = mean_absolute_error(y_test, y_pred_svr)
print(f'The MAE for the SVR algorithm is: {mae_svr}')

# Mean Absolute Error for LR
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f'The MAE for the LR algorithm is: {mae_lr}')
print('\n')

# Mean Absolute Percentage Error for SVR
mape_svr = mean_absolute_percentage_error(y_test, y_pred_svr)
print(f'The MAPE for the SVR algorithm is: {mape_svr}')

# Mean Absolute Percentage Error for LR
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
print(f'The MAPE for the LR algorithm is: {mape_lr}')


n_array = numpy.array([])
new_array = numpy.append(n_array, [kfold_final_svr, kfold_final_lr, mse_svr, mse_lr, rmse_svr, rmse_lr, mae_svr, mae_lr, mape_svr, mape_lr])
# print(new_array)

new_df = pd.DataFrame (new_array)
## save to xlsx file

filepath = f'results/Consumer Discretionary/{filename_grab}_results.xlsx'
new_df.to_excel(filepath, index=False)