import pandas as pd
import mplfinance as mpf

file = 'dataset/FB.csv'
data = pd.read_csv(file)
#print(data.info())

# Convert object to datetime
data.Date = pd.to_datetime(data.Date)
#print(data.info())

# Set date column as index
data = data.set_index('Date')

print(mpf.plot(data, type='line', volume = True))

print(mpf.plot(data['2020-03']))
print(mpf.plot(data, type='candle', mav =(20), tight_layout=True, style='yahoo'))
