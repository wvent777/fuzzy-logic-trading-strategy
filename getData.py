import pandas as pd
import yfinance as yf
import os
from tabulate import tabulate
import datetime as dt
from pandas_datareader import data as pdr
yf.pdr_override()

pd.options.display.max_columns = 500
TRAINING_TICKERS = ['NVDA', 'AMZN', 'GOOG', 'TSLA', 'AAPL']
TESTING_TICKERS = ['META', 'GOOGL', 'TSM']
# LOWER_DATE = '2017-01-01'  Test
# UPPER_DATE = '2020-01-01'  test
LOWER_DATE = dt.datetime.now()
UPPER_DATE = LOWER_DATE - dt.timedelta(days=90)
for ticker in TRAINING_TICKERS + TESTING_TICKERS:
    if not os.path.exists('Data/' + ticker + '.csv'):
        data = yf.download(ticker, start=LOWER_DATE, end=UPPER_DATE)
        data.to_csv('Data/' + ticker + '.csv')

data = pd.read_csv('Data/' + TRAINING_TICKERS[0] + '.csv')
print(tabulate(data[0:10], headers='keys', tablefmt='psql'))


# Import Data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Adj Close']
    pctChange = stockData.pct_change()
    meanReturns = pctChange.mean()
    covMatrix = pctChange.cov()
    return meanReturns, covMatrix, stockData, pctChange


# endDate = dt.datetime.now()
# startDate = endDate - dt.timedelta(days=90)
#
# test, _, data, _ = get_data(TRAINING_TICKERS, startDate, endDate)
# print(data)
# data = pd.DataFrame(data)
# print(tabulate(data, headers='keys', tablefmt='psql'))
