import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt

# Import Data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Adj Close']
    pctChange = stockData.pct_change()
    meanReturns = pctChange.mean()
    covMatrix = pctChange.cov()
    return meanReturns, covMatrix, stockData, pctChange
