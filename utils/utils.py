import numpy as np
import pandas as pd
import datetime
from datetime import date
from calendar import mdays
from dateutil.relativedelta import relativedelta
from utils import data
from utils import functions as f

"""
Manipulations on stock data mainly;
Stock data received as a dictionary where data.keys() -> timestamps
and data.values() are corresponding closing prices.
"""

today = datetime.date.today() # Current date
r = 0.04 # Rough assumption for risk-free interest considering Treasury bills

def find_yearly_last_entry(data):
    # Raw stock data come with day timestamps; find the last stamp per year.
    dates = list(data.keys())
    close_dates = []
    for i, curr in enumerate(dates):
        if i > 0:
            prev = dates[i-1]
            if curr.year != prev.year:
                close_dates.append(prev)
    return close_dates

def get_daily_returns(data):
    #Daily logarithmic returns --> log(close/open)
    dates = list(data.keys())
    no_dates = len(dates)
    returns = {}
    window = 21 #how many days are used to calculate returns
    for i in range(no_dates-window-1, no_dates-1):
        day = dates[i+1].day
        open = data[dates[i]]
        close = data[dates[i+1]]
        returns[dates[i+1]] = np.log(close/open)
    return returns

def get_yearly_returns(data):
    #Yearly logarithmic returns --> log(close/open)
    # Assumes we have data spanning at least one full calendar year.
    close_dates = find_yearly_last_entry(data)
    no_years = len(close_dates)
    returns = {}
    for i in range(no_years-1):
        year = close_dates[i+1].year
        returns[year] = np.log(data[close_dates[i+1]] / data[close_dates[i]])
    return returns

def volatility_day(returns):
    #Returns the standard deviation of daily stock returns
    return np.std(returns)

def volatility_year(returns):
    #Same as above but refers to yearly volatility
    #Assumes data over the last year and 252 trading days
    return np.sqrt(252) * np.std(returns)

def find_expiration_date(interval):
    #Calculate option expiration date and days to expiration
    if interval == 'month':
        delta = relativedelta(months=1)
    elif interval == 'quarter':
        delta = relativedelta(months=3)
    elif interval == 'year':
        delta = relativedelta(years=1)
    elif interval == 'two_years':
        delta = relativedelta(years=2)
    else:
        raise ValueError('Wrong interval! Value can be \'month\', \'quarter\', \'year\' or \'two_years\'!')

    expiration_date = today + delta
    expiration_date = expiration_date + datetime.timedelta((4-expiration_date.weekday()) % 7) #next friday after 1 month passes
    days_to_expiration = (expiration_date - today).days
    tau = days_to_expiration / 365

    return expiration_date, days_to_expiration, tau

def option_spread(bs_price, ask_price):
    return ((bs_price - ask_price) / ask_price) * 100

def get_stock_price(ticker):
    # Get current stock info
    last_year = today + relativedelta(years=-1)
    stock_data = data.get_stock_data(ticker, start_date=last_year) #get the data for given stock for 1Y
    current_price = list(stock_data.values())[-1]
    stock_returns = get_daily_returns(stock_data)
    sigma = volatility_year(list(stock_returns.values())) #calculate the asset's Y volatility

    return current_price, sigma

def options_metrics(ticker, type, interval):
    #Calculate BS metrics for given options

    # Find current price and historic volatility
    current_price, sigma = get_stock_price(ticker)
    # Find time variables
    expiration_date, days_to_expiration, tau = find_expiration_date(interval)
    # Get options data
    options = data.get_options_data(ticker, type, expiration_date)

    strike = list(options['Strike'])

    if type == 'call':
        bs = f.black_scholes_call(current_price, strike, r, tau, sigma)
        greeks = f.greeks_call(current_price, strike, r, tau, sigma)
    else:
        bs = f.black_scholes_put(current_price, strike, r, tau, sigma)
        greeks = f.greeks_put(current_price, strike, r, tau, sigma)

    spread = option_spread(bs, list(options['Last Price']))
    options.insert(3, 'BS', bs, True)
    options.insert(4, 'Spread (%)', spread, True)
    options.insert(5, 'delta', greeks[0], True)
    options.insert(6, 'gamma', greeks[1], True)
    options.insert(7, 'vega', greeks[2], True)
    options.insert(8, 'theta', greeks[3], True)
    options.insert(9, 'rho', greeks[4], True)

    options = options.rename(columns={'Last Price' : 'Last',
                                    'Implied Volatility' : 'IV'})

    print(options)
    print(f'{current_price = }')
    print(f'{sigma = }')
    print(f'{expiration_date = }')
    print(f'{days_to_expiration = }')

def update_call_options(ticker, interval):
    # Get call options info
    options_metrics(ticker, 'call', interval)

def update_put_options(ticker, interval):
    # Get call options info
    options_metrics(ticker, 'put', interval)
