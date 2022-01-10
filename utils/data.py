import numpy as np
import pandas as pd
import datetime
from yahoo_fin import stock_info, options

def get_stock_data(ticker, start_date=None, end_date=None):
    # Wrapper for stock_info.get_data() to do some preprocessing.
    data = stock_info.get_data(ticker, start_date, end_date)
    data = {d: data['close'][d] for d in data.index}

    return data

def get_options_data(ticker, type, date=None):
    # Wrapper for options.get_calls() / options.get_puts() to do some preprocessing.
    if type not in ['call', 'put']:
        raise ValueError('Something is wrong! Option type can only be \'call\' or \'put\'!')

    if type == 'call':
        data = options.get_calls(ticker, date=date)
    else:
        data = options.get_puts(ticker, date=date)

    data = data.drop(labels=['Last Trade Date', 'Bid', 'Change', '% Change', 'Volume', 'Open Interest'], axis=1)
    data.set_index('Contract Name', inplace=True)
    data['Strike'] = np.array(data['Strike'])
    #convert %volatility to float (0->1)
    data['Implied Volatility'] = [s[:-1] for s in list(data['Implied Volatility'])]
    data['Implied Volatility'] = np.array(data['Implied Volatility']).astype(float) / 100

    return data
