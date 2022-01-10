import numpy as np
import pandas as pd
import datetime
from datetime import date
from calendar import mdays
from utils import data, utils
from utils import functions as f

stock = "AAPL"
type = 'call'
interval = 'month'

if type == 'put':
    utils.update_call_options(stock, interval)
else:
    utils.update_put_options(stock, interval)
