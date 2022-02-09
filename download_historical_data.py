from ib_insync import *
from marksman_objects import *
from datetime import datetime
# import numpy as np
# import pandas as pd
from marksman_ib_queries import *
from postgresql_db import *
import pytz


ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# query from IB TWS and upload to DB:

tickers = ['AMZN', 'AAPL']
tickers = ['AAPL']

startDate = datetime(2021,10,8)
# startDate = pytz.timezone('America/New_York').localize(startDate)
endDate = datetime(2021,11,12)
# endDate = pytz.timezone('America/New_York').localize(endDate)
# endDate = pytz.timezone('America/New_York').localize(endDate)

# print(to_timezone(startDate, pytz.timezone('America/New_York'), naive = True))

barSizes = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins'\
, '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours'\
, '3 hours', '4 hours', '8 hours','1 day', '1 week', '1 month']
barSizes = ['10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours'\
, '3 hours', '4 hours', '8 hours','1 day', '1 week', '1 month']
barSizes = ['30 mins']
barSizes.reverse()

# barSizes = ['1 month']
FORCEuseRTH = False

schema = 'trades'
# uploader = lambda tableName, df: append_df_to_db(SQLengine, tableName, df) upsert_df_to_db(SQLengine, tableName, df, schema=schema),
uploader = lambda tableName, df: [
                                    df_insert_prime(SQLconn, tableName, df, schema=schema),
                                    set_comment(SQLconn, tableName, datetime.now(), schema=schema),]
# uploader = None
ticker_historical_data_trades_populate_db(ib, tickers, barSizes, startDate, endDate, uploader, useRTH=FORCEuseRTH)
