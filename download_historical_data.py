import pytz

import marksman_ib_queries
import marksman_db as mdb
import postgresql_db as db

from ib_insync import IB
from datetime import datetime

from postgresql_db import conn as SQLconn, engine as SQLengine

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# query from IB TWS and upload to DB:

tickers = ['AMZN', 'AAPL']
tickers = ['AMZN']

startDate = datetime(2021,10,8)
startDate = datetime(2022,2,15)
# startDate = pytz.timezone('America/New_York').localize(startDate)
endDate = ''
# endDate = pytz.timezone('America/New_York').localize(endDate)
# endDate = pytz.timezone('America/New_York').localize(endDate)

# print(to_timezone(startDate, pytz.timezone('America/New_York'), naive = True))

barSizes = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min',
            '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins',
            '30 mins', '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
            '1 day', '1 week', '1 month']
barSizes = ['10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours',
            '3 hours', '4 hours', '8 hours','1 day', '1 week', '1 month']
barSizes = ['30 mins']
barSizes.reverse()

# barSizes = ['1 month']
FORCEuseRTH = False

schema = 'trades'
def uploader(tableName, df):
    db.upsert_df_to_db(SQLengine, tableName, df, schema=schema)
    mdb.df_insert_prime(SQLconn, df, tableName, schema + '_prime', schema=schema)
# uploader = None
marksman_ib_queries.ticker_historical_data_trades_populate_db(ib, tickers, barSizes, startDate,
                                                                endDate, uploader, useRTH=FORCEuseRTH)
