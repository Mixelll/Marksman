import numpy as np
import pandas as pd
from ib_insync import *
from tzlocal import get_localzone
from marksman_extras import *
from marksman_objects import Ticker
from marksman_db import df_insert_prime
from postgresql_db import append_df_to_db
from datetime import datetime, timedelta, date





# IB TWS query:
def ticker_historical_data(ib, tickerPackage: Ticker, startDate, endDate, barSizeSetting,
                            whatToShow: float='TRADES', useRTH=True, timeout:float=600, **kwargs):
    contract = Stock(tickerPackage.symbol, tickerPackage.exchange, tickerPackage.currency)
    barSizeSetting = bars_size(barSizeSetting)
    if not endDate: endDateDT = datetime.now(get_localzone())
    else: endDateDT = endDate

    return ib.reqHistoricalData(contract, endDateTime=endDate, durationStr=duration(endDateDT-startDate, **kwargs),
                                barSizeSetting=barSizeSetting, whatToShow=whatToShow, useRTH=useRTH, timeout=timeout)


# query from IB TWS amd convert to pandas dataframe:
def ticker_historical_data_trades_populate_db(ib, tickers, barSizes, startDate, endDate, uploader, useRTH=None):
    day = timedelta(days=1)
    for barsSize in barSizes:
        barSizeStr = bars_size(barsSize)
        barSizeTD = td_parser(barSizeStr)

        if useRTH is None:
            if barSizeTD <= timedelta(minutes=30): useRTH, ORTH = False, ''
            else: useRTH, ORTH = True, ''
        else:
            if useRTH or barSizeTD<=timedelta(minutes=30): ORTH = ''
            else: ORTH = '_ORTH'

        for ticker in tickers:
            bars = ticker_historical_data(ib, Ticker(symbol=ticker), startDate, endDate, barsSize, useRTH=useRTH, timeout=60000000)
            df = util.df(bars)
            df['ib_is_date'] = df['date']
            # print(df)
            if np.issubdtype(df.dtypes[0], np.object):
                if isinstance(df.loc[0, 'date'], date):
                    df['date'] = df['date'].map(lambda x: datetime.combine(x, datetime.min.time())).dt \
                    .tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_backward')
                if barSizeTD >= timedelta(weeks = 1):
                    # firstDate = df['date'].iat[0]
                    df['date'] = df['date'].shift(periods=1)
                    # df['date'].iat[0] = firstDate - barSizeTD
                    df.drop(labels = 0, inplace = True)
            elif np.issubdtype(df.dtypes[0], np.datetime64):
                df['date'] = to_timezone(df['date'], get_localzone())
            # print(df)
            df['ny_date'] = df['date'].dt.tz_convert('America/New_York').dt.tz_localize(None)
            df['volume'] = df['volume'].map(lambda x: x*100)
            widths = df['date'].diff().shift(periods=-1)
            last = df['date'].iloc[-1]
            widths.iloc[-1] = min(to_timezone(datetime.now(), get_timezone(last)) - last , barSizeTD)
            df['duration'] = widths.map(lambda x: [x, barSizeTD][[x, 1.2*barSizeTD] \
            .index(min(x, 1.2*barSizeTD))]).map(lambda x: x.total_seconds())
            if barSizeTD>day:
                xSeconds = 365 * 24 * 3600
            elif barSizeTD==day:
                xSeconds = 51 * 5 * 24 * 3600
            elif barSizeTD<day:
                xSeconds = 51 * 5 * 6.5 * 3600
            df['close-open'] = df['close'] - df['open']
            df['high-low'] = df['high'] - df['low']
            df['percent'] = df['close-open'] / df['open']
            df['percent_high-low'] = 2* df['high-low'] / (df['high'] + df['low'])
            df['percent_year'] = xSeconds*df['percent'] / df['duration']
            df['money_moved'] = df['average'] * df['volume']
            df['close-open_volume'] = df['close-open'] * df['volume']
            # df['market_cap_change'] = 510 * 10**6 * df['open-close']
            df.set_index(['date'], inplace=True)
            print(tuple(df.index.names))
            print(df.dtypes)
            print(df)
            uploader(ticker + '_' + barSizeStr + ORTH, df)
