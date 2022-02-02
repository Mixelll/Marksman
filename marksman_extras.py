import math
import copy
import pytz
import types
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta as td
from pytz import timezone
from tzlocal import get_localzone
from math import log10, floor


def stat_dic(df):
    print(df.cov())
    print(df.describe())
    print(np.array_split(df,3))
    return {'pandas.DataFrame.cov' : df.cov().to_json(),
            'pandas.DataFrame.describe' : df.describe().to_json()}


def df_ind_col_op(df, column = None, op = min):
    try:
         return op(df[column])
    except:
         return op(df.index)


def iter2list(o):
    if (isinstance(o, range) or isinstance(o, types.GeneratorType)) and not isinstance(o, str):
        return list(iter(o))

    return o

def str2pd_interval(o, tz='America/New_York'):
    oo = o.split(',')
    f = lambda x : to_timezone(pd.Timestamp(x), pytz.timezone(tz))
    return pd.Interval(f(oo[0][1:]), f(oo[1][:-1]))

    # return o

def pd_interval2str(o):
    if isinstance(o, pd.Interval):
        oo = str(o)
        return oo[0] + str(o.left) + ',' + str(o.right) + oo[-1]

    return o


def iter_length(*args):
    out = []
    for x in args:
        if x is None: out.append(0)
        elif isinstance(x, str): out.append(1)
        else: out.append(len(list(copy.deepcopy(x))))

    return out


def multiply_iter(v, n=1):
    if isinstance(v, str): return [v]*n
    o = copy.deepcopy(v)

    try:
        iter(v)
        return o
    except TypeError:
        return [o]*n


def mbool(val, st=' ', useBool=True):
    if isinstance(val, str): return bool(val.strip(st))
    elif useBool: return bool(val)
    else: return True


def n_times(list, f):
    out = []
    for e in list:
        out.extend([e] * f)

    return out


def round_to(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def get_timezone(input):
    typeStr = str(type(input)).lower()
    if typeStr.find('pandas') != -1:
        try:
            out = input.dt.tz
        except:
            out = input.tz
    else: out = str(input.tzinfo.zone)

    return out


def to_timezone(input, tz_in, naive=False):
    typeStr = str(type(input)).lower()
    if typeStr.find('pandas') != -1:
        try:
            if input.dt.tz == None: out = input.dt.tz_localize(str(get_localzone()), ambiguous='infer',
                                                                nonexistent='shift_backward').dt.tz_convert(str(tz_in))
            else: out = input.dt.tz_convert(tz_in)
            if naive: out = out.dt.tz_localize(None)
        except:
            if input.tz == None: out = input.tz_localize(str(get_localzone()), ambiguous='infer',
                                                        nonexistent='shift_backward').tz_convert(str(tz_in))
            else: out = input.tz_convert(tz_in)
            if naive: out = out.tz_localize(None)
    else:
        if isinstance(tz_in, str): out = input.astimezone(timezone(tz_in))
        else: out = input.astimezone(tz_in)
        if naive: out = out.replace(tzinfo=None)

    return out


def epoch(timeQuants, timeQuantsVals, delta, **kwargs):
    timeQuants = [x for _, x in sorted(zip(timeQuantsVals, timeQuants), key=lambda pair: pair[0], reverse=True)]
    timeQuantsVals = sorted(timeQuantsVals, reverse=True)
    unit = kwargs.get('unit', '')
    if isinstance(unit, str) and unit: fInd = timeQuants.index(unit)
    else: fInd = -1

    if any(char.isdigit() for char in timeQuants[0]):
        a = [abs(x-delta) for x in timeQuantsVals]
        return timeQuants[a.index(min(a))]
    else:
        for i in range(0,len(timeQuants)):
            if (delta/timeQuantsVals[i]) >= 1:
                if fInd>i: IND = fInd;
                else: IND = i
                break
    prefix = str(math.ceil(delta/timeQuantsVals[IND])) + ' '

    return prefix + timeQuants[IND]


def td_parser(strList):
    dic = {'Y':td(days=365), 'M':td(days=30), 'W':td(days=7), 'D':td(days=1), 'H':td(hours=1), 'S':td(seconds=1),
        '1 secs':td(seconds=1),  '5 secs':td(seconds=5), '10 secs':td(seconds=10), '15 secs':td(seconds=15), '30 secs':td(seconds=30),
        '1 min':td(minutes=1), '2 mins':td(minutes=2), '3 mins':td(minutes=3), '5 mins':td(minutes=5), '10 mins':td(minutes=10),
        '15 mins':td(minutes=15), '20 mins':td(minutes=20), '30 mins':td(minutes=30), '1 hour':td(hours=1), '2 hours':td(hours=2),
        '3 hours':td(hours=3), '4 hours':td(hours=4), '8 hours':td(hours=8), '1 day':td(days=1), '1 week':td(weeks=1), '1 month':td(days=30)}
    # dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    if isinstance(strList, list): return [dic[st] for st in strList]
    else: return dic[strList]


def duration(delta, **kwargs):
    if isinstance(delta, str): return delta
    timeQuants = ['Y','M','W','D','H','S']

    return epoch(timeQuants, td_parser(timeQuants), delta, **kwargs)


def bars_size(delta, **kwargs):
    timeQuants = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins',
                '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours',
                '3 hours', '4 hours', '8 hours','1 day', '1 week', '1 month']
    if isinstance(delta, str): return delta

    return epoch(timeQuants, td_parser(timeQuants), delta, **kwargs)
