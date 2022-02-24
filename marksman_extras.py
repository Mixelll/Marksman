import copy
import json
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from math import log10, floor, ceil
from pytz import timezone
from tzlocal import get_localzone


def df_freq_dict(df, index = 'date'):
    freqs = ['20Y','10Y', '5Y', '3Y', 'Y', 'Q', 'M', 'W', '2D', 'D', '3H', 'H', '15m', '5m', '1m', '15s', '5s', '1s']
    deltas = td_parser(freqs)
    dfInd = df_return_ind_col(df, index)
    splitter = df_create_freq_splitter(df, index)

    a = [abs(max(dfInd)-min(dfInd) - x) for x in deltas]
    ind1 = a.index(min(a))
    a = [abs(dfInd.to_series().diff().value_counts().index[0] - x) for x in deltas]
    ind2 = a.index(min(a))
    split = []
    statDict = {}
    # print(datetime.now())
    for i in range(ind1+1,ind2-1):
        print(freqs[i])
        print(deltas[ind1]/deltas[i])
        if deltas[ind1]/deltas[i] <= 5000:
            split = splitter(freqs[i])
        if split:
            statDict = dict_merge(statDict, stat_dict(clean_split(split, index), index, arrayOp=lambda x: {freqs[i]: x}))

    return statDict

def clean_split(split, index=None, num = 0.6):
    # print(split)
    split_clean = [x for x in split if len(x)]

    split_len = [len(x) for x in split_clean]
    split_cleanest = []
    keep = None
    le_mean = sum(split_len) / len(split_len)
    if num < 1:
        num *= le_mean

    for x in split_clean:
        if keep is not None:
            if len(split_cleanest):
                dfIndK = df_return_ind_col(keep, index)
                dfIndX = df_return_ind_col(x, index)
                dfIndPrev = df_return_ind_col(split_clean[-1], index)
                distanceXK = min(dfIndX) - max(dfIndK)
                distanceKP = min(dfIndK) - max(dfIndPrev)
                if distanceXK < distanceKP:
                    x = pd.concat([keep, x])
                else:
                    split_cleanest[-1] = pd.concat([split_cleanest[-1], keep])
            else:
                x = pd.concat([keep, x])

        if len(x) < num:
            keep = x
        else:
            split_cleanest.append(x)
            keep = None

    # print([len(x) for x in split])
    # print([len(x) for x in split_cleanest])
    return split_cleanest

def df_create_freq_splitter(df, index=None):
    if index is None:
        index = df.index.names[0]
    if df_check_ind(df, index):
        return lambda f: [i for _,i in df.groupby(pd.Grouper(level=index, freq=f))]
    else:
        return lambda f: [i for _,i in df.groupby(pd.Grouper(key=index, freq=f))]



def stat_dict(dfArray, index=None, arrayOp = None):
    if arrayOp:
        def intervals(op): return arrayOp(df_interval_split_op(dfArray, op, index=index))
    else:
        def intervals(op): return df_interval_split_op(dfArray, op, index=index)

    return {'pandas.DataFrame.cov':  intervals(lambda x: x.cov()),
            'pandas.DataFrame.describe': intervals(lambda x: x.describe())}

def df_interval_split_op(dfArray, op, index=None):
    out = {}
    for df in dfArray:
        if len(df):
            dfInd = df_return_ind_col(df, index)
            out[pd.Interval(min(dfInd), max(dfInd))] = op(df)
    return out


def df_return_ind_col(df, index=None):
    if df_check_ind(df, index):
        return df.index
    else:
        return df[index]

def df_check_ind(df, index):
    if index in df.index.names:
        return True
    else:
        try:
            df[index]
            return False
        except:
            raise ValueError(f'Name {index} not found in indices or column names')


def iter2list(o):
    # if (isinstance(o, range) or isinstance(o, types.GeneratorType)) and not isinstance(o, str):
    if not isinstance(o, str):
        return list(iter(o))
    return o


def str2pd_interval(o, tz='America/New_York'):
    if o[0] not in '([' or  o[-1] not in '])':
        raise ValueError(f'Object {str(o)} of {str(type(o))} does not represent a pd.Interval')
    oo = o.split(',')
    def f(x): return to_timezone(pd.Timestamp(x), timezone(tz))
    return pd.Interval(f(oo[0][1:]), f(oo[1][:-1]))

    # return o

def pd_interval2str(o):
    if isinstance(o, pd.Interval):
        oo = str(o)
        return oo[0] + str(o.left) + ',' + str(o.right) + oo[-1]
    return o

def element_array_merge(a, b):
    if not isinstance(a, list):
        a = [a]
    if not isinstance(b, list):
        b = [b]

    a.extend(b)
    return a

def dict_merge(a, b, m=0):
    "merges b into a"
    for key in b:
        if key in a:
            aD, bD = isinstance(a[key], dict), isinstance(b[key], dict)
            if a[key] is None:
                a[key] = b[key]
            elif aD and bD:
                dict_merge(a[key], b[key])
            elif aD or bD:
                if m // 10 == 0:
                    if m % 10 == 0:
                        a[key] = b[key]
                    elif m % 10 == 1 and a[key] != b[key]:
                            a[key] = element_array_merge(a[key], b[key])
                    elif m % 10 == 2:
                            a[key] = element_array_merge(a[key], b[key])
                elif m // 10 == 1 and not aD and bD:
                        a[key] = b[key]
            else:
                if m % 10 == 0:
                    a[key] = b[key]
                elif m % 10 == 1 and a[key] != b[key]:
                        a[key] = element_array_merge(a[key], b[key])
                elif m % 10 == 2:
                        a[key] = element_array_merge(a[key], b[key])
        else:
            a[key] = b[key]
    return a

def object_captain_hook(o, default = [(str2pd_interval, pd.DataFrame.from_dict), str2pd_interval]):
    if callable(default):
        default = [default]

    for d in default:
        if callable(d) or len(d)==1:
            if not callable(d):
                d = d[0]
            od = {}
            for k,v in o.items():
                k0 = k
                try:
                    k = d(k)
                except:
                    pass
                try:
                    v = d(v)
                except:
                    pass
                try:
                    od[k] = v
                except:
                    od[k0] = v
        else:
            od = {}
            for k,v in o.items():
                try:
                    kd =  d[0](k)
                    if kd:
                        od[kd] = d[1](v)
                    else:
                        od[k] = v
                except:
                    od[k] = v
        o = od
    return od

def iter_length(*args):
    out = []
    for x in args:
        if x is None:
            out.append(0)
        elif isinstance(x, str):
            out.append(1)
        else:
            out.append(len(list(copy.deepcopy(x))))
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
    if isinstance(val, str):
        return bool(val.strip(st))
    elif useBool:
        return bool(val)
    return True


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
    else:
        out = str(input.tzinfo.zone)
    return out


def to_timezone(input, tz_in, naive=False):
    typeStr = str(type(input)).lower()
    if typeStr.find('pandas') != -1:
        try:
            if input.dt.tz == None:
                out = input.dt.tz_localize(str(get_localzone()), ambiguous='infer',
                                            nonexistent='shift_backward').dt.tz_convert(str(tz_in))
            else:
                out = input.dt.tz_convert(tz_in)
            if naive:
                out = out.dt.tz_localize(None)
        except:
            if input.tz == None:
                out = input.tz_localize(str(get_localzone()), ambiguous='infer',
                                        nonexistent='shift_backward').tz_convert(str(tz_in))
            else:
                out = input.tz_convert(tz_in)
            if naive:
                out = out.tz_localize(None)
    else:
        if isinstance(tz_in, str):
            out = input.astimezone(timezone(tz_in))
        else:
            out = input.astimezone(tz_in)
        if naive:
            out = out.replace(tzinfo=None)
    return out


def epoch(timeQuants, timeQuantsVals, delta, unit=None):
    timeQuants = [x for _, x in sorted(zip(timeQuantsVals, timeQuants), key=lambda pair: pair[0], reverse=True)]
    timeQuantsVals = sorted(timeQuantsVals, reverse=True)
    fInd = timeQuants.index(unit) if unit else -1

    if any(char.isdigit() for char in timeQuants[0]):
        a = [abs(x-delta) for x in timeQuantsVals]
        return timeQuants[a.index(min(a))]
    else:
        for i in range(0,len(timeQuants)):
            if (delta/timeQuantsVals[i]) >= 1:
                IND = max(fInd,i)
                break
    prefix = str(ceil(delta/timeQuantsVals[IND])) + ' '

    return prefix + timeQuants[IND]


def td_parser(strList):
    td = timedelta
    forms = [(('Y','y','year'), td(days=365)), (('Q','q','quarter'), td(days=90)), (('M','month'), td(days=30)),
    (('W','w','week'), td(weeks=1)), (('D','d','day'), td(days=1)), (('H','h','hour'), td(hours=1)),
    (('T','m','min','minute'), td(minutes=1)), (('S','s','sec','second'), td(seconds=1))]
    def fi(s): return [dt for st,dt in forms if s in st][0]
    # dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    out = []
    strs = [strList] if isinstance(strList, str) else strList
    for s in strs:
        if len(s.lstrip('0123456789 .')) > 1:
            s = s.rstrip('s')
        try:
            if any(c.isdigit() for c in s):
                ss = s.lstrip('0123456789 .')
                out.append(float(s.replace(ss,'')) * fi(ss))
            else:
                out.append(fi(s))
        except:
            raise ValueError(f'Name {s.lstrip("0123456789 .")} not found in time delta names') from None
    if isinstance(strList, str):
        return out[0]
    return out


def duration(delta, **kwargs):
    if isinstance(delta, str):
        return delta
    timeQuants = ['Y','M','W','D','S']
    return epoch(timeQuants, td_parser(timeQuants), delta, **kwargs)


def bars_size(delta, **kwargs):
    timeQuants = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins',
                '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours',
                '3 hours', '4 hours', '8 hours','1 day', '1 week', '1 month']
    if isinstance(delta, str):
        return delta
    return epoch(timeQuants, td_parser(timeQuants), delta, **kwargs)
