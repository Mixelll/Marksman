import copy
import json
import pandas as pd
import ib_insync
from typing import NamedTuple
import marksman_extras as me
from marksman_extras import iter2list

class Ticker(NamedTuple):
    symbol: str
    exchange: str = 'SMART'
    currency: str = 'USD'


class JustinEncoder(json.JSONEncoder):
    def default(self, o):
       try:
           iterable = o
       except TypeError:
           pass
       else:
           return iter2list(o)
       # Let the base class default method raise the TypeError
       return json.JSONEncoder.default(self, o)


class JamesonSuper:
    def __init__(self, *args, **kwargs):
        self.set_default()
        return super().__init__(*args, **kwargs)


    def set_default(self, default=None):
        if callable(default):
            self.defFuncs = [default]
        else:
            self.defFuncs = default


    def default_sup(self, o):
        for d in self.defFuncs:
            if callable(d):
                # print(o)
                try:
                    r = d(o)
                except (TypeError, ValueError):
                    pass
                else:
                    return r
            else:
                if isinstance(o, d[1]):
                    try:
                        r = d[0](o)
                    except (TypeError, ValueError):
                        pass
                    else:
                        return r
        return o


    def dict_parse(self, o):
        # print(type(o))
        if isinstance(o, dict):
            try:
                oc = {self.default_sup(k):self.dict_parse(v) for k,v in o.items()}
            except TypeError:
                pass
            else:
                return oc
        return o


class JamesonEncoder(JamesonSuper, json.JSONEncoder):
    def set_default(self, default=[[me.pd_interval2str, pd.Interval], me.iter2list]):
        super().set_default(default)

    def default(self, o):
        try:
            r = self.default_sup(o)
        except TypeError:
            pass
        else:
            return r

        return json.JSONEncoder.default(self, o)


    def iterencode(self, o, _one_shot=False):
        if isinstance(o, dict):
            return super().iterencode(self.dict_parse(o), _one_shot=_one_shot)

        return super().iterencode(o, _one_shot=_one_shot)


class JamesonDecoder(JamesonSuper, json.JSONDecoder):
    def decode(self, *args, **kwargs):
        o = super().decode(*args, **kwargs)
        print(type(self.dict_parse(o)))
        return self.dict_parse(o)

    def set_default(self, default=me.str2pd_interval):
        super().set_default(default)


# print(json.dumps({'lol' : zip((1,2),(3,4))}, cls=JamesonEncoder))

# def suppress(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception:
#             pass
#     return wrapper
#
# def upload_to_db(func):
#     def inner1():
#         print("Hello, this is before function execution")
#         df = func()
#
#         append_df_to_db(SQLengine, ticker, df)
#
#     return inner1
