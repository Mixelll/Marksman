import copy
import json
import pandas as pd
import ib_insync
from typing import NamedTuple
import marksman_extras as me

class Ticker(NamedTuple):
    symbol: str
    exchange: str = 'SMART'
    currency: str = 'USD'


class JamesonSuper(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        self.set_default()

        return super().__init__(*args, **kwargs)


    def set_default(self, default=[[me.pd_interval2str, pd.Interval], lambda x : list(iter(x))]):
        if callable(default):
            self.defFuncs = [default]
        else:
            self.defFuncs = default


    def default(self, o):
        for d in self.defFuncs:
            if callable(d):
                try:
                    r = d(o)
                except TypeError:
                    pass
                else:
                    return r
            else:
                if isinstance(o, d[1]):
                    print(o)
                    try:
                        r = d[0](o)
                    except TypeError:
                        pass
                    else:
                        return r

        return json.JSONEncoder.default(self, o)


    def dict_parse(self, o):
        try:
            oc = {self.default(k):self.dict_parse(v) if isinstance(v, dict) else v for k,v in o.items()}
            # print(oc)
        except TypeError:
            pass
        else:
            return oc

        return o


class JamesonEncoder(JamesonSuper):

    def set_default(self, default=[[me.pd_interval2str, pd.Interval], lambda x : list(iter(x))]):
        if callable(default):
            self.defFuncs = [default]
        else:
            self.defFuncs = default

    def dict_parse(self, o):
        # try:
# self.default(k):self.dict_parse(v)
        oc = {print(self.default(k)) for k,v in o.items()}
        # print(oc)
        oc = None
        # except TypeError:
        #     pass
        # else:
        #     return oc

        return oc


    def iterencode(self, o, _one_shot=False):
        if isinstance(o, dict):
            return super().iterencode(self.dict_parse(o), _one_shot=_one_shot)

        return super().iterencode(o, _one_shot=_one_shot)


# class JamesonDecoder(json.JSONDecoder):
#     def raw_decode(self, s, idx=0):
#         o = super(JamesonDecoder, self).raw_decode(s, idx=idx)
#
#         return dict_parse(self, o)
#
#     def set_default(self, default=[[me.pd_interval2str, pd.Interval], lambda x : list(iter(x))]):
#         if callable(default):
#             self.defFuncs = [default]
#         else:
#             self.defFuncs = default
#
#     def default(self, o):
#         for d in self.defFuncs:
#             if callable(d):
#                 try:
#                     r = d(o)
#                 except TypeError:
#                     pass
#                 else:
#                     return r
#             else:
#                 if isinstance(o, d[1]):
#                     try:
#                         r = d[0](o)
#                     except TypeError:
#                         pass
#                     else:
#                         return r
#
#
#     def dict_parse(self, o):
#         try:
#             oc = {self.default(k) if isinstance(k, pd.Interval) else k:
#                     self.dict_parse(v) if isinstance(v, dict) else v for k,v in o.items()}
#         except TypeError:
#             pass
#         else:
#             return oc
#
#         return o


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
