import copy
import inspect
import json
import pandas as pd
import re

import marksman_extras as me

from typing import NamedTuple


class Ticker(NamedTuple):
    symbol: str
    exchange: str = 'SMART'
    currency: str = 'USD'


class JustinEncoder(json.JSONEncoder):

    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)

class JacksonEncoder(json.JSONEncoder):

    def default(self, o):
        try:
            r = o.to_json()
            # print(r)
        except TypeError:
            pass
        else:
            return r
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
            if callable(d) or len(d)==1:
                if not callable(d):
                    d = d[0]
                try:
                    r = d(o)
                except (TypeError, ValueError):
                    pass
                else:
                    return r
            else:
                if d[1](o):
                    try:
                        r = d[0](o)
                    except (TypeError, ValueError):
                        pass
                    else:
                        return r
        return o

    # def special_key(self, k): return False
    # def special_value(self, v): return False
    def dict_parse(self, o):
        if isinstance(o, dict):
            try:
                # oc = {self.special_key(k) if self.special_key(k) else self.default_sup(k):
                #     self.special_value(v) if self.special_key(k) else self.dict_parse(v)
                #     for k,v in o.items()}
                oc = {self.default_sup(k): self.dict_parse(v) for k,v in o.items()}
            except TypeError:
                pass
            else:
                return oc
        return o


class JamesonEncoder(JamesonSuper, json.JSONEncoder):

    def set_default(self, default=[[me.pd_interval2str, lambda o: isinstance(o, pd.Interval)],
        [lambda o: o.to_dict(),
            lambda o: re.search("'.*'", str(type(o))).group().replace("'", '') \
            in ('pandas.core.frame.DataFrame')],
        me.iter2list]):
        super().set_default(default)

    def default(self, o):
        try:
            r = self.default_sup(o)
            # print(r)
        except TypeError:
            pass
        else:
            return r
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)

    def iterencode(self, o, _one_shot=False):

        if isinstance(o, dict):
            return super().iterencode(self.dict_parse(o), _one_shot=_one_shot)
        return super().iterencode(o, _one_shot=_one_shot)


# class JamesonDecoder(JamesonSuper, json.JSONDecoder):
#
#     def special_key(self, k): return me.str2pd_interval(k)
#     def special_value(self, v): return pd.read_json(v)
#     def decode(self, *args, **kwargs):
#         o = super().decode(*args, **kwargs)
#         return self.dict_parse(o)
#
#     def set_default(self, default=me.str2pd_interval):
#         super().set_default(default)


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
