import pytz as tz
from tzlocal import get_localzone
from pytz import timezone
from zoneinfo import ZoneInfo
import marksman_extras as ps
import uuid
import itertools as it
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from marksman_objects import *
from postgresql_db import *
from marksman_extras import *
from psycopg2 import sql
from datetime import datetime, timedelta, date
import time
import pytz
import collections
s = sql.SQL
l = lambda x : pytz.timezone('America/New_York').localize(pd.Timestamp(x))
lol = l(datetime.now())
r = pd.Interval(l(datetime(2021,10,8)), l(datetime.now()))
q = {r : {r : range(0,7)}}

# print(oc)

# print(json.loads('{"(2021-10-08 00:00:00-04:00,2022-02-02 14:45:03.496819-05:00]": {"(2021-10-08 00:00:00-04:00,2022-02-02 14:45:03.496819-05:00]": 77}}'))
# print(json.dumps(q, cls=JamesonEncoder))
print(json.dumps({r : {r : 'k'},'hhh' : {r : 'k'}, 'no' : q}, cls=JamesonEncoder))

# print(q)
# print(pd_interval2str(r))
# # time.sleep(1)
# lol = l(datetime.now())
# print(q[r])
# print(lol in r)
# print(None.5())
# print(torch.__version__)
# x = np.array([[1, 0, 0], [0, 0, 0]])
# print(x)
# print(np.cov(x))
#
# y = torch.from_numpy(x)
# print(torch.cov(y.transpose(0,1)))

# x = np.array([[1, 3, 5], [2, 4, 6]])
# print(x.ravel(order = 'C'))
# print(x.shape)
# x = np.array([[[1, 3, 5], [2, 4, 6]], [[7, 9, 11], [8, 10, 12]]], order = 'C')
# print(x[0,1,2])
# print(x.shape)
# print(x.ravel(order = 'C'))
# class JamesonSuper(json.JSONEncoder):
#     def __init__(self, *args, **kwargs):
#         self.set_default()
#
#         return super().__init__(*args, **kwargs)
#
#
#     def set_default(self, default=[[me.pd_interval2str, pd.Interval], lambda x : list(iter(x))]):
#         if callable(default):
#             self.defFuncs = [default]
#         else:
#             self.defFuncs = default
#
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
#         return json.JSONEncoder.default(self, o)
#
#
#     def dict_parse(self, o):
#         try:
#             oc = {self.default(k):self.dict_parse(v) if isinstance(v, dict) else v for k,v in o.items()}
#         except TypeError:
#             pass
#         else:
#             return oc
#
#         return o

# index = pd.MultiIndex.from_product(iterables, names=["first", "second"])
# s = pd.Series(np.random.randn(8), index=index)
# print(slice(None, "A3","A1"))
# print([x for x in it.product([5, 2, 4,])])
# x = {'a' : 1}
