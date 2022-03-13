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
import datetime as dt
from datetime import datetime, timedelta, date, time
import time as ti
import pytz
import collections
import re
from uuid import uuid4
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

s = sql.SQL
lz = lambda x : pytz.timezone('America/New_York').localize(pd.Timestamp(x))
l = lambda x : pd.Timestamp(x)
def intez(a, b): return pd.Interval(lz(a), lz(b))
def inte(a, b): return pd.Interval(l(a), l(b))
x = it.product((5,6), (7,8))

startDate = time(9,30,tzinfo=timezone('America/New_York'))
startDate = datetime(2022,2,15,18).astimezone(timezone('America/New_York'))
startDate = startDate.astimezone(timezone('Israel'))

print(startDate.time())
# print(time(9,30,tzinfo=dt.timezone(timedelta(hours=-5))).tzinfo)
# startDate = datetime(2022,2,15)
print(timezone('America/New_York').localize(datetime(2022,2,15)))
# print(to_timezone(startDate, timezone('America/New_York')))
# print(startDate)
r = intez('2021-11-11 14:35:14.777941', '2021-11-11 16:30:00')
rl = r.left \
.strftime("%Y-%m-%d %H:%M:%S %z")
rr = r.right.strftime("%Y-%m-%d %H:%M:%S %z")
# rn = inte(rl, 'ghgh')

# matplotlib.rcParams['figure.facecolor'] = '#ffffff'


q = (5,8,9)
j = json.dumps(q, cls=JamesonEncoder)
dict_merge({'a': {'b': 6}}, {'a': {'b': 6}})
# print('From')
# p = {r : {r : 'k'},'hhh' : {r : 'k'}, 'no' : q}
# p = {r : 5}
# print(p)
# print('To')
# j = json.dumps(p, cls=JamesonEncoder)
# print(json.loads(j, object_hook=print))
# print(j)
# print('And back to')
# print(json.loads(j , cls=JamesonDecoder))
# json.loads('{"(2021-10-08 00:00:00-04:00,2022-02-02 23:37:58.859681-05:00]": {"(2021-10-08 00:00:00-04:00,2022-02-02 23:37:58.859681-05:00]": "k"}, "hhh": {"(2021-10-08 00:00:00-04:00,2022-02-02 23:37:58.859681-05:00]": "k"}, "no": {"(2021-10-08 00:00:00-04:00,2022-02-02 23:37:58.859681-05:00]": {"(2021-10-08 00:00:00-04:00,2022-02-02 23:37:58.859681-05:00]": [0, 1, 2, 3, 4, 5, 6]}}}' , cls=JamesonDecoder)

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

# index = pd.MultiIndex.from_product(iterables, names=["first", "second"])
# s = pd.Series(np.random.randn(8), index=index)
# print(slice(None, "A3","A1"))
# print([x for x in it.product([5, 2, 4,])])
# x = {'a' : 1}
