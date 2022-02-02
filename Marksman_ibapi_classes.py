from inspect import getmembers, isfunction, isclass

import ibapi
from ibapi import connection
from ibapi import client

from ibapi import wrapper
from ibapi import utils
from ibapi.client import EClient
from ibapi.utils import iswrapper


class mwrapper(wrapper.EWrapper):
    pass
class mclient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)
class mapp(mwrapper, mclient):
     def __init__(self):
         mwrapper.__init__(self)
         mclient.__init__(self, wrapper=self)
def TWS_connect():
    myapp = mapp()
    myapp.connect("127.0.0.1", 7497, clientId=1)
    return myapp
