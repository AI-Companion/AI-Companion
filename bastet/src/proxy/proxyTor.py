import requests
import threading
import logging as lg
import time

from fake_useragent import UserAgent
from stem import Signal
from stem.control import Controller

_logger = lg.getLogger(__name__)

class ProxyTor(object):
    __instance = None
    __session  = None
    __threadWorker = None
    __observers = []
    __ua       = UserAgent() # random user agent
    
    def __new__(cls, freq=5):
        if not ProxyTor.__instance:
            ProxyTor.__instance = object.__new__(cls)
            cls.__freq = freq
            ProxyTor.start()
            ProxyTor.setup_worker()
        return ProxyTor.__instance

    def __del__(self):
        self.__threadWorker.join()
        
    
    @classmethod
    def start(cls):
        session = requests.session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=800, pool_maxsize=800, max_retries=20)
        # Tor uses the 9050 port as the default socks port
        session.proxies = {'http':  'socks5://127.0.0.1:9050',
                        'https': 'socks5://127.0.0.1:9050'}
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.mount('socks5://', adapter)
        session.headers = {'User-Agent': ProxyTor.__ua.random}
        ProxyTor.__session = session
    
    @classmethod
    def stop(cls):
        _logger.warn("Tor Proxy stopped")
        cls.__threadWorker.join()
    
    @classmethod
    def setup_worker(cls):
        ProxyTor.__threadWorker = threading.Thread(target=ProxyTor.update_proxy)
        ProxyTor.__threadWorker.start()
        
    @classmethod
    def update_proxy(cls):
        while True:
            with Controller.from_port(port = 9051) as controller:
                controller.authenticate(password="azerty*123")
                controller.signal(Signal.NEWNYM)
            ProxyTor.start()
            _logger.info("switching to a new ip")
            ProxyTor.notify()
            _logger.info("updating ip for suscribers")
            time.sleep(cls.__freq)
    
    @classmethod
    def get_tor_session(cls):
        return ProxyTor.__session

    @classmethod
    def attach(cls, obs):
        if obs not in ProxyTor.__observers: ProxyTor.__observers.append(obs) 
        
    @classmethod
    def detach(cls, obs):
        if obs in ProxyTor.__observers: ProxyTor.__observers.remove(obs) 
    
    @classmethod
    def notify(cls):
        map(lambda x : x.update(), ProxyTor.__observers)
        
# from bs4 import BeautifulSoup

# def parseValues(target):
#     elem = {}
#     tds = target.findAll("td") 
    

# while True:
#     prox = ProxyTor(freq = 50)
#     sess = prox.get_tor_session()
#     # url = "https://www.investing.com/currencies/usd-cad"
#     # url = "https://cryptowat.ch/assets"
#     # url = "https://finance.yahoo.com/quote/{0}{1}%3DX?p={0}{1}%3DX".format("usd", "eur")
#     # url = "https://coinalyze.net/bitcoin/usd/coinbase/price-chart-live/"
#     url = "https://www.tradingview.com/markets/cryptocurrencies/prices-all/"
#     crypto_ticker = []
#     page = sess.get(url).text
#     # page = sess.get("http://icanhazip.com/")
#     # print(page)
#     soup = BeautifulSoup(page, features="html.parser")
#     target = soup.findAll("tr")[1:]
#     parseValues(target)
#     print(target[0])