import threading
import re
import time
import logging as lg
from datetime import datetime

from src.proxy.proxyTor import ProxyTor
from src.proxy.proxyObservers import Observer
from bs4 import BeautifulSoup
from src.scrap.statusEnum import StateWorker

_logger = lg.getLogger(__name__)

class Worker(Observer, threading.Thread):
    def __init__(self, master, source, fromCurr, toCurr):
        threading.Thread.__init__(self)
        self.__source = source
        self.__from_curr = fromCurr
        self.__to_curr = toCurr
        self.filled_url = ""
        self._status = StateWorker.running
        self._master = master
        
    def setup(self):
        try:
            _logger.info("Starting Worker {0}".format(self))
            # get initial ip
            self.session_update()
            # subscribe to the observable
            ProxyTor.attach(self)
            # populate source url with the values
            self.set_full_url()
            #start thread
            _logger.info("worker of trade of {0} to {1} is fired".format(self.__from_curr, self.__to_curr))
        except Exception as e:
            _logger.error("error occured while setting up worker {0}, {1}".format(self,str(e)))
            
    
    def session_update(self):
        self.__sess = ProxyTor.get_tor_session()
        
    def update(self):
        self.session_update()
    
    def set_full_url(self):
        self.filled_url = self.__source.populateUrl(self.__from_curr, self.__to_curr)
    
    def set_state(self, state):
        self._status = state
    
    def set_source(self, source):
        self.__source = source
    
    def get_curr(self):
        return self.__from_curr
    
    def kill(self):
        ProxyTor.detach(self)
        self.set_state(StateWorker.aborted)
        _logger.warning("worker of trade {} to USD is killed ...".format(self.__from_curr))
    
    def send_value(self, val, now):
        _logger.info("value of trade of {0} to USD at {1} is {2}".format(self.__from_curr, now, val)) # temporarely until api setup
    
    def wait_for_instructions(self):
        while self._status == StateWorker.pending:
            _logger.warning("worker of trade {} to USD is still pending resolution ...".format(self.__from_curr))
            time.sleep(5)
        self.scrapp()
    
    def scrapp(self):
        try:
            while self._status == StateWorker.running:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
                self.set_full_url()
                page = self.__sess.get(self.filled_url).text
                # target = self.__sess.get("https://httpbin.org/ip").text
                soup = BeautifulSoup(page, features="html.parser")
                target = soup.find(self.__source.pattern["tagTarget"], self.__source.pattern["attributes"]).text
                self.send_value(re.sub('[^0-9\.]', '', target.strip()), now)
        except Exception as e:
             _logger.error("An error Occured while scrapping {0} to {1} , {2} switching state to pending ...".format(self.__from_curr, self.__to_curr, e))
             self._status = StateWorker.pending
             self._master.workerFail(self, e)
             self.wait_for_instructions()
             pass
    
    def run(self):
        self.setup()
        try:
            self.scrapp()
        except Exception as e:
            _logger.error("an error occured while starting worker {0}".format(e))
            self.kill()
            
            
        