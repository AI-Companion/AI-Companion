import threading
import logging as lg

from src.utils.proxyTor import ProxyTor
from src.utils.proxyObservers import Observer
from bs4 import BeautifulSoup

_logger = lg.getLogger(__name__)

class Worker(Observer, threading.Thread):
    def __init__(self, source, fromCurr, toCurr):
        threading.Thread.__init__(self)
        self.__source = source
        self.__from_curr = fromCurr
        self.__to_curr = toCurr
        self.filled_url = ""
        
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
    
    def kill(self):
        ProxyTor.detach(self)
        self.join()
    
    def send_value(self, val):
        _logger.info("value of trade of {0} to {1} is {2}".format(self.__from_curr, self.__to_curr, val)) # temporarely until api setup
    
    def scrapp(self):
        try:
            while True:
                self.set_full_url()
                page = self.__sess.get(self.filled_url).text
                soup = BeautifulSoup(page, features="html.parser")
                target = soup.find(self.__source.pattern["tagTarget"], self.__source.pattern["attributes"]).text
                self.send_value(target)
        except Exception as e:
             _logger.error("An error Occured while scrapping {0} to {1} , {2}".format(self.__from_curr, self.__to_curr, e))
    
    def run(self):
        self.setup()
        try:
            self.scrapp()
        except Exception as e:
            _logger.error("an error occured while starting worker {0}".format(e))
            self.kill()
            
            
        