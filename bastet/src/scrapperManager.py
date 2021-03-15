import logging as lg
from src.proxy.proxyTor import ProxyTor
from src.scrap.masterScrapper import MasterScrapper

_logger = lg.getLogger(__name__)

class ScrapperManager:
    
    def __init__(self, config=None):
        self.__config = config
    
    def start(self):
        # start the proxy ie the ip changer ( started once initiated)
        _logger.info("firing up Tor proxy")
        self.__proxy = ProxyTor()
        _logger.info("Tor proxy setup")
        
        #start scrapper master with the config
        _logger.info("firing up master scrapper")
        self.__master = MasterScrapper(self.__config)
        self.__master.start()
        _logger.info("firing up master started")
        
    def stop(self):
        self.__proxy.stop()
        self.__master.stop()
        
    