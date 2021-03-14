import logging as lg
from src.db.dbManager import DbManager
from src.model.source import Source
from src.model.worker import Worker

_logger = lg.getLogger(__name__)

class ScrapperMaster(object):
    __instance = None
    __db = None
    
    def __new__(cls, config={}):
        if not ScrapperMaster.__instance:
            ScrapperMaster.__instance = object.__new__(cls)
            cls.__config = config
            ScrapperMaster.__workers = []
            ScrapperMaster.__workers_to_source_map = {}
            ScrapperMaster.__sources_to_workers_map = {
                        "simple":{
                            "val":[]                        },
                        "crypto":{
                            "val":[]
                        }
                    }
        return ScrapperMaster.__instance
    
    @classmethod
    def start(cls):
        cls.__db = DbManager()
        cls.bootStrapConfig()
        cls.startWorkers()
        
    @classmethod
    def stop(cls):
        map(lambda x: x.join(), cls.__workers)
    
    @classmethod
    def startWorkers(cls):
        _logger.info("Firing up Workers")
        try:
            for w in cls.__workers:
                w.start()
        except Exception as e:
            _logger.error("An error occured while Firing up the workers, {0}".format(e))
        
    @classmethod
    def setupSources(cls): # only crypto sources for now
        try:
            _logger.info("Setting up Sources")
            sources = cls.__db.getAllSources(type='crypto')
            for key in sources:
                _source = Source(sources[key]['url'], 'crypto', sources[key]['pattern'])
                cls.__sources_to_workers_map["crypto"][_source] = []
                cls.__sources_to_workers_map["crypto"]["val"].append(_source)
            _logger.info("Successfully setup Sources")
        except Exception as e:
            _logger.error("error occured while setting up Sources, {0}".format(str(e)))
            raise
    
    @classmethod    
    def setupWorkers(cls):
        try:
            _logger.info("Setting up Workers")
            config = cls.__config["crypto"] if cls.__config else cls.__db.getTickersList(type="crypto")
            print("aaaaaa")
            for curr in config:
                _source = cls.__sources_to_workers_map["crypto"]["val"][-1] # for every worker lets take a single source for now
                _worker = Worker(source=_source, fromCurr=cls.__db.getTickers_spec(type="crypto", ticker=curr)["currency"], toCurr=curr) # for now the value of every currency is regarding usd
                cls.__workers.append(_worker)
                cls.__workers_to_source_map[_worker] = _source
                cls.__sources_to_workers_map["crypto"][_source].append(_worker)
            _logger.info("Successfully setup {0} Workers".format(len(cls.__workers)))

        except Exception as e:
            _logger.error("error occured while setting up workers, {0}".format(str(e)))
            raise
            
    @classmethod
    def bootStrapConfig(cls):
        cls.setupSources()
        cls.setupWorkers()
        