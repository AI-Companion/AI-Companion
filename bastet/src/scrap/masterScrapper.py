import logging as lg
from src.db.dbManager import DbManager
from src.resolvers.resolver import Resolver
from src.scrap.source import Source
from src.scrap.worker import Worker
from src.scrap.workerStatus import WorkerStatus
from src.scrap.statusEnum import StateWorker

_logger = lg.getLogger(__name__)

class MasterScrapper(object):
    __instance = None
    __db = None
    
    def __new__(cls, config={}):
        if not MasterScrapper.__instance:
            MasterScrapper.__instance = object.__new__(cls)
            cls.__config = config
            cls._resolver = None
            cls._statusKeeper = None
            MasterScrapper.__workers = []
            MasterScrapper.__workers_to_source_map = {}
            MasterScrapper.__sources_to_workers_map = {
                        "values":[], # contains list of sources
                        "simple":{
                            "val":[] # contains list of simple currency sources           
                            },
                        "crypto":{
                            "val":[] # contains list of crypto currency sources
                        }
                    }
        return MasterScrapper.__instance
    
    @classmethod
    def start(cls):
        cls.__db = DbManager()
        cls.bootStrapConfig()
        cls.startWorkers()
        
    @classmethod
    def stopWorkers(cls):
        map(lambda x: x.kill(), cls.__workers)
        
    @classmethod
    def killWorker(cls, worker):
        worker.kill()
    
    @classmethod
    def startWorkers(cls):
        _logger.info("Firing up {0} Workers".format(len(cls.__workers)))
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
                cls.__sources_to_workers_map["values"].append(_source)
            _logger.info("Successfully setup {0} Sources".format(len(sources)))
        except Exception as e:
            _logger.error("error occured while setting up Sources, {0}".format(str(e)))
            raise
    
    @classmethod    
    def setupWorkers(cls):
        try:
            _logger.info("Setting up Workers")
            config = cls.__config["crypto"] if cls.__config else cls.__db.getTickersList(type="crypto")
            for curr in [config[3]]:
                _source = cls.__sources_to_workers_map["crypto"]["val"][-1] # for every worker lets take a single source for now
                _worker = Worker(master=cls.__instance, source=_source, fromCurr=cls.__db.getTickers_spec(type="crypto", ticker=curr)["currency"], toCurr=curr) # for now the value of every currency is regarding usd
                cls.__workers.append(_worker)
                cls.__workers_to_source_map[_worker] = _source
                cls.__sources_to_workers_map["crypto"][_source].append(_worker)
            _logger.info("Successfully setup {0} Workers".format(len(cls.__workers)))
        except Exception as e:
            _logger.error("error occured while setting up workers, {0}".format(str(e)))
            raise
        
    @classmethod
    def setWorkerSource(cls, worker: Worker, source: Source):
        worker.set_source(source)
        worker.set_state(StateWorker.running)
        _logger.info("worker of {0} is restarted with a new source".format(worker.get_curr()))
            
    @classmethod
    def bootStrapConfig(cls):
        cls.setupSources()
        cls.setupWorkers()
        cls._statusKeeper = WorkerStatus(worker_to_source_map=cls.__workers_to_source_map, master=cls.__instance, freq=7)
        _logger.info("Status keeper started ...")
        cls._resolver = Resolver(sources=cls.__sources_to_workers_map["values"], worker_to_source_map=cls.__workers_to_source_map, master=cls.__instance, statuskeeper= cls._statusKeeper)
        _logger.info("Resolver started ...")
        
    @classmethod
    def workerFail(cls, worker: Worker, failure: Exception):
        cls._statusKeeper.update(worker, StateWorker.pending)
        cls._resolver.resolve(worker, failure)
