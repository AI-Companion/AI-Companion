from src.scrap.statusEnum import StateWorker
from src.scrap.workerStatus import WorkerStatus
import logging as lg

_logger = lg.getLogger(__name__)

class DefErrResolver:
    def __init__(self, resolver, master, statuskeeper):
        self._resolver = resolver
        self._master = master
        self._statuskeeper = statuskeeper
        self._exception_type = None
    
    def fatalError(self, worker):
        self._resolver.killWorker(worker)
        self._master.killWorker(worker)
        self._statuskeeper.update(worker, StateWorker.aborted, new_source=None)
        _logger.error("fatal failure detected for a worker")
    
    def nonFatalError(self, worker, new_source=None):
        self._statuskeeper.update(worker, StateWorker.running, new_source)
        _logger.info("recovering from failure for a worker")
            
    def resolve(self, worker):
        self.fatalError(worker)