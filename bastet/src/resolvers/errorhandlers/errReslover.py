from src.scrap.statusEnum import StateWorker
import logging as lg

_logger = lg.getLogger(__name__)

class DefErrResolver:
    def __init__(self, resolver, master):
        self._resolver = resolver
        self._master = master
        self._exception_type = None
    
    def fatalError(self):
        _logger.error("fatal failure detected for a worker")
        return StateWorker.aborted
    
    def recoverError(self):
        _logger.info("recovering from failure for a worker")
        return StateWorker.running
    
    def resolve(self):
        return self.fatalError