import logging as lg
from src.scrap.statusEnum import StateWorker

_logger = lg.getLogger(__name__)

class ErrorResolver:
    def __init__(self, sources, worker_to_source_map):
        self._sources = sources
        self._workerStateSource = {}
        self.start(worker_to_source_map)
    
    def start(self, worker_to_source_map):
        for _worker, _source in worker_to_source_map.items():
            self._workerStateSource[_worker] = {"failed": [], "current":_source, "pending": list(filter( lambda x: x != _source, self._sources))}
            
    def resolve(self, worker, failure: Exception):
        pass # AttributeError