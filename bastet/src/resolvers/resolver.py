import logging as lg
from src.scrap.statusEnum import StateWorker
from src.resolvers.resolverFact import ResolverFact

_logger = lg.getLogger(__name__)

class Resolver:
    def __init__(self, sources, worker_to_source_map, master, statuskeeper):
        self._sources = sources
        self._workerStateSource = {}
        self._resolver_fact = None
        self._master = master
        self._statuskeeper = statuskeeper
        self.start(worker_to_source_map)
    
    def start(self, worker_to_source_map):
        self._resolver_fact = ResolverFact(resolver = self, master= self._master, statuskeeper= self._statuskeeper)
        for _worker, _source in worker_to_source_map.items():
            self._workerStateSource[_worker] = {"failed": [], "current":_source, "pending": list(filter( lambda x: x != _source, self._sources))}
            
    def resolve(self, worker, failure: Exception):
        return self._resolver_fact.get_resolver(failure).resolve(worker)
    
    def killWorker(self, worker):
        self._workerStateSource[worker]["failed"] = self._workerStateSource[worker]["current"]
        self._workerStateSource[worker]["current"] = None
    
    def switchSource(self, worker):
        self._workerStateSource[worker]["failed"] = self._workerStateSource[worker]["current"]
        new_source = self._workerStateSource[worker]["current"]["pending"].pop()
        self._workerStateSource[worker]["current"] = new_source
        self._master.setWorkerSource(worker, new_source)
        return 