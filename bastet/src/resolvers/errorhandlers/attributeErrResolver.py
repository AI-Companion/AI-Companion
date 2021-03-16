from src.resolvers.errorhandlers.errReslover import DefErrResolver
from src.scrap.statusEnum import StateWorker
from src.scrap.workerStatus import WorkerStatus
import logging as lg

_logger = lg.getLogger(__name__)

class AttributeErrResolver(DefErrResolver):
    def __init__(self, resolver, master, statuskeeper):
        super().__init__(resolver, master, statuskeeper)
        
    def resolve(self, worker):
        if len(self._resolver._workerStateSource[worker]["pending"]) == 0:
            self._resolver._workerStateSource[worker]["failed"] = self._resolver._workerStateSource[worker]["current"]
            self.fatalError(worker)
        else:
            self._resolver._workerStateSource[worker]["failed"] = self._resolver._workerStateSource[worker]["current"]
            new_source = self._resolver._workerStateSource[worker]["pending"].pop()
            self._resolver._workerStateSource[worker]["current"] = new_source
            self._master.setWorkerSource(worker, new_source)
            _logger.info("Switching worker {0} to a new Source {1}".format(worker, new_source.url))
            self.nonFatalError(worker, new_source)
            
