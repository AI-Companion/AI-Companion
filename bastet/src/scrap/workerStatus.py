import logging as lg
from src.scrap.statusEnum import StateWorker

_logger = lg.getLogger(__name__)

class WorkerStatus:
    """
    keeps record of workers status
    self._status = {
        StateWorker.state = []
    }
    
    self._workerState = {
        "worker instance":{
            "state": StateWorker.state,
            "source": "the source that the worker is scrapping"
        }
    }
    """
    def __init__(self, worker_to_source_map):
        self._status = {StateWorker.running:[], StateWorker.pending:[], StateWorker.aborted:[]}
        self._workerState = {}
        self.start(worker_to_source_map)
    
    def start(self, worker_to_source_map):
        for _worker, _source in worker_to_source_map.items():
            self._workerState[_worker] = {"state": StateWorker.running, "source":_source}
            self._status[StateWorker.running].append(_worker)
    
    def update(self, worker, new_state, new_source=None):
        old_state = self._workerState[worker]["state"]
        self._workerState[worker]["state"] = new_state
        if new_source != None: self._workerState[worker]["source"] = new_source
        self._status[old_state].remove(worker)
        self._status[new_state].append(worker)
    
    def getWorkerState(self, worker):
        return self._workerState[worker]
    
    def getStatus(self):
        return {
            "running": len(self._status[StateWorker.running]),
            "pending":len(self._status[StateWorker.pending]),
            "aborted":len(self._status[StateWorker.aborted])
            }
    