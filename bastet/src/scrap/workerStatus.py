import logging as lg
import threading
import time
from src.scrap.statusEnum import StateWorker

_logger = lg.getLogger(__name__)

class WorkerStatus:
    """
    keeps record of workers status
    self._status = {
        StateWorker.state = []
    }
    
    self._workerState = {
        "worker instance": StateWorker.state,
    }
    """
    def __init__(self, worker_to_source_map, master, freq=10):
        self._status = {StateWorker.running:[], StateWorker.pending:[], StateWorker.aborted:[]}
        self._workerState = {}
        self.__threadWorker = None
        self._freq = freq
        self._master = master
        self.start(worker_to_source_map)
    
    def start(self, worker_to_source_map):
        for _worker, _source in worker_to_source_map.items():
            self._workerState[_worker] =  StateWorker.running
            self._status[StateWorker.running].append(_worker)
        self.startDebrifer()
    
    def update(self, worker, new_state, new_source=None):
        old_state = self._workerState[worker]
        self._workerState[worker] = new_state
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
    
    def sendStatus(self):
        _logger.info("Status Worker started ...")
        try:
            while True:
                time.sleep(self._freq)
                status = self.getStatus()
                _logger.info(status)
        except Exception as e:
            _logger.info("Status Worker failed ...")
            raise
    
    def startDebrifer(self):
        self.__threadWorker = threading.Thread(target=self.sendStatus)
        self.__threadWorker.start()