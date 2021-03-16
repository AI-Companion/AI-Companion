from abc import ABC, abstractmethod

class Observer(ABC):
    
    @abstractmethod
    def update(self):
        pass
    
    def preProcess(self):
        pass
    
    def postProcess(self):
        pass
    
    def scrapp(self):
        pass