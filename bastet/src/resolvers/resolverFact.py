from src.resolvers.errorhandlers.attributeErrResolver import AttributeErrResolver
from src.resolvers.errorhandlers.errReslover import DefErrResolver
import logging as lg

_logger = lg.getLogger(__name__)

class ResolverFact:
    def __init__(self, resolver, master, statuskeeper):
        super().__init__()
        self._error_resolver_map = {}
        self._resolver = resolver
        self._master = master
        self._statuskeeper = statuskeeper
        self.start()
        
    def start(self):
        self._error_resolver_map = {
            AttributeError : AttributeErrResolver(resolver = self._resolver, master = self._master, statuskeeper= self._statuskeeper),
            "default" : DefErrResolver(resolver = self._resolver, master = self._master, statuskeeper= self._statuskeeper)
            
        }
        
    def get_resolver(self, error: Exception):
        try:
            return self._error_resolver_map[error.__class__]
        except Exception as e:
            _logger.error("failure not taken into consideration still, failure : {0}".format(e))
            return self._error_resolver_map["default"]
             