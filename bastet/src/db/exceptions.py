import logging as lg

_logger = lg.getLogger(__name__)

class DbConnectionException(Exception): # this is for redis 
    # _logger.error("error occured while connecting to database")
    pass