import logging as lg
import os
import src
import json
from .exceptions import DbConnectionException

_logger = lg.getLogger(__name__)

#Singleton
class DbManager(object):
    __instance = None
    __db = None
    __file_path = ""
    __data = {}
    
    def __new__(cls):
        if not DbManager.__instance:
            DbManager.__instance = object.__new__(cls)
            DbManager.setupFile()
            DbManager.connect()
        return DbManager.__instance
    
    @classmethod
    def setupFile(cls):
        if os.getenv("VIRTUAL_ENV"):
            cls.__file_path = os.path.join(
                    os.path.dirname(
                    os.path.normpath(os.getenv("VIRTUAL_ENV"))
                    ),
                    "src/db/sources.json"
                )
        else:
            cls.__file_path = os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(src.__file__))
                    ),
                    "src/db/sources.json"
                )
    
    @classmethod
    def connect(cls):
        try:
            with open(cls.__file_path) as f:
                cls.__data = json.load(f)  
                _logger.info("Successfully connected and load data from DB")
                
        except FileNotFoundError:
            _logger.error("couldn't connect to DB")
            raise DbConnectionException
        
        except Exception as e:
            _logger.error("error occured while connecting to DB, {0}".format(str(e)))
    
    @classmethod
    def getAllSources(cls, type='simple'):
        return cls.__data[type]['sources']
    
    @classmethod
    def getSource(cls, type='simple', source='yahoo'):
        return cls.__data[type]['sources'][source]
    
    @classmethod
    def getTickersList(cls, type='simple'):
        return cls.__data[type]["tickers"]
    
    @classmethod
    def getAllTickers_specs(cls, type='simple'):
        return cls.__data[type]["tickers_specs"]

    @classmethod
    def getTickers_spec(cls, type='simple', ticker='usd'):
        return cls.__data[type]["tickers_specs"][ticker]
    
    