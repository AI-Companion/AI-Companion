import json

class Source:
    def __init__(self, urll, currType, pattern):
        self.url = urll
        self.curr_type = currType
        self.pattern = pattern
        
    def populateUrl(self, from_curr, to_curr):
        return self.url.format(from_curr, to_curr)
        
    def __eq__(self, other):
        return self.__key() == other.__key()
    
    def __key(self):
        return (self.url, self.curr_type, json.dumps(self.pattern))
        
    def __hash__(self):
        return hash(self.__key())