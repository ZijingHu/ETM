import os
import yaml

class Parameters(object):
    '''store, save, and load parameters
    
    INTERFACE
    ---------
    self.__init__
      params [dict] input params
      load [str] path to load the params
      
    self.save
      path [str] path to save the params
      
    self.load
      path [str] path to load the params
      
    self.update
      new_dict [dict] partially or completely update params
    '''
    def __init__(self, params={}, load=None):
        self._params = params
        if load:
            self.load(load)

    def __getitem__(self, key):
        return self._params[key]
        
    def save(self, path):
        with open(os.path.join(path, 'params.yml'), 'w') as f:
            yaml.dump(self._params, f)
            
    def load(self, path):
        with open(os.path.join(path, 'params.yml'), 'r') as f:
            self._params = yaml.safe_load(f.read())
            
    def update(self, new_dict):
        self._params.update(new_dict)