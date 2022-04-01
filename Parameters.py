import yaml

class Parameters(object):
    def __init__(self, params={}):
        self._params = params
        
    def __getitem__(self, key):
        return self._params[key]
        
    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self._params, f)
            
    def load(self, path):
        with open(path, 'r') as f:
            self._params = yaml.safe_load(f.read())
            
    def update(self, new_dict):
        self._params.update(new_dict)