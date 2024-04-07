# base_algorithm.py

class BaseAlgorithm:
    def select_action(self, state, epoch):
        raise NotImplementedError
    
    def optimize_model(self, batch):
        raise NotImplementedError
    
    def save(self, filepath):
        raise NotImplementedError
    
    def load(self, filepath):
        raise NotImplementedError
