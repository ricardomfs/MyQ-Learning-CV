from collections import deque
import random
import pickle
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)
    
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        self.memory
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
    
    def save_memory(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.memory, file)

    def load_memory(self, file_path):
        with open(file_path, 'rb') as file:
            self.memory = pickle.load(file)