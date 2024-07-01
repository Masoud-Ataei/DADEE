from collections import namedtuple, deque
import random
import numpy as np
mem_parameters = ('x', 'u', 'xdot')
Transition = namedtuple('Transition', mem_parameters)

class ReplayMemory(object):
    ## TODO: ((Change lists to unpack *zip))
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.er = 1e-1
        self.cnt = 0
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def push_diff(self, *args):
        """Save a transition"""
        if len(self) > 0:
            new = np.concatenate(args).reshape((1,-1))
            data = np.array( [np.concatenate([l.x, l.u, l.xdot]) for l in list(self.memory)])
            diff = data - new
            dis  = np.linalg.norm(diff)
            if dis < self.er:
                self.cnt += 1
                print(self.cnt, 'ignored')
                return
        self.memory.append(Transition(*args))

    def sample(self, batch_size):        
        l = len(self.memory)
        assert l > 0
        if batch_size > l:
            return random.sample(self.memory, l) 
        return random.sample(self.memory, batch_size) 

    def get_samples(self, batch_size, seed = 0):
        if seed > 0:
            random.seed(seed)
        sam = self.sample(batch_size=batch_size)
        return np.array([t.x for t in sam]),np.array([t.u for t in sam]),np.array([t.xdot for t in sam])

    def get_dataset(self):        
        return np.array([t.x for t in self.memory]),np.array([t.u for t in self.memory]),np.array([t.xdot for t in self.memory])


    def last_sample(self):
        # print(self.memory[-1])
        return [v for v in self.memory[-1]]
        
    def __len__(self):
        return len(self.memory)
