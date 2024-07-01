# Import packages.
import numpy as np

class trajectory():
    def __init__(self, parms_env):
        path = parms_env['path']        
        assert path.shape[0] > 0
        self.path = path
        self.obstacles = parms_env['obstacles']
        self.step = 0
        self.end  = False
        self.dist = 0
        self.n    = self.path.shape[0]
        self.dist_er = parms_env['dist_er']

    def reset(self):
        self.step = 0
        self.end  = False
    
    def check_destination(self, xt):
        step = self.step
        xd = self.path[step]
        n  = self.path.shape[0]
        end = self.end
        self.dist = np.linalg.norm(xt[:2] - xd[:2])
        if(self.dist <= self.dist_er):
            if((step+1) < n):
                step += 1
            else:
                end = True
        
        ##### update variables
        self.end = end
        self.step = step
    
    def get_end(self):
        return self.end
    
    def get_xd(self, xt):
        self.check_destination(xt)
        return (self.get_end() , self.path[self.step])
    
    def get_obstacles(self):
        return self.obstacles
    
    def get_dist(self):        
        return self.dist
