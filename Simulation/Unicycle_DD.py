# Import packages.
from dataclasses import dataclass
import numpy as np

class UnicyleModel_Ack1:
    def __init__(self, L, R):
        self.L = L
        self.R = R

    def get_Fx(self, xt : np.ndarray, sigma = 0.0, mu = 0):
        # _x = x,y,theta,v
        theta = xt[2]
        f = np.zeros((3,1))
        g = np.zeros((3,2))
        g[0, 0] = np.math.cos(theta)
        g[1, 0] = np.math.sin(theta)        
        g[2, 1] = 1 / self.L
        F = np.zeros((3, 3))        
        F[:,0] = f[:,0]
        F[:,1:] = g        
        F = F + sigma * np.random.randn(3,3) + mu
        return F

    def get_JFx(self, xt : np.ndarray):
        theta = xt[2]        
        # f0 g0 g4
        # f1 g1 g5
        # f2 g2 g6     
        JFx = np.zeros((9,3))    
        # dg0/dtheta
        JFx[1, 2] = -np.math.sin(theta)
        # dg1/dtheta 
        JFx[4, 2] =  np.math.cos(theta)
        return JFx

class UnicyleModel:
    def __init__(self, L, R):
        self.L = L
        self.R = R

    def get_Fx(self, xt : np.ndarray, sigma = 0.0, mu = 0):
        # _x = x,y,theta,v
        theta = xt[2]
        v     = xt[3]
        f = np.zeros((4,1))
        f[0] = v * np.math.cos(theta)
        f[1] = v * np.math.sin(theta)
        g = np.zeros((4,2))
        g[2,0] = 1 / self.L
        g[3,1] = 1
        F = np.zeros((4,3))        
        F[:,0] = f[:,0]
        F[:,1:] = g        
        F = F + sigma * np.random.randn(4,3) + mu
        return F

    def get_JFx(self, xt : np.ndarray):
        theta = xt[2]
        v     = xt[3]
        # f0 g0 g4
        # f1 g1 g5
        # f2 g2 g6
        # f3 g3 g7        
        JFx = np.zeros((12,4))    
        # df0/dtheta , df0/dv
        JFx[0, 2] = -v * np.math.sin(theta)
        JFx[0, 3] = np.math.cos(theta)
        # df1/dtheta , df1/dv
        JFx[3, 2] =  v * np.math.cos(theta)
        JFx[3, 3] = np.math.sin(theta)
        return JFx
        
class DD_Model:
    def __init__(self, L, R, M):
        self.L = L
        self.R = R
        self.M = M
    def get_Fx(self, xt : np.ndarray, sigma = 0.0, mu = 0):
        # _x = x,y,theta,wr, wl 
        R = self.R
        L = self.L
        M = self.M
        # theta = xt[2]
        [x, y, theta, w, v] = xt
        f = np.zeros((5,1))
        f[0] = v * np.math.cos(theta)
        f[1] = v * np.math.sin(theta)
        f[2] = w
        g = np.zeros((5,2))
        g[3,0] = 1
        g[4,1] = 1 / M
        F = np.zeros((5,3))        
        F[:,0] = f[:,0]
        F[:,1:] = g        
        F = F + sigma * np.random.randn(5,3) + mu
        return F
    

    def get_JFx(self, xt : np.ndarray):
        R = self.R
        L = self.L
        M = self.M
        # theta = xt[2]
        [x, y, theta, w, v] = xt
        # f0 g0 g4
        # f1 g1 g5
        # f2 g2 g6
        # f3 g3 g7        
        JFx = np.zeros((15,5))    
        # df0/dtheta , df0/dur, df0/dul
        JFx[0, 2] = -v * np.math.sin(theta)
        JFx[0, 4] =      np.math.cos(theta)
        # df1/dtheta , df1/dur, df1/dul
        JFx[3, 2] =  v * np.math.cos(theta)
        JFx[3, 4] =      np.math.sin(theta)
        # df2/dur, df1/dul   
        JFx[6, 3] =  1
        
        return JFx
        

@dataclass
class vehicle_class:
    """Class for keeping track of the vechile state."""
    xt: np.ndarray
    model: DD_Model or UnicyleModel
    def get_Fxt(self, sigma = 0.0, mu = 0):
        return self.model.get_Fx(self.xt, sigma, mu)

    def get_JFxt(self, sigma = 0.0, mu = 0):
        return self.model.get_JFx(self.xt)

class Simulator():
    def __init__(self, x=np.array([0.0,0.0,0.0,0.0,0.0]), xdot=np.array([0.0,0.0,0.0,0.0,0.0]), u = np.array([1.0,0.0,0.0]), model_type = "DD",  L = 1, R = 1, M = 1):
        self.step = 0
        self.t    = 0
        self.x    = x #X_t
        self.xdot = xdot #Xdot_t
        self.u = u #[1,v,w]: u_t
        if model_type == "DD":
            self.model = DD_Model(L = L, R = R, M = M)
        elif model_type == "Ack2":
            self.model = UnicyleModel(L = L, R = R)            
        elif model_type == "Ack1":
            self.model = UnicyleModel_Ack1(L = L, R = R)
        self.logs = {"x":[],"u":[],"xdot":[],"F":[],"t":[],"step":[]}
        
    def update(self, u, dt=0.001, verbose = False):
        # ut = 1.0, w, a(vdot)
        self.u  = u        
        # [_,v,z] = self.u
        # [x,y,theta] = self.x
        
        # F(xt)
        F = self.model.get_Fx(self.x, sigma=0.0)
        
        # xdot_t
        self.xdot = F @ self.u
        
        # x_t+1
        x_tp1 = self.x + self.xdot * dt
        # x_tp1[2] = x_tp1[2]  % (2*np.pi)
        while x_tp1[2] > np.pi: 
            x_tp1[2] -= (2* np.pi)
        
        while x_tp1[2] <= -np.pi: 
            x_tp1[2] += (2* np.pi)
        

        #### x and xdot are valid for t ####
        self.logs["x"].append(self.x.flatten())
        self.logs["xdot"].append(self.xdot.flatten())
        self.logs["u"].append(self.u.flatten())
        self.logs["F"].append(F.flatten())
        self.logs["t"].append(self.t)
        self.logs["step"].append(self.step)     
        
        #### x is valid for t+1 ####
        self.step += 1
        self.t += dt        
        self.x  = x_tp1        
        if verbose:
            print(self)   

    def set_u(self, u):
        self.u = u        
    
    def set_x(self, x):
        self.x = x
    
    def set_xdot(self, xdot):
        self.xdot = xdot
        
    def get_x(self):
        x = self.logs["x"]
        return np.array([(x[i][0],x[i][1],x[i][2],x[i][3]) for i in range(len(x))])

    def get_xdot(self):
        xdot = self.logs["xdot"]
        return np.array([(xdot[i][0],xdot[i][1],xdot[i][2],xdot[i][3]) for i in range(len(xdot))])
        
    def get_u(self):
        u = self.logs["u"]
        return np.array([(1.0, u[i][1],u[i][2]) for i in range(len(u))])
    
    def get_Flogs(self):
        F = self.logs["F"]
        return np.array([f for f in F])

    def get_tp1(self):        
        return self.t
    def get_stepp1(self):        
        return self.step

    # def get_Fx(self):
    #     return self.model.get_Fx(self.x)
    
    def get_Veh_t(self, t):
        return type(self)(x=self.logs["x"][t], 
                          xdot=self.logs["xdot"][t],
                          u=self.logs["u"][t],
                          L=self.L)
    def __str__(self):
        return f"xt+1:{self.x},xdot:{self.xdot},ut:{self.u}"
