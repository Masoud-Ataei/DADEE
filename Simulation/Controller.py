# Import packages.
# from msvcrt import kbhit
# from multiprocessing.sharedctypes import Value
# from re import U
import cvxpy as cp
import numpy as np
from Trajectory import *
from DDCF import *
from dataclasses import dataclass
# from SWAG_Orig import *

@dataclass
class controller_vars:
    """Class for keeping track of the vechile state."""
    u      = cp.Variable(2)
    delta1 = cp.Variable(1)
    delta2 = cp.Variable(1)
    l      = cp.Variable(1)
        

class controller_DetFx():
    def __init__(self, parms_con, controlFunction : control_Functions, traj : trajectory):
    
        self.traj    = traj
        #### Optimization Parameters ####
        self.lamba  = parms_con['lambda']
        self.Lx     = parms_con['Lx']
        self.ktilde = parms_con['ktilde']

        self.var    = controller_vars()
        self.prob   = cp.Problem(cp.Minimize(cp.norm2(self.var.u)))
        self.model  = {"P": [],"q": [],"G": [],"h": []}
        self.CF     = controlFunction
        
        self.status = "init"
        self.vmax  = parms_con['vmax']
        self.kvel  = parms_con['kvel']
        self.wmax  = parms_con['wmax']
        self.kw    = parms_con['kw']
        self.u_lim = parms_con['u_lim']
        self.out_circ = parms_con['out_circ']
        self.SOCP  = parms_con['SOCP']
        
    def set_optimization_parameters(self, ktilde = np.zeros(2), lamba = np.ones(2), Lx = np.ones(2)):
        #### Optimization Parameters ####
        self.lamba  = lamba
        self.Lx     = Lx
        self.ktilde = ktilde



    def estimate_u_EHRD(self, _VecFx : np.ndarray, _JFx : np.ndarray, xt, FxEstimator, obstacles,
                        verbose=False, ktilde=np.zeros(2), clc_cons=False):
        self.status = "running"
        Lx          = self.Lx
        self.ktilde = ktilde
        lamba       = self.lamba
        
        wmax        = self.wmax
        kw          = self.kw
        _u          = np.array([1.0,0.0,0.0])        
        end, xd     = self.traj.get_xd(xt)
        _delta1     = np.array([0.0])  
        _delta2     = np.array([0.0])  
        _Fx         = _VecFx.reshape(-1,3)
        self.var    = controller_vars()
        u, delta1, delta2 = self.var.u,self.var.delta1, self.var.delta2
        l           = cp.Variable(1)
        _f = _Fx[:,0]
        _g = _Fx[:,1:]        
        ## Ack1
        _Jf = _JFx[[0,3,6],:]
        _Jg = _JFx[[1,2,4,5,7,8],:]
        
        Stat_CBC = [np.zeros(3) , np.zeros(9)]
        lamba1, lamba2 = lamba
        if np.linalg.norm(xd[:2] - xt[:2]) <= 1.0:
            lamba1 = lamba1 * 1 / np.linalg.norm(xd[:2] - xt[:2])
        
        if not end:
            objective = l
            constrains = []
            _cp = self.CF._cp
            Stat_CBC = []
            scale = np.diag([100.0 , 1.0])
            scale /= np.linalg.norm(scale)
            
            objective = cp.sum_squares(u @ scale - ktilde) 
            
            if clc_cons:
                
                _clc1 = self.CF.CLC1(_Fx, _JFx, xt, xd)
                clc1 = _clc1[0] + _clc1[1:] @ u

                objective += lamba1 * clc1
                
            margin = 0.2

            # CBC_out 
            if len(self.out_circ) > 0:
                # #### Deterministic ###
                obs_out = np.array(self.out_circ)
                if FxEstimator:
                    [CBC2mean, CBC2sigma]  = FxEstimator.get_CBC_out(self.CF, xt, obs_out)
                    
                else:
                    CBC2mean  = self.CF.CBC2_out (xt, obs_out, _Fx, _JFx)
                    CBC2sigma = np.zeros((3,3)) + np.eye(3) * 1e-3

                if (np.linalg.norm(CBC2sigma) <= 1e-10 ):                    
                    CBC2sigma_half = CBC2sigma * 0 + np.eye(3) * 1e-6

                CBC2sigma = CBC2sigma + np.eye(3) * 1e-3
                
                CBC2sigma_half = np.linalg.cholesky( CBC2sigma)

                Stat_CBC.append(CBC2mean.flatten())
                Stat_CBC.append(CBC2sigma_half.flatten())
                
                CBCmean = CBC2mean[0] + CBC2mean[1:] @ u
                Var_CBC = CBC2sigma_half[0,:] + u @ CBC2sigma_half[1:, :]
                if self.SOCP:
                    constrains.append(cp.SOC(CBCmean + margin, _cp * Var_CBC))
                else:
                    constrains.append(CBCmean >= 0)

            for obs in obstacles:
                if FxEstimator:
                    [CBC2mean, CBC2sigma]  = FxEstimator.get_CBC(self.CF, xt, obs)
                
                else:
                    CBC2mean  = self.CF.CBC2(xt, obs, _Fx, _JFx)
                    CBC2sigma = np.zeros((3,3)) + np.eye(3) * 1e-3

                if (np.linalg.norm(CBC2sigma) <= 1e-10 ):                    
                    CBC2sigma_half = CBC2sigma * 0 + np.eye(3) * 1e-6
                
                CBC2sigma = CBC2sigma + np.eye(3) * 1e-3
                CBC2sigma_half = np.linalg.cholesky( CBC2sigma)
                
                Stat_CBC.append(CBC2mean.flatten())
                Stat_CBC.append(CBC2sigma_half.flatten())
                
                CBCmean = CBC2mean[0] + CBC2mean[1:] @ u
                Var_CBC = CBC2sigma_half[0,:] + u @ CBC2sigma_half[1:, :]
                
                if self.SOCP:
                    constrains.append(cp.SOC(CBCmean + margin, _cp * Var_CBC))
                else:
                    constrains.append(CBCmean >= 0)
                    
            u_lim = self.u_lim.copy()
            
            constrains.append(u[0] <=  u_lim[0])   #w >= wmin    
            constrains.append(u[0] >= -u_lim[0])   #w >= wmin    
            constrains.append(u[1] <=  u_lim[1])   #a >= amin   
            constrains.append(u[1] >= -u_lim[1])   #a >= amin               

            prob = cp.Problem(cp.Minimize(objective), constrains)
            try:
                prob.solve(solver = cp.ECOS, verbose=verbose )   
                             
                if verbose:
                    print(prob.status  , f"u:{u.value},delta1:{delta1.value},delta2:{delta2.value}")
                        
                if prob.status not in ["infeasible", "unbounded", "infeasible_inaccurate", "optimal_inaccurate"]:                                
                    _u[1]     = u[0] .value
                    _u[2]     = u[1] .value
                    if clc_cons and len(obstacles)>0:
                        _delta1[0] = 0.0 #delta1.value[0]
                        pass
                else:
                    _u[1] = 0.0
                    _u[2] = 0.0
                # print("CVXST", prob.status, _u, _delta1, _delta2)
                self.prob      = prob
                self.var.u     = u
                self.var.delta1= delta1
                self.var.delta2= delta2
                
                self.status = prob.status
            except:
                print("solver crashed")
                _u[1], _u[2] = 0.0, 0.0
                _delta1, _delta2 = -1, -1
        else:
            self.status = "rest"
        # Stat_CBC = np.concatenate(Stat_CBC).flatten()
        return _u, _delta1, _delta2, Stat_CBC
