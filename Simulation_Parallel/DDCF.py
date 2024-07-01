# Import packages.
import autograd
import autograd.numpy as anp
import numpy as np

class control_Functions():
    def __init__(self, parms_con):
        self._cp    = parms_con['cp']       
        self.kbV1   = parms_con['kbV1']
        self.kbV2   = parms_con['kbV2']
        self.kbH    = parms_con['kbH']

        self.WR     = parms_con['car']['Wheel_R'] #Wheel_Radius
        self.RR     = parms_con['car']['Robot_R'] #Robot_Radius
        self.L      = parms_con['car']['Lenght']  #Robot_Lenght        
        self.car_name = parms_con['car']['name']
        self.car_model = parms_con['car']['car_model']

    def hx(self, xt, obstacle) -> np.ndarray:

        if   self.car_model == 'DD':
            [x, y, theta, w, v] = xt
        elif self.car_model == 'Ack2':
            [x, y, theta, v] = xt
        elif self.car_model == 'Ack1':
            [x, y, theta] = xt

        RR = self.RR
        if len(obstacle) == 3: #circle
            [Ox , Oy, Or] = obstacle

            dx = (Ox - x)
            dy = (Oy - y)
            if self.car_model == 'DD':
                d = self.L   #/ 5                
                dx -= d * anp.cos(theta)        
                dy -= d * anp.sin(theta)
                
                b = dx ** 2 + dy **2 -(RR+Or)**2
                
            elif self.car_model == 'Ack2':
                b = dx ** 2 + dy **2 -(RR+Or)**2
            elif self.car_model == 'Ack1':
                d = self.L  * 0.0 #/ 5
                dx -= d * anp.cos(theta)        
                dy -= d * anp.sin(theta)
                b = dx ** 2 + dy **2 -(RR+Or+d)**2
            return b 
        ## TODO
        if len(obstacle) == 5: #ellipsoid
            [Ox,Oy, a, b, t] = obstacle            
            
            a, b = a + RR, b + RR                     
            dx = (Ox - x)
            dy = (Oy - y)
            d = self.L  / 2 #* 0.0 #/ 5
            dx += d * anp.cos(theta)        
            dy += d * anp.sin(theta)
            p   = np.array([dx,dy])  
            rad = np.diag([1/a,1/b])
            rot = np.array([[np.cos(-t),-np.sin(-t)], 
                            [np.sin(-t),np.cos(-t)]])
            f = p @ rot @  rad              
            return (f @ f.T - 1.0) * (np.max([a,b]) ** 2)

    def hxp_(self, xt, obstacle, _Fx):        
        _Fx  = _Fx.reshape(-1,3)
        _f   = _Fx[:,0]
        
        _Gh = self.Ghx(xt, obstacle)
        _hdot = _Gh.T @ _Fx
        
        return _hdot[0]

    def hxp(self, xt, obstacle, _Fx):        
        _Fx  = _Fx.reshape(-1,3)
        _f   = _Fx[:,0]
        
        _Gh = self.Ghx(xt, obstacle)
		## rd1
        if self.car_model =='Ack1':
            _hdot = _Gh.T @ _Fx      
            return _hdot[0]
		## rd2
        _hdot = _Gh.T @ _f    
        return _hdot


    def hxpp(self, xt, obstacle, _Fx, _JFx):
        _Fx  = _Fx.reshape(-1,3)
        _f   = _Fx[:,0]
        if self.car_model == 'DD':
            _JFx = _JFx.reshape(-1,5)
            _Jf  = _JFx[[0,3,6,9,12],:]
        elif self.car_model =='Ack2':
            _JFx = _JFx.reshape(-1,4)
            _Jf  = _JFx[[0,3,6,9],:]
        elif self.car_model =='Ack1':
            _JFx = _JFx.reshape(-1,3)
            _Jf  = _JFx[[0,3,6],:]
        
        _Gh = self.Ghx(xt, obstacle)
        _Jh = self.Jhx(xt, obstacle)
        _J_Gh_f  = _Jh @ _f + _Gh.T[0] @ _Jf
        
        _hdotdot = _J_Gh_f  @ _Fx
        return _hdotdot
        
    def CBC1(self, xt, obstacle,  _Fx : np.ndarray) -> np.ndarray:
        
        [k0,k1] = self.kbH
        
        _h    = self.hx (xt, obstacle)
        _hdot = self.hxp(xt, obstacle, _Fx)        
        _cbc1 = _hdot + k0 * _h
        return _cbc1
    
    def CBC2(self, xt, obstacle, _Fx : np.ndarray, _JFx : np.ndarray):
        [k0,k1] = self.kbH
        if self.car_model == 'Ack2':
            _h    = self.hx (xt, obstacle)
            _hdot = self.hxp(xt, obstacle, _Fx)        
            _hdotdot = self.hxpp(xt, obstacle, _Fx, _JFx)
            _cbc2 = _hdotdot
            _cbc2[0] += (k0 + k1) * _hdot + (k1 * k0) * _h        
        elif self.car_model == 'Ack1':
            _h    = self.hx (xt, obstacle)
            _hdot = self.hxp(xt, obstacle, _Fx)        
            _cbc2 = _hdot
            _cbc2[0] += k0 * _h     

        return _cbc2

    def CBC2_sai1(self, xt, obstacle, _Fx : np.ndarray, _JFx : np.ndarray):
        [k0,k1] = self.kbH
        _h    = self.hx (xt, obstacle)
        _hdot = self.hxp_(xt, obstacle, _Fx)        
        
        _cbc2 = _hdot
        _cbc2[0] += k0 * _h        
        return _cbc2
    
    def CBC2_sai2(self, xt, obstacle, _Fx : np.ndarray, _JFx : np.ndarray):
        [k0,k1] = self.kbH
        _h    = self.hx (xt, obstacle)
        _hdot = self.hxp(xt, obstacle, _Fx)        
        _hdotdot = self.hxpp(xt, obstacle, _Fx, _JFx)
        _cbc2 = _hdotdot
        _cbc2[0] += (k0+k1) * _hdot + (k1*k0) * _h        
        return _cbc2
    
    def CBC(self, xt, obstacle, u, _Fx : np.ndarray, _JFx : np.ndarray):            
        _cbc2 = self.CBC2(xt, obstacle, _Fx, _JFx)
        if _cbc2.shape == u.shape:
            _cbc2 = _cbc2 @ u
        else:
            _cbc2 = _cbc2[0] + _cbc2[1:] @ u
        return _cbc2

    def hx_out (self, xt, obstacle) -> np.ndarray:
        if   self.car_model == 'DD':
            [x, y, theta, w, v] = xt
        elif self.car_model == 'Ack2':
            [x, y, theta, v] = xt
        elif self.car_model == 'Ack1':
            [x, y, theta] = xt

        RR = self.RR
        if len(obstacle) == 3: #circle
            [Ox , Oy, Or] = obstacle
            
            dx = (Ox - x)
            dy = (Oy - y)
            if self.car_model == 'DD':
                d = self.L   #/ 5
                
                dx += d * anp.cos(theta)        
                dy += d * anp.sin(theta)
                
                b = (Or)**2 - dx ** 2 - dy **2 
            elif self.car_model == 'Ack2':
                b = (Or)**2 - dx ** 2 - dy **2 
            elif self.car_model == 'Ack1':
                d = self.L  #/ 5
                dx -= d * anp.cos(theta)        
                dy -= d * anp.sin(theta)
                b = (Or - RR)**2 - dx ** 2 - dy **2 
            return b 
        ## TODO
        if len(obstacle) == 5: #ellipsoid
            [Ox,Oy, a, b, t] = obstacle            
            a, b = a - RR, b - RR                     
            dx = (Ox - x)
            dy = (Oy - y)
            d = self.L  / 2.0 #/ 5
            dx -= d * anp.cos(theta)        
            dy -= d * anp.sin(theta)
            p   = np.array([dx,dy])  
            rad = np.diag([1/a,1/b])
            rot = np.array([[np.cos(-t),-np.sin(-t)], 
                            [np.sin(-t),np.cos(-t)]])
            f = p @ rot @  rad              
            return (1.0 - f @ f.T ) * (np.min([a,b]) ** 2)

    def hxp__out(self, xt, obstacle, _Fx):        
        _Fx  = _Fx.reshape(-1,3)
        _f   = _Fx[:,0]
        
        _Gh = self.Ghx_out(xt, obstacle)
        _hdot = _Gh.T @ _Fx
        return _hdot[0]

    def hxp_out (self, xt, obstacle, _Fx):        
        _Fx  = _Fx.reshape(-1,3)
        _f   = _Fx[:,0]

        _Gh = self.Ghx_out(xt, obstacle)
        ## rd1
        if self.car_model =='Ack1':
            _hdot = _Gh.T @ _Fx      
            return _hdot[0]
		## rd2
        _hdot = _Gh.T @ _f    
        return _hdot
    
    def hxpp_out(self, xt, obstacle, _Fx, _JFx):
        _Fx  = _Fx.reshape(-1,3)
        _f   = _Fx[:,0]
        if self.car_model == 'DD':
            _JFx = _JFx.reshape(-1,5)
            _Jf  = _JFx[[0,3,6,9,12],:]
        elif self.car_model =='Ack2':
            _JFx = _JFx.reshape(-1,4)
            _Jf  = _JFx[[0,3,6,9],:]
        elif self.car_model =='Ack1':
            _JFx = _JFx.reshape(-1,3)
            _Jf  = _JFx[[0,3,6],:]
        
        _Gh = self.Ghx_out(xt, obstacle)
        _Jh = self.Jhx_out(xt, obstacle)
        _J_Gh_f  = _Jh @ _f + _Gh.T[0] @ _Jf
        
        _hdotdot = _J_Gh_f  @ _Fx
        return _hdotdot
    
    def Ghx_out(self, xt, obstacle) -> np.ndarray:
        _GhxF = autograd.grad(self.hx_out,0)         
        _Ghx = _GhxF(xt,obstacle)
         
        return np.atleast_2d(_Ghx).T

    def Jhx_out(self, xt, obstacle) -> np.ndarray:
        _JhxF = autograd.jacobian(autograd.jacobian(self.hx_out,0) ,0 )
        _Jhx = _JhxF(xt,obstacle)
        
        return _Jhx
    
    def CBC2_out(self, xt, obstacle, _Fx : np.ndarray, _JFx : np.ndarray):
        [k0,k1] = self.kbH
        if self.car_model == 'Ack2':
            _h    = self.hx_out (xt, obstacle)
            _hdot = self.hxp_out(xt, obstacle, _Fx)        
            _hdotdot = self.hxpp_out(xt, obstacle, _Fx, _JFx)
            _cbc2 = _hdotdot
            _cbc2[0] += (k0 + k1) * _hdot + (k1 * k0) * _h        
        elif self.car_model == 'Ack1':
            _h    = self.hx_out (xt, obstacle)
            _hdot = self.hxp_out(xt, obstacle, _Fx)        
            _cbc2 = _hdot
            _cbc2[0] += k0 * _h     

        return _cbc2

    def Ghx(self, xt, obstacle) -> np.ndarray:
        _GhxF = autograd.grad(self.hx,0)         
        _Ghx = _GhxF(xt,obstacle)
         
        return np.atleast_2d(_Ghx).T

    def Jhx(self, xt, obstacle) -> np.ndarray:
        _JhxF = autograd.jacobian(autograd.jacobian(self.hx,0) ,0 )
        _Jhx = _JhxF(xt,obstacle)
        
        return _Jhx

    def velx(self, xt, vmax) -> np.ndarray:
        # [x, y, theta, ur, ul] = xt
        if self.car_model =='DD':
            [x, y, theta, w, v] = xt
        else:
            [x, y, theta, v] = xt

        b = (vmax ** 2 - v ** 2)
        return b

    def Gvelx(self, xt, vmax) -> np.ndarray:
        _GhxF = autograd.grad(self.velx,0)         
        _Ghx = _GhxF(xt,vmax)
         
        return np.atleast_2d(_Ghx).T

    def Jvelx(self, xt, vmax) -> np.ndarray:
        _JhxF = autograd.jacobian(autograd.jacobian(self.velx,0) ,0 )
        _Jhx = _JhxF(xt,vmax)
        
        return _Jhx

    def Wx(self, xt, wmax) -> np.ndarray:        
        if self.car_model =='DD':
            [x, y, theta, w, v] = xt
        else:
            [x, y, theta, v] = xt

        b = wmax ** 2 - w ** 2
        return b

    def GWx(self, xt, wmax) -> np.ndarray:
        _GhxF = autograd.grad(self.Wx,0)         
        _Ghx = _GhxF(xt,wmax)
         
        return np.atleast_2d(_Ghx).T

    def JWx(self, xt, wmax) -> np.ndarray:
        _JhxF = autograd.jacobian(autograd.jacobian(self.Wx,0) ,0 )
        _Jhx = _JhxF(xt,wmax)
        
        return _Jhx

    def V1x(self, xt, xd) -> np.ndarray:
        if   self.car_model =='DD':
            [x, y, theta, w, v] = xt
        elif self.car_model =='Ack2':
            [x, y, theta, v] = xt
        elif self.car_model =='Ack1':
            [x, y, theta]    = xt

        if self.car_model =='DD':
            dx = -(xd[0] - x )
            dy = -(xd[1] - y )

            d = self.L

            b = (dx ** 2) + (dy **2)

        elif self.car_model == 'Ack2':
            dx = xd[0] - x
            dy = xd[1] - y
            b = (0.5 * (dx ** 2) + 0.5 * (dy **2) )

            xd,yd = xd[:2]
            dx,dy = x-xd, y- yd
            vx = v * anp.cos(theta)
            vy = v * anp.sin(theta)
            alpha = 1.0       
            beta  = 1.0
            b = dx ** 2 + dy ** 2 + alpha * (vx + beta * dx) ** 2 + alpha * (vy + beta * dy) ** 2 #+ 0.001 * th ** 2

        elif self.car_model == 'Ack1':
            dx = xd[0] - x
            dy = xd[1] - y
            d = self.L
            dx -= d * anp.cos(theta)        
            dy -= d * anp.sin(theta)
            b = (0.5 * (dx ** 2) + 0.5 * (dy **2))
            
        return   b

    def GV1x(self, xt, xd) -> np.ndarray:
        _GVF = autograd.grad(self.V1x,0)         
        _GVx = _GVF(xt,xd)

        return np.atleast_2d(_GVx).T

    def HV1x(self, xt, xd) -> np.ndarray:
        _JVxF = autograd.jacobian(autograd.jacobian(self.V1x,0) ,0 )
        _JVx = _JVxF(xt, xd)        
        return _JVx

    def CLC_detail(self, _Fx : np.ndarray, _JFx : np.ndarray, xt, xd, u) -> np.ndarray:
        """
        _Fx: ndarray   5x3
        _JFx: ndarray 15x4
        """
        _Fx = _Fx.reshape(-1,3)        
        _f  = _Fx[:,0]
        _g  = _Fx[:,1:]
        if self.car_model == 'DD':
            _JFx = _JFx.reshape((-1,5))
            _Jf = _JFx[[0,3,6,9, 12],:]
            _Jg = _JFx[[1,2,4,5,7,8,10,11,13,14],:]
        else:
            _Jf = _JFx[[0,3,6,9],:]
            _Jg = _JFx[[1,2,4,5,7,8,10,11],:]

        _V  = self.V1x (xt, xd)
        _GV = self.GV1x(xt, xd)
        _HV = self.HV1x(xt, xd)
        # RELATIVE DEGREE 2
        _H_GV_f  = _HV.T @ _f + _GV.T @ _Jf
        _Vdot = _GV.T @ _f
        _Vdotdot = _H_GV_f  @ _Fx
        _clc = _Vdotdot[0].copy()
        _clc[0] += self.kbV1[0] * _Vdot[0] + self.kbV1[1] * _V
        return _clc @ u, _clc, _H_GV_f, _Vdotdot, _Vdot, _V

    
    def CLC1(self, _Fx : np.ndarray, _JFx : np.ndarray, xt, xd) -> np.ndarray:
        """
        _Fx: ndarray   5x3
        _JFx: ndarray 15x4
        """
        [kb0,kb1] = self.kbV1
        
        _V  = self.V1x (xt, xd)
        _GV = self.GV1x(xt, xd)
        
        _Vdot = _GV.T @ _Fx
        _clc = _Vdot[0].copy()
        _clc[0] += (kb0) * _V
        return _clc

    def CLC2(self, _Fx : np.ndarray, xt, xd) -> np.ndarray:        
        _V  = self.V2x (xt, xd)
        _GV = self.GV2x(xt, xd)

        # RELATIVE DEGREE 1
        kbv2 = self.kbV2[0]
        
        _Vdot = _GV.T @ _Fx
        _clc = _Vdot[0].copy()
        _clc[0] += kbv2 * _V        
        return _clc

    def CLC1_(self, _Fx : np.ndarray, _JFx : np.ndarray, xt, u : np.ndarray, xd):            
        _clc1 = self.CLC1(_Fx, _JFx, xt, xd)
        if _clc1.shape == u.shape:
            _clc1 = _clc1 @ u
        else:
            _clc1 = _clc1[0] + _clc1[1:] @ u
        return _clc1

    def CLC2_(self, _Fx : np.ndarray, xt, u : np.ndarray, xd):            
        _clc2 = self.CLC2(_Fx, xt, xd)
        if _clc2.shape == u.shape:
            _clc2 = _clc2 @ u
        else:
            _clc2 = _clc2[0] + _clc2[1:] @ u
        return _clc2
        
