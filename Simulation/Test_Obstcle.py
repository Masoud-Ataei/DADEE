# Import packages.
import json_tricks as json
import datetime
import os
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Unicycle_DD import Simulator, DD_Model, vehicle_class, UnicyleModel, UnicyleModel_Ack1
from Trajectory import trajectory
from DDCF import control_Functions
from Controller import controller_DetFx
from DDEstimator  import estimate_DDFx #Ensemble and BNN use this
from ReplayMemory import ReplayMemory
import pybullet as p
import glob, os, shutil
from make_circle import make_circle
from PyBullet_Functions import init_pybullet, update_simulator
import tensorflow as tf
from utils import Uncertainty_metrics
from matplotlib.patches import Circle, Ellipse
    
def obs_ray(results):    
    points = []
    res = np.array([list(r[:3]+r[3]+r[4]) for r in results if r[0]>=0])
    if res.shape[0] > 0:
        indx0 = np.where(np.linalg.norm( res[:-1,3:5]-res[1:,3:5], axis=1) > 0.3)[0]+1
        l = np.split(res, indx0)
        if len(l)>1 and np.linalg.norm( res[0,3:5]-res[-1,3:5]) < 0.3:
            l[0] = np.concatenate(( l[0], l[-1]))
            del l[-1]                
        for l0 in l:
            points.append(l0[:,3:5])  
    return points

def run_robot(sim : Simulator, parms, Traj : trajectory, control : controller_DetFx, CF: control_Functions ,
              FxEstimator : estimate_DDFx, memory : ReplayMemory, 
              logFile = False, plots = False, verbose = False, save_model = False, save_plot = False):
    
    max_time_hour = parms['run']['max_time_hour']
    max_time = 60 * 60 * max_time_hour
    run_start_time = time.time()  # remember when we started

    veh_model = sim.model
    parms_env = parms['env']
    run       = parms['run']['run']
    model_det = parms['run']['model_det']

    uem = ''
    if parms['NN']['DEUP']:
        uem = 'DEUP_'
    elif parms['NN']['MLLV']:
        uem = 'MLLV_'
    else:
        uem = ''

    now   = datetime.datetime.now()
    if not os.path.exists("log/"):
        os.mkdir("log/")
    fpath = "log/" + uem + model_det + "/"
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    
    use_real_obstacle = parms['env']['use_real_obstacle']
    lr = parms['NN']['lr']

    time_step= parms['sim']['time_step']
    num_steps= parms['sim']['num_steps']
    smpl_itr = parms['sim']['smpl_itr']
    car_name = parms['car']['name']
    car_L    = parms['car']['Lenght'] #Robot Radius (Lenght)
    car_WR    = parms['car']['Wheel_R'] #Wheel Radius
    car_RR    = parms['car']['Robot_R'] #Robot Radius
    car_M     = parms['car']['Mass'] #
    car_model = parms['car']['car_model'] #

    if car_name != 'Simple':
        numRays = parms['sim']['n_Rays']     # LIDAR scan ray numbers
        rayFrom = np.zeros((numRays,3))
        rayTo   = np.zeros((numRays,3))
        rayLen  = parms['sim']['rayLen']
        tRayFrom = parms['sim']['t_0']
        tRayRange= parms['sim']['t_n']
        pass
    
    print(f'controler:{model_det} car_name:{car_name} model:{car_model}')    
    fpath = "log/" + uem + model_det + "/" + now.strftime("%Y%m%d_%H%M%S") + "_" + car_model + "_" + car_name + f"_cp{parms['controler']['cp']:.2f}".replace('.','_') + "/"
    parms_env['start-time']   = now
    parms_env['fpath'] = fpath
    sim_images = parms['sim']['save_images']
    sim_img_w = parms['sim']['img_w']
    sim_img_h = parms['sim']['img_h']
    u1lim, u2lim = parms['controler']['u_lim']
    out_circ = parms['controler']['out_circ']
    # u2lim = parms['controler']['alim'][1]
    
    try:
        os.mkdir(fpath)
        os.mkdir(fpath + 'models/')
        os.mkdir(fpath + 'code_bk/')
        os.mkdir(fpath + 'figs/')
        if sim_images:
            os.mkdir(fpath + 'sim_imgs/')
        print("Directory '%s' created successfully" % fpath)        
    except OSError as error:
        print("Directory '%s' can not be created" % fpath)

    try:    
        files = glob.iglob(os.path.join("", "*.py"))
        for file in files:
            if os.path.isfile(file):
                shutil.copy2(file, fpath + 'code_bk/')
        print("Backed up py files.")
    except:
        print("Couldn't back up py files.")

    json.dump(parms, open(fpath + "init_vars.json",'w'))

    ################################ Local Variables ################################        
    inp_obs = parms_env['obstacles'].copy()
    d = 3
    if len(inp_obs) > 0:
        if len(inp_obs[0]) > 5:
            d = 5
    obstacles = [obs[:d] for obs in inp_obs]    
    
    logs = {}
    logs['path'] = parms_env['path'].copy(); 
    logs['vel_car']   = []
    logs['real-time'] = []
    logs['sim-time']  = []    
    logs['step']    = []
    logs['indx'   ] = []
    logs['Stat_CBC'] = []
    logs['CnvxSt']   = []
    logs['ktilde']   = []
    logs['CBC1_x'  ]= []
    logs['CBC2_x'  ]= []
    logs['h_x'   ]= []
    logs['hp_x'  ]= []
    logs['hpp_x' ]= []
    logs['V1_x'      ] = []
    logs['V2_x'      ] = []
    logs['Err'       ] = []
    logs['CLC1_x'    ] = []
    logs['CLC2_x'    ] = []
    logs['delta1'    ] = []
    logs['delta2'    ] = []
    
    logs['FElist'    ] = []
    logs['FSlist'    ] = []
    logs['FDlist'    ] = []
    logs['FVlist'    ] = []
    logs['FVarlist'  ] = []   
    logs['FVSlist'  ] = []   
    logs['JElist'    ] = []
    logs['JSlist'    ] = []
    logs['FWl-FWr'] = []
    logs['results'] = []
    logs['weights'] = []
    logs['hited'] = False
    logs['Not_optimal'] = []
    
    nZeroVel = 0
    uinit = np.zeros(3); uinit[0] = 1.0
    u  = uinit.copy()
    xt = parms_env['x0'].copy()
    sim_time = 0
    d1,d2 =0,0
    stat_cbc = []

    ################################ Connect Pybullet simulation ################################
    if car_name != 'Simple':
        bullet_flag = parms['sim']['type'] # no_gui, gui, mp4
        id_lookup = init_pybullet(bullet_flag, time_step, False, fpath, Traj.path)        
        car = id_lookup['Husky']

        ray_ends = []
        angle    = [] 

        for i in range(numRays):
            tmp = -tRayFrom + tRayRange * i/numRays
            ray_end = np.array([rayLen * np.sin(tmp), rayLen * np.cos(tmp),0])
            
            angle.append(tmp)
            ray_ends.append(ray_end)

        for _ in range(240):                                                # wait several secends for loading the environment
            p.stepSimulation()
            time.sleep(1./960.)

        if car_model == 'DD' :
            p.setJointMotorControl2(car,0,p.VELOCITY_CONTROL,force = 0)
            p.setJointMotorControl2(car,1,p.VELOCITY_CONTROL,force = 0)
        
        position, orientation = p.getBasePositionAndOrientation(car)
        orien = p.getEulerFromQuaternion(orientation)        
        LIDARpos = position + np.array([0.0, 0.0, 0.45])
        orientation_mat = p.getMatrixFromQuaternion(orientation)
        for i in range (numRays):
            rayFrom[i,:] = LIDARpos
            rayTo[i,:] = LIDARpos + np.dot(np.array(orientation_mat).reshape(3,3),ray_ends[i])

        results = p.rayTestBatch(rayFrom,rayTo)
        obs_pnts = obs_ray(results)
        if use_real_obstacle:
            obstacles = []
            for pnts in obs_pnts:                
                obs_circle = make_circle(pnts)                
                if obs_circle:
                    obstacles.append(obs_circle)
                
            obstacles = np.array(obstacles)
        if sim_images:        
            fov = 20
            aspect = sim_img_w / sim_img_h
            near = 0.05
            far = 20
            view_matrix = p.computeViewMatrix([6,-6,6], [0, 0, 0.0], [-1, 1, 1])
            projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

            img_infor = p.getCameraImage(sim_img_w, sim_img_h, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            imgrgb = img_infor[2][:, :, :3]
            
            matplotlib.image.imsave(fpath + 'sim_imgs/pic_.png', imgrgb)                                
        pass
    ################################################################
    exit = True
    while exit:
        if FxEstimator:
            _FxE , _FxV = FxEstimator.predictFx(xt.reshape(1,-1), s = 0) 
            
            if FxEstimator is not None:
                _FxVS = FxEstimator.S
            _JFxE = FxEstimator.predict_mean_Fx_gradient(xt.reshape(1,-1)) [0]
        else:
            _FxE  = veh_model.get_Fx (xt)
            _JFxE = veh_model.get_JFx(xt)
        exit = False
        online = parms['NN']['online']
        ### TODO
        for i,obs in enumerate(obstacles):
            hx   = CF.hx  (xt, obs)
            if car_model != 'Ack1':
                cbc1 = CF.CBC1(xt, obs, _FxE)
            cbc2 = CF.CBC2(xt, obs, _FxE, _JFxE) @ uinit
            if car_model != 'Ack1':
                print(f"obs{i}: {obs[:2]} : hx:{hx},cbc1:{cbc1},cbc2:{cbc2}.")
            else:
                print(f"obs{i}: {obs[:2]} : hx:{hx},cbc2:{cbc2}.")
            if hx <=0:
                print(f"Simulation started in an unsafe region hx{hx}!! By obstacle {obs[:2]}.")
                exit = True
                # sys.exit()
            if  cbc2 <=0:
                print(f"Simulation started in an unsafe region CBC2={cbc2}!! By obstacle {obs[:2]}.")
                exit = True
                # sys.exit()
            if car_model != 'Ack1':
                if  cbc1 <=0:
                    print(f"Simulation started in an unsafe region CBC1={cbc1}!! By obstacle {obs[:2]}.")
                    exit = True
                    # sys.exit()
        if exit:
            FxEstimator.reset()
        
    if plots:
        plt.style.use('ggplot')

        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(1, figsize=(5,5))
        ax = fig.add_subplot(111)

        fig2 = plt.figure(2, figsize=(5,5))        
        ax21 = fig2.add_subplot(231)
        ax22 = fig2.add_subplot(232)
        ax23 = fig2.add_subplot(233)
        ax24 = fig2.add_subplot(234)
        ax25 = fig2.add_subplot(235)
        ax26 = fig2.add_subplot(236)

        plt.show()
    logs['losslist']  = []
    
    max_iter = parms['max_iter']
    xt       = parms_env['x0']
    dt       = parms['dt']
    
    logs['x']    = []
    logs['u']    = []
    logs['xdot'] = []
    logs['metric'] = [['conf','area', 'MLP', 'RL2E', 'RMSCE']]
    logs['excecute_time'] = {'train'     : [],
                             'inference' : [],
                             'control'   : []}
    json.dump(logs, open(fpath + "logs.json",'w'))
    
    x_new    = xt.copy()
    _xdot    = xt * 0
    ccp      = 0.01
    control.status = "optimal"
    itr_optimal = 0
    exit = False
    acc, val = 0.0, 0.0
    for itr in range(max_iter):
        ## Get car position and orientation
        ### TODO
        if parms['run']['car_name'] == 'Simple':     
            _xdot = (sim.x - xt)
            _xdot[2] = np.arctan2(np.sin(_xdot[2]), np.cos(_xdot[2]))
            _xdot /= dt                    
            xt = sim.x.copy()

        else:            
            xt = x_new.copy()
            position, orientation = p.getBasePositionAndOrientation(car)
            orien = p.getEulerFromQuaternion(orientation)
            x_,y_ = position[:2]
            theta_ = orien[2]
            _vx   = (x_- xt[0]) / dt
            _vy   = (y_- xt[1]) / dt
            if smpl_itr:
                _vx   = (x_- xt[0]) / dt / num_steps
                _vy   = (y_- xt[1]) / dt / num_steps
            vel_2 = np.sqrt(_vx **2 + _vy **2 )            
            th = xt[2]
            th = theta_
            
            vel_trans, vel_ang = p.getBaseVelocity(car)
            _vx,_vy,_ = vel_trans
            vel_2 = _vx * np.cos(th) + _vy * np.sin(th)
            vel_trans, vel_ang = p.getBaseVelocity(car)
            _vx, _vy, _ = vel_trans
            
            vel_ = np.linalg.norm(vel_trans[:2])
            logs['vel_car'].append([vel_,vel_2])
            if car_model == 'Ack2':
                x_new = np.array([x_,y_,theta_, vel_2])
            elif car_model == 'Ack1':
                x_new = np.array([x_,y_,theta_])
            elif car_model == 'DD':
                w_ = (theta_ - xt[2]) 
                w_ = np.arctan2(np.sin(w_), np.cos(w_))/ dt #/ num_steps
                if smpl_itr:
                    w_ = np.arctan2(np.sin(w_), np.cos(w_))/ dt / num_steps
                x_new = np.array([x_,y_,theta_, w_, vel_2])
            
            _xdot = (x_new - xt)
            _xdot[2] = np.arctan2(np.sin(_xdot[2]), np.cos(_xdot[2]))
            _xdot = _xdot / dt   #/ num_steps             
            if smpl_itr:
                _xdot = _xdot / dt / num_steps

        if FxEstimator:
            if control.status != 'infeasible':
                memory.push_diff(xt.copy().astype(dtype = np.float32), u.copy().astype(dtype = np.float32), _xdot.copy().astype(dtype = np.float32)) 
        
        logs['x'   ].append(xt   .copy())
        logs['u'   ].append(u    .copy())
        logs['xdot'].append(_xdot.copy()) 
        
        if control.status  == 'optimal':
            itr_optimal += 1
        
        # # ################################ LIDAR scan simulation ################################
        # # ################################### Obstacles #########################################        
        if parms['run']['car_name'] != 'Simple': 
            LIDARpos = position + np.array([0.0, 0.0, 0.45])
            orientation_mat = p.getMatrixFromQuaternion(orientation)
            for i in range (numRays):
                rayFrom[i,:] = LIDARpos
                rayTo[i,:] = LIDARpos + np.dot(np.array(orientation_mat).reshape(3,3),ray_ends[i])

            results = p.rayTestBatch(rayFrom,rayTo)
            logs['results'].append(results)
            obs_pnts = obs_ray(results)
            if use_real_obstacle:                    
                obstacles = []
                for pnts in obs_pnts:
                    obs_circle = make_circle(pnts)                
                    
                    if obs_circle:
                        obstacles.append(obs_circle)
                    
                obstacles = np.array(obstacles)
            pass
        # # #######################################################################################

        epochs = 20
        stat = 'infeasible'
        Force = False
        Not_optimal = 0            
        while (stat != 'optimal') and (not exit): # and (Not_optimal >= 0):
            time_elapsed = time.time() - run_start_time
            logs['excecute_time']['train'].append(-time.time())
            if time_elapsed > max_time:
                exit = True
                print('!!!  max time is reached!!!')
                break

            if Not_optimal <= -100:
                exit = True
                print('!!!  Not optimal reach 100!!!')
                break
            
            if FxEstimator and itr >0 and online:
                if model_det == 'Baseline' or model_det == 'Res' :
                    if (-Not_optimal) % 10 == 9:
                        Nl = len(FxEstimator.model_mu[0].trainable_variables)
                        RNl = np.random.randint(Nl)
                        VVV = FxEstimator.model_mu[0].trainable_variables[RNl]
                        VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        
                        if (parms['NN']['DEUP'] or parms['NN']['MLLV']):
                            Nl = len(FxEstimator.model_var[0].trainable_variables)
                            RNl = np.random.randint(Nl)
                            VVV = FxEstimator.model_var[0].trainable_variables[RNl]
                            VVV.assign_add(np.random.randn(*VVV.shape) * 0.01)
                        
                        print(-Not_optimal + 1 )  
                        epochs = 1

                    if len(memory) >=2:
                        if itr % 10 == 9 or Force:                                     
                            loss = FxEstimator.fit_model(memory, valid_data = None, valid_split = 0.1, epochs=epochs, var_epochs = 10, batch_size = 1000, lr = lr, verbose= False)                    
                            logs['losslist' ].append(loss)                            
                        
                if model_det == 'Ensemble'  or model_det == 'Ancor':
                    if (-Not_optimal) % 10 == 9:
                        for i,_ in enumerate(FxEstimator.models_mu):
                            Nl = len(FxEstimator.models_mu[i][0].get_weights())
                            RNl = np.random.randint(Nl)
                            VVV = FxEstimator.models_mu[i][0].trainable_variables[RNl]
                            VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        if (parms['NN']['DEUP'] or parms['NN']['MLLV']):
                            Nl = len(FxEstimator.model_var[0].get_weights())
                            RNl = np.random.randint(Nl)
                            VVV = FxEstimator.model_var[0].trainable_variables[RNl]
                            VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        print(-Not_optimal + 1 )  
                        epochs = 1
                    if len(memory) >=2:
                        if itr % 10 == 9 or Force:                                                                   
                            loss = FxEstimator.fit_models(memory, valid_data = None, valid_split = 0.1, epochs=epochs, var_epochs = 10, batch_size = 1000, lr = lr, verbose= False)
                            logs['losslist' ].append(loss)
                    else:
                        FxEstimator.reset()
                
                if model_det == 'SWAG':
                    if (-Not_optimal) % 10 == 9:
                        Nl = len(FxEstimator.model_mu[0].get_weights())
                        RNl = np.random.randint(Nl)
                        VVV = FxEstimator.model_mu[0].trainable_variables[RNl]
                        VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        if (parms['NN']['DEUP'] or parms['NN']['MLLV']):
                            Nl = len(FxEstimator.model_var[0].get_weights())
                            RNl = np.random.randint(Nl)
                            VVV = FxEstimator.model_var[0].trainable_variables[RNl]
                            VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        print(-Not_optimal + 1 )  
                        epochs = 1

                    if len(memory) >=2:
                        if itr % 10 == 9 or Force:
                            acc, val = FxEstimator.fit_swag(memory, batch_size = 250, epochs_burn = 1, epochs_var = 10, lr = lr, delta = 0.01, verbose = False)
                            logs['acclist'].append(acc)
                            logs['val_loss' ].append(val)
                    else:
                        FxEstimator.reset()
                
                if model_det == 'LA':
                    if (-Not_optimal) % 10 == 9:
                        Nl = len(FxEstimator.model_mu[0].get_weights())
                        RNl = np.random.randint(Nl)
                        VVV = FxEstimator.model_mu[0].trainable_variables[RNl]
                        VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        if (parms['NN']['DEUP'] or parms['NN']['MLLV']):
                            Nl = len(FxEstimator.model_var[0].get_weights())
                            RNl = np.random.randint(Nl)
                            VVV = FxEstimator.model_var[0].trainable_variables[RNl]
                            VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        print(-Not_optimal + 1 )  
                        epochs = 1

                    if len(memory) >=2:
                        if itr % 10 == 9 or Force:
                            # acc, val = FxEstimator.fit_swag(memory, batch_size = 250, lr = 0.03, delta = 0.001, opt = 'Adam', const_var = 0.1, const = 0.0, verbose= False)
                            acc, val = FxEstimator.fit_LA(memory, epochs = 10, batch_size = 250, lr = lr, loss_coef = 1.0, const_var = 0.01, const = 0.0, verbose=False)
                            logs['acclist'].append  (acc)
                            logs['val_loss' ].append(val)
                    else:
                        FxEstimator.reset()
                    
                if model_det == 'MC-D':
                    if (-Not_optimal) % 10 == 9:
                        Nl = len(FxEstimator.model_mu[0].get_weights())
                        RNl = np.random.randint(Nl)
                        VVV = FxEstimator.model_mu[0].trainable_variables[RNl]
                        VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        if (parms['NN']['DEUP'] or parms['NN']['MLLV']):
                            Nl = len(FxEstimator.model_var[0].get_weights())
                            RNl = np.random.randint(Nl)
                            VVV = FxEstimator.model_var[0].trainable_variables[RNl]
                            VVV.assign_add(np.random.randn(*VVV.shape) * 0.001)
                        print(-Not_optimal + 1 )  
                        epochs = 1

                    if len(memory) >=2:
                        if itr % 10 == 9 or Force:
                            acc, val = FxEstimator.fit_model_mc(memory, epochs=epochs, batch_size = 250, lr = lr, delta = 0.01, opt = 'Adam', const_var = 0.1, const = 0.01, verbose= False)

                            logs['acclist' ].append(acc)
                            logs['val_loss' ].append(val)
            logs['excecute_time']['train'][-1] += time.time()
            logs['excecute_time']['inference'].append(-time.time())
            if FxEstimator and itr >0:
                _FxE , _FxV = FxEstimator.predictFx(xt.reshape(1,-1), s = 0) 
                _FxE  = _FxE[0].reshape((-1, FxEstimator.n_u))
                if FxEstimator is not None:
                    _FxVS = FxEstimator.S
                _JFxE = FxEstimator.predict_mean_Fx_gradient(xt.reshape(1,-1)) [0]
            else:            
                _FxE  = veh_model.get_Fx (xt)
                _JFxE = veh_model.get_JFx(xt)
            logs['excecute_time']['inference'][-1] += time.time()
            ktilde = np.zeros(2)
            clc_cons = True

            st_ccp = 0.001
            st_target = 0.99
            clc_cons = True
            if ccp>=st_target:
                ccp = st_target
            ccp += st_ccp
            if FxEstimator and parms['run']['model_det'] != 'Known' and online:            
                if ccp>np.random.rand(1):
                    ktilde = np.zeros(2)                                
                else:                
                    
                    if car_name == "Simple":
                        ktilde = (np.random.rand(2) - 0.5) * np.array([u1lim, u2lim]) * 0.25 #* 0.25                    
                    if car_name == "Turtle" or car_name == 'Prius':
                        ktilde = (np.random.rand(2) - 0.5) * np.array([u1lim, u2lim]) * 0.25 #* 0.25                    
                    if car_name == 'Husky':
                        ktilde = (np.random.rand(2) - 0.5) * np.array([u1lim, u2lim]) * 0.25 #* 0.25
                    if len(obstacles)>0:    
                        clc_cons = False
                
            else:
                ktilde = np.zeros(2)
                clc_cons = True

            ### Find Optimal Control
            logs['excecute_time']['control'].append( -time.time())
            if run != 'Rand':     
                u, d1, d2, stat_cbc = control.estimate_u_EHRD( _FxE,_JFxE, xt, FxEstimator=FxEstimator, obstacles= obstacles,
                                                        verbose=False, ktilde=ktilde, clc_cons=clc_cons) 
                stat =  control.status
            else:
                if itr % 50 == 0:
                    u     = np.ones(3, dtype = np.float32)
                    u[1:] = (np.random.rand(2) - 0.5) * np.array([u1lim, u2lim]) * 2

                d1, d2, stat_cbc = 0,0,[np.zeros(3), np.zeros([3,3])]
                stat = 'optimal'
            
            logs['excecute_time']['control'][-1] += time.time()
                        
            epochs = 1
            exit, xd     = Traj.get_xd(xt)
            if exit:
                Traj.reset()
                exit = False
            if stat != 'optimal':
                ccp = st_target - st_ccp
                Force = True
                Not_optimal -= 1
        logs['Not_optimal'].append(Not_optimal)
        if Not_optimal < 0:
            nopt = np.array(logs['Not_optimal'])
            print(f'Not optimal {Not_optimal} max {np.min(nopt)} Avg {np.mean(nopt)}')
            
        #### 'Calculate optimal'Calibrations::
        if parms['run']['model_det'] != 'Known': 
            if len(memory) > 3:
                _xin, _uin, _yin = memory.get_samples(len(memory))
                _my, _sy = FxEstimator.predictXdot(_xin, _uin)
                if (not (parms['NN']['DEUP'] or parms['NN']['MLLV']) and (model_det == 'Baseline' or model_det == 'Res' )):                    
                    pass
                else:
                    _bin,_conf,_area, MLP, RL2E, RMSCE = Uncertainty_metrics(_yin, _my, _sy)
                    print(f'loss {acc:.3f},val {val:.3f}, area {_area:.3f}, MLP {MLP:.3f}, RL2E {RL2E:.3f}, RMSCE {RMSCE:.3f}')
                    logs['metric'].append([_conf,_area, MLP, RL2E, RMSCE])
        
        ### Apply Control to robot
        FWr, FWl = 0, 0        
        if car_name == "Simple": 
            sim.update(u, dt = dt)   
            
        else:
        
            if car_model == 'Ack1':
                v, w = u[1] , u[2] #* car_WR
                w = w * 2.0
            
            ### FWr:: rotation velocity of right wheel
            ### FWl:: rotation velocity of left  wheel                  
            FWr = ((2*v) + (w*car_L))/(2*car_WR)
            FWl = ((2*v) - (w*car_L))/(2*car_WR)   
            
            vel_control = [FWr, FWl]                
            
            update_simulator(car_name, vel_control, num_steps, time_step)
                    
        if car_name != "Simple": 
            if sim_images:

                img_infor = p.getCameraImage(sim_img_w, sim_img_h, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                imgrgb = img_infor[2][:, :, :3]
                
                matplotlib.image.imsave(fpath + 'sim_imgs/pic_' +str(itr)+ '.png', imgrgb)
        
        logs['real-time']    .append(datetime.datetime.now())
        logs['sim-time']     .append(sim_time)
        logs['step']         .append(itr)
        logs['indx'   ]      .append(itr)
        
        logs['CnvxSt']       .append(control.status)
        logs['ktilde']       .append(ktilde)
        
        logs['h_x'   ]       .append(np.array([CF.hx  (xt, obs)                     for obs in obstacles]))
        logs['hp_x'  ]       .append(np.array([CF.hxp (xt, obs, _FxE)               for obs in obstacles]))
        
        logs['V1_x'      ]   .append(np.array([ CF.V1x(xt, xd)]))
        
        logs['FElist'    ]   .append(_FxE.flatten())
        logs['FVarlist'    ]   .append(_FxV.flatten())
        logs['FVSlist'    ]   .append(_FxVS)
        
        logs['FSlist'    ]   .append(veh_model.get_Fx (xt).flatten())
        sim_time += dt
        def take_diff_var_Fx(Fx, EFx, len_var = 100):
            Fx  = np.array(Fx)
            EFx = np.array(EFx)
            _DiffFx = Fx[-1] - EFx[-1]
            len_var = len_var if len(Fx) >= len_var else len(Fx)
            _DiffFx = Fx[-len_var:]-EFx[-len_var:]
            _VarFx = 0
            return _DiffFx[-1],np.var(_DiffFx,axis=0)
        _diffFx, _varFx = take_diff_var_Fx(logs['FSlist'    ],logs['FElist'    ], len_var=100)
        logs['FDlist'    ]   .append(_diffFx)
        logs['FVlist'    ]   .append(_varFx)
        logs['JElist'    ]   .append(_JFxE.flatten())
        logs['JSlist'    ]   .append(veh_model.get_JFx(xt).flatten())
        
        logs['Stat_CBC'  ]   .append(stat_cbc)
        logs['CBC1_x'    ]   .append(np.array([CF.CBC1(xt, obs, _FxE)               for obs in obstacles]))
        logs['CBC2_x'    ]   .append(np.array([CF.CBC2(xt, obs, _FxE, _JFxE) @ u    for obs in obstacles]))
        
        logs['hpp_x'     ]   .append(np.array([CF.hxpp(xt, obs, _FxE, _JFxE) @ u    for obs in obstacles]))
        
        logs['CLC1_x'    ]   .append([CF.CLC1_(_FxE, _JFxE, xt, u, xd)]) 
        logs['delta1'    ]   .append(d1)
        logs['delta2'    ]   .append(d2)        
        logs['FWl-FWr'   ]   .append([FWl,FWr])
        if parms['NN']['DEUP'] or parms['NN']['MLLV']:
            if model_det == 'Baseline' or model_det == 'Res' :
                logs['weights'] = [FxEstimator.model_mu[0].get_weights(), FxEstimator.model_var[0].get_weights(), ]
            if model_det == 'Ensemble':
                logs['weights'] = [[model_mu[0].get_weights() for model_mu in FxEstimator.models_mu], FxEstimator.model_var[0].get_weights(), ]
            if model_det == 'SWAG':
                logs['weights'] = [FxEstimator.model_mu[0].get_weights(), FxEstimator.model_var[0].get_weights(), FxEstimator.w_swa, FxEstimator.s_diag_sq, FxEstimator.D_hat,]
            if model_det == 'MC-D':
                logs['weights'] = [FxEstimator.model_mu[0].get_weights(), FxEstimator.model_var[0].get_weights(),]
            if model_det == 'LA':
                logs['weights'] = [FxEstimator.model_mu[0].get_weights(), FxEstimator.model_var[0].get_weights(), FxEstimator.H.numpy(), ]
            if model_det == 'Ancor':
                logs['theta'] = [[model_mu[0].theta_numpy for model_mu in FxEstimator.models_mu], FxEstimator.model_var[0].theta_numpy, ]
                logs['weights'] = [[model_mu[0].get_weights() for model_mu in FxEstimator.models_mu], FxEstimator.model_var[0].get_weights(), ]
            
        else:
            if model_det == 'Baseline' or model_det == 'Res':
                logs['weights'] = [FxEstimator.model_mu[0].get_weights(),]
            if model_det == 'Ensemble':
                logs['weights'] = [[model_mu[0].get_weights() for model_mu in FxEstimator.models_mu], ]
            if model_det == 'SWAG':
                logs['weights'] = [FxEstimator.model_mu[0].get_weights(), FxEstimator.w_swa, FxEstimator.s_diag_sq, FxEstimator.D_hat,]
            if model_det == 'MC-D':
                logs['weights'] = [FxEstimator.model_mu[0].get_weights(),]
            if model_det == 'LA':
                logs['weights'] = [FxEstimator.model_mu[0].get_weights(), FxEstimator.H.numpy(), ]
            if model_det == 'Ancor':
                logs['weights'] = [[model_mu[0].get_weights() for model_mu in FxEstimator.models_mu], ]
                logs['theta']   = [[model_mu[0].theta_numpy for model_mu in FxEstimator.models_mu], ]
        if verbose:            
            if len(obstacles)>0:
                print(f"opt{control.prob.status},{control.var.value}| Ep{control._Ep[:,:,0]}| Eq{control._Eq[:,0]}|")
                print(f"Fx{control._Fx},Vx{control._V}| GV{control._GV}| hx{control._hx}| Ghx{control._Ghx}")
            else:
                print(f"opt{control.prob.status},{control.var.value}| Eq{control._Eq[:,0]}|")
                print(f"Fx{control._Fx},Vx{control._V}| GV{control._GV}")

        def det_collide(cir, ellipse, n = 10, plot = False):
            # points on circle    
            ts = 2 * np.pi * np.linspace(0,1,n)
            # map circle pointson cir
            ps = np.array([np.sin(ts),np.cos(ts)]).T * cir[2] + cir[np.newaxis, :2]
            #check points in ellipse cordinate
            if plot:
                fig, ax = plt.subplots()
                ax.add_patch(Ellipse(ellipse[:2],ellipse[2]*2,ellipse[3]*2,ellipse[4], color = 'blue', alpha = 0.3))
                ax.add_patch(Circle (cir[:2],cir[2], color = 'red', alpha = 0.3))
                # def collide_circle_ellipse():
                ax.set_xlim([-4,4])
                ax.set_ylim([-4,4])
                ax.axis('equal')
                plt.scatter(ps[:,0],ps[:,1])
            Ps = []
            for p in ps:
                dx,dy = p - ellipse[:2]
                a,b = ellipse[2:4]
                t = ellipse[4]
                p   = np.array([dx,dy])  
                rad = np.diag([1/a,1/b])
                rot = np.array([[np.cos(-t),-np.sin(-t)], 
                                [np.sin(-t),np.cos(-t)]])
                f = p @ rot @  rad              
                Ps.append(f @ f.T - 1.0)
            Ps = np.array(Ps)
            return np.any(Ps < 0.0)
        for obs in obstacles:
            if len(obs) == 3: #circle
                if np.linalg.norm(xt[:2]-obs[:2]) < (obs[2] + car_RR ):
                    print(f"x:{xt}, hx:{logs['h_x'][-1]} hpx:{logs['hp_x'][-1]} cbc1:{logs['CBC1_x'][-1]} cbc2:{logs['CBC2_x'][-1]} ")
                    print(f"hit{obs}")
                    logs['hited'] = True
                    exit = True
            elif len(obs) == 5: #ellipsoid
                if det_collide(np.array([xt[0],xt[1],car_RR]), obs, plot = False):
                    print(f"x:{xt}, hx:{logs['h_x'][-1]} hpx:{logs['hp_x'][-1]} cbc1:{logs['CBC1_x'][-1]} cbc2:{logs['CBC2_x'][-1]} ")
                    print(f"hit{obs}")
                    logs['hited'] = True
                    exit = True
                #TODO
                pass
            
        time_elapsed = time.time() - run_start_time
        if time_elapsed > max_time:
            exit = True
            print('!!!  max time is reached!!!')
        
        if itr +1  >= max_iter:
            exit = True
            print('!!!  max itr  is reached!!!')
        
        if itr % 20 == 0 or exit:
            if exit:
                if parms['run']['car_name'] == 'Simple':     
                    _xdot = (sim.x - xt)
                    _xdot[2] = np.arctan2(np.sin(_xdot[2]), np.cos(_xdot[2]))
                    _xdot /= dt                    
                    xt = sim.x.copy()
                else:            
                    xt = x_new.copy()
                    position, orientation = p.getBasePositionAndOrientation(car)
                    orien = p.getEulerFromQuaternion(orientation)
                    x_,y_ = position[:2]
                    theta_ = orien[2]
                    _vx   = (x_- xt[0]) / dt
                    _vy   = (y_- xt[1]) / dt
                    if smpl_itr:
                        _vx   = (x_- xt[0]) / dt / num_steps
                        _vy   = (y_- xt[1]) / dt / num_steps
                    vel_2 = np.sqrt(_vx **2 + _vy **2 )
                    
                    th = xt[2]
                    th = theta_
                    vel_2 = _vx * np.cos(th) + _vy * np.sin(th)
                    
                    vel_trans, vel_ang = p.getBaseVelocity(car)
                    _vx, _vy, _ = vel_trans
                    
                    vel_ = np.linalg.norm(vel_trans[:2])
                    logs['vel_car'].append([vel_,vel_2])
                    if car_model == 'Ack2':
                        x_new = np.array([x_,y_,theta_, vel_2])
                    elif car_model == 'Ack1':
                        x_new = np.array([x_,y_,theta_])
                    else:
                        w_ = (theta_ - xt[2]) 
                        w_ = np.arctan2(np.sin(w_), np.cos(w_))/ dt #/ num_steps
                        if smpl_itr:
                            w_ = np.arctan2(np.sin(w_), np.cos(w_))/ dt / num_steps
                        x_new = np.array([x_,y_,theta_, w_, vel_2])
                    
                    _xdot = (x_new - xt)
                    _xdot[2] = np.arctan2(np.sin(_xdot[2]), np.cos(_xdot[2]))
                    _xdot = _xdot / dt   #/ num_steps             
                    if smpl_itr:
                        _xdot = _xdot / dt / num_steps
                if FxEstimator:
                    if control.status != 'infeasible':
                        memory.push(xt.copy().astype(dtype = np.float32), u.copy().astype(dtype = np.float32), _xdot.copy().astype(dtype = np.float32)) 

                logs['x'   ].append(xt   .copy())
                logs['u'   ].append(u    .copy())
                logs['xdot'].append(_xdot.copy()) 
            
            if logFile:
                json.dump(logs, open(fpath + "logs1.json",'w'))             
                os.remove(fpath + "logs.json")
                os.rename(fpath + "logs1.json",fpath + "logs.json")                        
                    
                if plots:
                    fig .savefig(fpath + "fig.pdf")
                    fig2.savefig(fpath + "fig2.pdf")
                
            if plots:
                _indx = np.array(logs['indx'])
                _x    = np.array(logs['x'])
                _u    = np.array(logs['u'])
                
                ax21.cla(); ax22.cla(); ax23.cla(); ax24.cla(); ax25.cla(); ax26.cla()
                
                if   car_model == 'DD':                    
                    ax21.plot(_x[:,4], label = 'vel')
                elif car_model == 'Ack2':                                        
                    ax21.plot(_x[:,3], label = 'vel')
                    ax21.plot(_u[:,1], label = 'u[1]')
                    ax21.plot(_u[:,2], label = 'u[2]')
                elif car_model == 'Ack1':                                        
                    ax21.plot(_u[:,1], label = 'vel')
                    ax21.plot(_u[:,2], label = 'w')
                
                ax21.plot(_x[:,2], label = 'theta')            
                
                if FxEstimator:
                    ax21.set_title(f'{itr_optimal/(itr+1)*100:.1f}')
                
                ax23.plot(_indx, np.array(logs['V1_x'])  , label = 'V1x')                        
                ax23.plot(_indx, np.array(logs['CLC1_x']), label = 'CLC1'  , color = 'cyan')
                ax23.plot(_indx, np.array(logs['delta1']), label = 'delta1', color = 'b')
                
                if len(obstacles)>0:                    
                    for i, h_xi in enumerate(zip(* np.array(logs['h_x']) )):
                        ax24.plot(_indx, h_xi, label = 'h%d' % i)
                    for i, h_xi in enumerate(zip(*np.array(logs['hp_x']))):
                        ax25.plot(_indx, h_xi, label = 'hp%d' % i)
                    for i, h_xi in enumerate(zip(*np.array(logs['hpp_x']))):
                        ax25.plot(_indx, h_xi, label = 'hpp%d' % i)
                                        
                    for i, CBC1_xi in enumerate(zip(*np.array(logs['CBC1_x']))):
                        ax26.plot(_indx, CBC1_xi, label = 'CBC1_%d' % i)
                    for i, CBC2_xi in enumerate(zip(*np.array(logs['CBC2_x']))):
                        ax26.plot(_indx, CBC2_xi, label = 'CBC2_%d' % i)
                    
                    ax24.legend() 
                    ax25.legend() 
                    ax26.legend() 
                    pass
                ax21.legend();  
                ax23.legend()
                
                #PLOT RAY                
                ax.cla()
                for i in range(Traj.path.shape[0]):
                    ax.scatter(Traj.path[i,0],Traj.path[i,1], color = 'g')                
                    ax.annotate(f"{i+1}", (Traj.path[i,0],Traj.path[i,1]))

                for obs in inp_obs:        
                    ax.add_patch(plt.Circle(obs[:2], obs[2], color='orange', alpha = 0.2))

                for obs in obstacles:
                    if len(obs) == 3:                    
                        ax.add_patch(plt.Circle(obs[:2], obs[2], color='red', alpha = 0.5))
                    else:
                        ax.add_patch(Ellipse(obs[:2], width=obs[2] * 2, height=obs[3] *2, angle= obs[4],color='red', alpha = 0.5))

                if car_name != 'Simple':
                    for pnts in obs_pnts:
                        ax.scatter(pnts[:,0],pnts[:,1], alpha=0.5)    
                if   car_model == 'DD':
                    [x, y, theta, w, vel] = xt
                    ax.set_title(f"th{np.rad2deg(xt[2]):.2f},w{w:.1f},u[{u[1]:.2f},{u[2]:.2f}],dist{Traj.get_dist():.2f},v{vel:.2f}")
                elif car_model == 'Ack2':
                    [x, y, theta, vel] = xt
                    ax.set_title(f"th{np.rad2deg(xt[2]):.2f},u[{u[1]:.2f},{u[2]:.2f}],dist{Traj.get_dist():.2f},v{vel:.2f}")
                elif car_model == 'Ack1':
                    [x, y, theta] = xt
                    vel = u[1]
                    ax.set_title(f"th{np.rad2deg(xt[2]):.2f},u[{u[1]:.2f},{u[2]:.2f}],dist{Traj.get_dist():.2f},v{vel:.2f}")
                
                if len(out_circ) == 3:
                    drawObject = Circle(out_circ[:2], radius=out_circ[2], fill=False, color="red", lw = 5.0)
                    ax.add_patch(drawObject)
                if len(out_circ) == 5:
                    drawObject = Ellipse(out_circ[:2], width=out_circ[2] * 2, height=out_circ[3] * 2, angle = out_circ[4], fill=False, color="red", lw = 5.0)
                    ax.add_patch(drawObject)

                ax.set_xlim(-5,5)
                ax.set_ylim(-5,5)

                ax.plot(_x[:,0],_x[:,1],'-',color='blue')
                ax.add_patch(plt.Circle(_x[-1][:2], car_RR, color='purple', alpha = 0.2))
                if len(obstacles) > 0:
                    uncer = np.linalg.norm( stat_cbc[1])
                    ax.add_patch(plt.Circle(_x[-1][:2], car_RR * uncer, color='green', alpha = 0.1))

                ax.arrow(x,y,car_RR * np.cos(theta),car_RR * np.sin(theta),color='red',width=0.01,head_width=0.02)
                ax.arrow(x,y,_xdot[0],_xdot[1],color='lightgreen',width=0.05,head_width=0.1)
                
                ax.set_aspect('equal', adjustable='box')
                plt.draw()
                
                plt.pause(0.01)
            if plots and save_plot:
                fig.savefig(fpath + 'figs/' + f'pic_{itr}.png')    
            
            if online and FxEstimator and save_model:
                for i,m in enumerate( FxEstimator.train_models):
                    m.save(fpath + f'models/train_model_{i:02d}' )
        if Traj.get_end():
            print("Good job!")
            Traj.reset()
            
        if not (control.status == 'optimal'):
            print(f"opt:{control.status}" ,f"U:{u}", f"nUZero{nZeroVel}", f"itr {itr}")
            nZeroVel += 1
            
        else:
            nZeroVel = 0
        
        if exit:            
            print("exit")
            break
        
    if plots:  
        fig.savefig("fig.png")
        fig2.savefig("fig2.png")
    print("Done!")
    if car_name != 'Simple':
        p.disconnect()    

def init_params(seed = 1):
    parms = {}    
    run       = 'Prob'     # Rand, Prob, Stand
    model_det = 'Ancor'    # Known, Baseline, Ensemble, SWAG, LA, MC-D, Ancor, BNN
    car_name  = 'Husky'    # Simple, Turtle, Husky, Prius
    car_model = 'Ack1'     # Ack1, TODO
    SOCP      = True
    Online    = True
    DEUP      = True
    reg       = 0.0001 #0.0001

    if run not in ['Prob', 'Rand']:
        raise Exception(f'run {run} is not defined')
    
    if model_det not in ['Known', 'Baseline', 'Ensemble', 'SWAG', 'LA', 'MC-D', 'Ancor', 'Res']: #, SWAG']:
        raise Exception(f'model_det {model_det} is not defined')
    
    if car_name not in ['Simple', 'Turtle', 'Husky', 'Prius']:
        raise Exception(f'run {car_name} is not defined')
    
    if car_model not in ['DD', 'Ack2', 'Ack1']:    #Ack1  , Ack2
        raise Exception(f'run {car_model} is not defined')
 
    max_time_hour = 0.75
    
    parms['run'] = {'run'      : run,         #Prob, Stand
                    'model_det': model_det,   #Known, Baseline, Ensemble, BNN, SWAG
                    'car_name' : car_name,    #Simple, Turtle, Husky, Prius
                    'car_model': car_model,   #Ack1  , Ack2  , DD
                    'max_time_hour' : max_time_hour
                    }
    
    out_circ = [0.0, 0.0, 3.5 , 4.5 , 0.0]

    t    = np.linspace(0,1.0, 4) * 2
    ddd  = 2.4
    path = np.vstack(( np.sin(t * np.pi) * ddd , -np.cos(t * np.pi) * ddd )).T 
    
    obstacles = [ [0.0, 0.0,1.0, 1.5, 0.0, "cylindr"],]
    
    if parms['run']['car_model'] == 'DD':
        n_x = 5 # x, y, th, w, v    # 1, p, a or 1, fr, fl?
        x0 = np.zeros(n_x); x0[:2] = path[0]; x0[2] = np.pi / 2 * 0 #[-0.25,-0.25]
    
    if parms['run']['car_model'] == 'Ack2':
        n_x = 4 # x, y, th, v       # 1, w, a
        x0 = np.zeros(n_x); x0[:2] = path[0]; x0[2] = np.pi / 2 * 0 #[-0.25,-0.25]
    
    if parms['run']['car_model'] == 'Ack1':
        n_x = 3 # x, y, th           # 1, w, v
        x0 = np.zeros(n_x); x0[:2] = path[0]; x0[2] = np.pi / 2 * 0 #[-0.25,-0.25]
        
    n_u = 3
    path[0] = x0[:2]

    dist_er = 0.1
    parms['env'] = {'path'     : path,
                    'obstacles': obstacles,
                    'x0'       : x0,
                    'dist_er'  : dist_er,
                    'use_real_obstacle' : False,  ## Robots now real circles around objects, don't have to detect them
                    } 
    parms['memcapacity'] = 10000    
    parms['seed']        = seed
    np.random.seed    (parms['seed'])
    tf.random.set_seed(parms['seed'])

    if car_name == "Simple":
        Robot_R = 0.18
        Wheel_R = 0.034
        Lenght  = 0.22   
        Mass    = 1
    if car_name == "Turtle":
        Robot_R = 0.18
        Wheel_R = 0.034
        Lenght  = 0.22
        Mass    = 2.4
    if car_name == 'Husky':
        Robot_R = 0.5
        Lenght  = 0.54
        Wheel_R = 0.17
        Mass    = 5
        
    if car_name == 'Prius':
        Robot_R = 0.5
        Lenght  = 0.54
        Wheel_R = 0.17
        Mass    = 5
    parms['env']['dist_er'] = Robot_R
    parms['car'] = {'name'      : car_name,
                    'car_model' : car_model,
                    'Robot_R'   : Robot_R,
                    'Lenght'    : Lenght,
                    'Wheel_R'   : Wheel_R,
                    'x0'        : x0,
                    'Mass'      : Mass}
    parms['env']['dist_er'] = Robot_R + 0.1
    if car_name == 'Turtle':
        parms['env']['dist_er'] = Robot_R + 0.1
    if car_name == 'Husky':
        parms['env']['dist_er'] = Robot_R + 0.2
    if car_name == 'Prius':
        parms['env']['dist_er'] = Robot_R + 0.5
        
    n = 1
    parms['sim'] = {'type'      : 'no_gui', #no_gui, gui, mp4                    
                    'time_step' : 100. * n,
                    'dt'        : 0.1 / n ,
                    'n_Rays'    : 200,  # LIDAR scan ray numbers
                    't_0'       : -90/180*np.pi, #start angle
                    't_n'       : 360/180*np.pi, #range angle
                    'rayLen'    : 3    ,
                    'save_images':False,      
                    'img_w'     : 640,
                    'img_h'     : 480,  
                    'smpl_itr'  : False                 
                    }
    if parms['sim']['smpl_itr']:
        parms['memcapacity'] = 10000 * 5
    parms['sim']['num_steps'] = int( parms['sim']['time_step'] * parms['sim']['dt'])
    parms['sim']['time_step'] = parms['sim']['num_steps'] / parms['sim']['dt']
    
    parms['NN']         = {'type'      : model_det, #Known, Baseline, Ensemble, BNN, SWAG
                           'lr'        : 0.001,
                           'batch_size': 1000,
                           'online'    : Online,
                           'n_ens'     : 5,  #Number of Ensembles
                           'n_sample'  : 5,  #Number of samples, Bayesian                           
                           'n_in'      : n_x - 2,
                           'n_out'     : n_x * n_u,
                           'n_x'       : n_x,
                           'n_u'       : n_u,
                           'n_l'       : 6,
                           'n_node'    : 30,
                           'reg'       : reg,
                           'activ'     : 'tanh',
                           'seed'      : parms['seed'],
                           'car'       : parms['car'],
                           'preTrain'  : False,
                           'swag'      : {'K' : 10, 'T' : 20, 'S' : 5},
                           'LA'        : {'temp': 1.0, 'prior': 1.0, 'sigma': 1.0, 'S': 5},
                           'MC-D'      : {'drop': 0.1, 'inc_hid' : True, 'n_ens': 5},
                           'Ancor'     : {'prior' : [0.0, 1.00], 'n_ens' : 5, 'reg' : 0.010},
                           'Res'       : {'Width' : 16, 'BatchNorm': True, 'ResLayers': ['I','D','D']},
                           'DEUP'      : DEUP,
                           'MLLV'      : False,
                           'models'    : None,
                           'init_path' : None, #path to log of simulation
                           }
    
    if parms['NN']['DEUP'] and parms['NN']['MLLV']:
        raise 'DEUP or MLLV: Both cannot be true'
    if model_det == 'Baseline' or model_det == 'Res':
        parms['NN']['n_ens']    = 1
        parms['NN']['n_sample'] = 1
        
    parms['dt'] = parms['sim']['dt']
    parms['max_iter'] = 5000
    
    #### objective, lambda0 , lambda1
    lamba = [100.0, 0.0]     #position , velocity # objective = ... + lambda0 * d0 ^2 + lambda1 * d1 ^2
    
    ### clc1: 2nd degree
    kbV1 = [1.0,2.0]     #V1'' + kb[0] * V1' + kb[1] * V1 <= d1
    ### clc2: 1st degree
    kbV2 = [0.0, 0.0]        #             V2' + k[0] * V2 <= d2  #V2=(V-k[1] *dx)^2
    ### cbf: 2nd degree
    kbH  = [1.0,2.0]     #E(H''   + kb[0] * H'  + kb[1] * H) *u >= cp * sqrt(u' * Var * u)
    
    vmax = 0.1
    kvel = 1.0
    wmax   = 0.1
    kw     = 2.0
    if car_name == "Simple":
        vmax = 1.0
        kvel = 2.0
        wmax   = 1.0
        kw     = 2.0
        u_lim = [0.5,0.5]
    if car_name == "Turtle":
        # u_lim = [wmax * Mass / 5,vmax * Mass / 5]
        vmax = 0.1
        kvel = 2.0
        u_lim = [0.5,0.5]
    if car_name == "Prius":
        # u_lim = [wmax * Mass / 5,vmax * Mass / 5]
        vmax = 0.1
        kvel = 2.0
        u_lim = [0.5,0.5]
    if car_name == 'Husky':
        u_lim = [1.0,1.0]
    u_lim = np.array(u_lim) 
    ktilde = np.array([0,0])
    Lx     = np.array([1.0,1.0])     
    p = _cp ** 2 / (_cp**2 +1)

    parms['controler'] = {'cp'    :_cp,
                          'p'     : p,
                          'lambda': lamba,
                          'kbV1'  : kbV1,
                          'kbV2'  : kbV2,
                          'kbH'   : kbH,
                          'vmax'  : vmax,
                          'kvel'  : kvel,
                          'wmax'  : wmax,
                          'kw'    : kw,
                          'u_lim' : u_lim,
                          'ktilde': ktilde,
                          'Lx'    : Lx,
                          'car'   : parms['car'],
                          'SOCP'  : SOCP,
                          'out_circ': out_circ}    
    return parms

def main(seed = 1):
    parms = init_params(seed = seed)
    np.set_printoptions(precision=3, suppress=True)
    print(f"Simulation is started: p:{parms['controler']['p']} , cp:{parms['controler']['cp']}")
    
    L  = parms['car']['Lenght']
    R  = parms['car']['Wheel_R']
    x0 = parms['car']['x0']
    M  = parms['car']['Mass']
    if parms['run']['car_name'] == 'Simple': 
        if parms['run']['car_model'] == 'DD':
            model = DD_Model(L = L, R = R, M = M)    
            veh = vehicle_class(xt=x0, model=model)
        if parms['run']['car_model'] == 'Ack2':
            model = UnicyleModel(L = L, R = R)    
            veh = vehicle_class(xt=x0, model=model)
        
        if parms['run']['car_model'] == 'Ack1':
            model = UnicyleModel_Ack1(L = L, R = R)    
            veh = vehicle_class(xt=x0, model=model)
        
    sim = Simulator(x=x0, u = np.array([ 1.        , 0.0, 0.0]), 
                    model_type= parms['car']['car_model'] , L = L, R = R, M = M)
    
    Traj = trajectory(parms['env'])
    CF   = control_Functions(parms['controler'])
    control = controller_DetFx(parms['controler'], controlFunction = CF, traj= Traj)    
    if parms['run']['model_det'] == 'Known': 
        FxEstimator = None # use Fx,JFx from simulator # Deterministic and no e-greedy
        memory      = None
        print('No estimator')
    else:
        if parms['NN']['init_path'] is not None:
            FxEstimator = init_model(parms['NN']['init_path'])   
        else:
            FxEstimator = estimate_DDFx(parms['NN'])   
        memory = ReplayMemory(parms['memcapacity'])

    run_robot(sim, parms, Traj, control, CF = CF, FxEstimator = FxEstimator, 
                                memory = memory, logFile = True, plots = True, verbose= False,
                                save_model = False, save_plot = False
                                # NN = keras.models.load_model("model/model_500tanh_sim.nn")
                                ) #Ofline model

def load_weights(model, weights, theta = None):
    if model.type == 'Baseline' or model.type == 'MC-D':
        model.model_mu[0].set_weights(weights[0])
        if model.DEUP:
            model.model_var[0].set_weights(weights[1])           
    elif model.type == 'Ensemble':
        for m,w in zip(model.models_mu,weights[0]):
            m[0].set_weights(w)
        if model.DEUP:
            model.model_var[0].set_weights(weights[1])  
    elif model.type == 'SWAG':
        model.model_mu[0].set_weights(weights[0])
        if model.DEUP:
            model.model_var[0].set_weights(weights[1])           
        model.w_swa    = weights[-3]
        model.s_diag_sq = weights[-2]
        model.D_hat     = weights[-1]      
    elif model.type == 'LA':
        model.model_mu[0].set_weights(weights[0])
        if model.DEUP:
            model.model_var[0].set_weights(weights[1])           
        model.H    = weights[-1]        
    elif model.type == 'Ancor':
        for m,w,th in zip(model.models_mu,weights[0], theta[0]):
            m[0].set_weights(w)
            m[0].theta = [tf.convert_to_tensor(t) for t in th]
            m[0].theta_numpy = th
        if model.DEUP:
            model.model_var[0].set_weights(weights[1])               
            model.model_var[0].theta = [tf.convert_to_tensor(t) for t in theta[1]]
            model.model_var[0].theta_numpy = theta[1]
        

    else:
        raise 'not implemented'
    return model
def init_model(path):
    init = json.load(path + 'init_vars.json')
    log  = json.load(path + 'logs.json')
    weights = log['weights']
    theta = None
    if 'theta' in log.keys():
        theta = log['theta']
    model = estimate_DDFx(init['NN'])    
    model = load_weights(model, weights, theta)
    return model

if __name__ == '__main__':     
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    np.random.seed(1)
    rand_seeds = np.random.randint(0, 10000, 20)
    _cp = 1.73
    now   = datetime.datetime.now()    
    main(seed = 1)  