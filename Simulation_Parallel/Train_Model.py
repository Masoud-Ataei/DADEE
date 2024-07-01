# parms, weights,theta
from DDEstimator import estimate_DDFx
from filelock import FileLock
import json_tricks as json
from utils import set_tf_GP_grow, set_tf_deterministic,set_seed
import numpy as np
import time
    
def get_weights(model, parms):
    var_weights = []
    var_theta   = []
    theta = []
    model_det = parms['NN']['type']
    if parms['NN']['DEUP'] or parms['NN']['MLLV']:
        var_weights = [model.model_var[0].get_weights()]
        if model_det == 'Ancor':    
            var_theta = [model.model_var[0].theta_numpy]
    if model_det == 'Baseline' or model_det == 'Res' :
        weights = [model.model_mu[0].get_weights()] + var_weights
    if model_det == 'Ensemble':
        weights = [[model_mu[0].get_weights() for model_mu in model.models_mu]] + var_weights
    if model_det == 'SWAG':
        weights = [model.model_mu[0].get_weights()] + var_weights + [model.w_swa, model.s_diag_sq, model.D_hat]
    if model_det == 'MC-D':
        weights = [model.model_mu[0].get_weights()] + var_weights
    if model_det == 'LA':
        weights = [model.model_mu[0].get_weights()] + var_weights + [model.H.numpy()]
    if model_det == 'Ancor':
        theta = [[model_mu[0].theta_numpy for model_mu in model.models_mu]] + var_theta
        weights = [[model_mu[0].get_weights() for model_mu in model.models_mu], model.model_var[0].get_weights(), ]
    return weights, theta

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
            m[0].theta = th
        if model.DEUP:
            model.model_var[0].set_weights(weights[1])               
                        
    else:
        raise 'not implemented'
    return model

def init_model(path):
    init = json.load(path + 'init_vars.json')
    log  = json.load(path + 'logs.json')
    weights = log['weights']
    model = estimate_DDFx(init['NN'])    
    model = load_weights(model, weights)
    return model



if __name__ == '__main__':     

    mempath = "/dev/shm/"

    set_tf_GP_grow()
    set_tf_deterministic()
    set_seed(1)
    lock_data  = FileLock(mempath + "data.json.lock")    
    lock_model = FileLock(mempath + "model.json.lock")    
    with lock_model:
        model_data = json.load(mempath + "model.json")   
        parms, weights,theta = model_data['parms'], model_data['weights'], model_data['theta']


    model = estimate_DDFx(parms['NN'])

    while True:
        with lock_data:
            data = json.load(mempath + "data.json")   
            x,u,xdot = np.array(data['x']), np.array(data['u']), np.array(data['xdot'])
        loss = None
        epochs = 10
        var_epochs = 10
        lr = 0.001
        batch_size = 100
        valid_split = 0.1
        verbose = True
        if model.type == 'Baseline' or model.type == 'Res':
            loss = model.fit_model(([x,u],xdot), None, valid_split=valid_split, epochs = epochs, batch_size=batch_size, lr = lr, var_epochs = var_epochs, preloss = loss, verbose = verbose)
        elif model.type == 'Ensemble'  or model.type == 'Ancor':
            loss = model.fit_models(([x,u],xdot), None, valid_split=valid_split, epochs=epochs, batch_size = batch_size, lr = lr, var_epochs = var_epochs, preloss = loss, verbose = verbose)
        elif model.type == 'SWAG':
            loss = model.fit_swag(([x,u],xdot), None, valid_split=valid_split, epochs=epochs, batch_size = batch_size, epochs_burn = 1, var_epochs = var_epochs, lr = lr, verbose = verbose)
        elif model.type == 'LA':
            loss = model.fit_LA  (([x,u],xdot), None, valid_split=valid_split, epochs=epochs, batch_size = batch_size, lr = lr, var_epochs = var_epochs, preloss = loss, verbose = verbose)
        elif model.type == 'MC-D':
            loss = model.fit_model_mc(([x,u],xdot), None, valid_split=valid_split, epochs=epochs, batch_size = batch_size, lr = lr, var_epochs = var_epochs, preloss = loss, verbose = verbose)
        else:
            raise f'{model.type} is not implemented.'
            
        weights,theta = get_weights(model, parms)
        with lock_model:
            json.dump({'parms' :parms, 
                        'weights':weights,
                        'theta':theta,
                        'loss': loss}, mempath + "model.json")   
        time.sleep(1)