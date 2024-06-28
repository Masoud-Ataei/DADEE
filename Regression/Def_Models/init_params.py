import tensorflow as tf
import numpy as np

def init_params(seed = 1, model_det = 'SWAG'):
    # Known, Baseline, Ensemble, BNN, SWAG
    parms = {}    
     
    parms['seed']        = seed
    np.random.seed    (parms['seed'])
    tf.random.set_seed(parms['seed'])
    
    memcapacity = 10000
    batch_size = 1000
    lr = 0.3
    n_in, n_out, n_hidden, activ = 1, 1, (4,12), "tanh"  
    parms['estimator'] = {'type'      : model_det,
                          'memcap'    : memcapacity,
                          'n_in'      : n_in,
                          'n_out'     : n_out,                          
                          'n_hidden'  : n_hidden,
                          'lr'        : lr,
                          'activ'     : activ,
                          'batch_size': batch_size,
                          'swag'      : {'K' : 10, 'T' : 20, 'S' : 5},
                          'n_ens'     : 5}
    
    return parms