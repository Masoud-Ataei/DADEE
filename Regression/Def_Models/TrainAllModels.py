from init_params import init_params
import sys
sys.path.append('../utils')
sys.path.append('../Def_Models')
from DEUP_Estimator_Anchor_Ensemble import estimate_Fx_NN_Anchor_Ensemble_DEUP as DADEE
from DEUP_Estimator_Vanilla_Uncertainty import estimate_Fx_NN_Vanilla_DEUP as DEUP
from Estimator_Anchor_Ensemble import estimate_Fx_NN_Anchor_Ensemble as Anchor


# from ReplayMemory import ReplayMemory
import tensorflow as tf
import numpy as np
import json_tricks as json
import matplotlib.pyplot as plt

from time import time

from utils import set_tf_deterministic,set_tf_GP_grow
from utils import data_2_dataset, set_seed, create_folders_if_not_exist
from utils import plot_graph4, print_stat2

set_tf_GP_grow()
set_tf_deterministic()
        
parms = init_params()
np.set_printoptions(precision=3, suppress=True)
# print(f"Simulation is started: p:{parms['controler']['p']} , cp:{parms['controler']['cp']}")

p_est = parms['estimator']
memcapacity = p_est['memcap']
swag, n_in, n_out, n_hidden, lr, activ = p_est['swag'],p_est['n_in'],p_est['n_out'],p_est['n_hidden'],p_est['lr'],p_est['activ']
n_hidden = (4,10)
activ = 'tanh'
lr = 0.01
n_ens = 5
# Load Dataset #Use Create_Dataset.ipynb file to generate dataset
data = json.load('../Dataset/1D_dataset.json')

train_data_set = data_2_dataset(data['train'])
valid_data_set = data_2_dataset(data['valid'])
test_data_set  = data_2_dataset(data['test'])
x,y              = data['test'][:,:-1],data['test'][:,-1]
x_train, y_train = data['train'][:,:-1],data['train'][:,-1]

print('### Train DADEE')
set_seed(parms['seed'])
batch_size = 20
epochs = 1000
n_ens = 5
FxEstimator = DADEE(n_net = n_ens, n_in=n_in, n_out=n_out, n_hidden=n_hidden, activ=activ, 
                            RMS_zero=False, prior=[0.0, 1.0], reg = 10.0)
t_start = time()
loss = FxEstimator.fit_models(train_data_set,epochs= epochs, epochs_var=20, batch_size=batch_size, lr = 0.0001, verbose=False, val_data=valid_data_set)

t_train = time()
my,sy = FxEstimator.predictPos(x)
t_infr  = time()

res = {'seed' : parms['seed'],
       'batch_size': batch_size,
       't_start': t_start,
       't_train': t_train,
       't_infr': t_infr,
       'loss': loss['mse'],
       'val_loss': loss['val_mse'],
       'epochs' : epochs,
       'loss_obj': loss
       }

my,sy = FxEstimator.predictPos(x)
fig, (ax1,ax2) = plot_graph4(None, None, x, y, x_train, y_train,my,  sy )
fig.suptitle(FxEstimator.type + ' network');

path = '../Results/'
plt.savefig(path + FxEstimator.type + ' network_data.jpg')
FxEstimator.save_model(path + FxEstimator.type + '.save')
res = print_stat2(FxEstimator, data, res)
json.dump(res, path + FxEstimator.type + '_res.save')

# Train DEUP
set_seed(parms['seed'])
batch_size = 20
epochs = 1000
FxEstimator = DEUP(n_in=n_in, n_out=n_out, n_hidden=n_hidden, activ=activ, reg = 0.01)
t_start = time()
loss = FxEstimator.fit_model(train_data_set,epochs= epochs,  batch_size=batch_size, lr = 0.0001, 
                             verbose=False, var_epochs = 20, val_data = valid_data_set)

t_train = time()
my,sy = FxEstimator.predictPos(x)
t_infr  = time()

res = {'seed' : parms['seed'],
       'batch_size': batch_size,
       't_start': t_start,
       't_train': t_train,
       't_infr': t_infr,
       'loss': loss['mse'],
       'val_loss': loss['val_mse'],
       'epochs' : epochs,
       'loss_obj': loss
       }

my,sy = FxEstimator.predictPos(x)
fig, (ax1,ax2) = plot_graph4(None, None, x, y, x_train, y_train,my,  sy )
fig.suptitle(FxEstimator.type + ' network');

plt.savefig(path + FxEstimator.type + ' network_data.jpg')
FxEstimator.save_model(path + FxEstimator.type + '.save')
res = print_stat2(FxEstimator, data, res)
json.dump(res, path + FxEstimator.type + '_res.save')

print('### Train Anchor')
set_seed(parms['seed'])
batch_size = 20
epochs = 1000
FxEstimator = Anchor(n_net = 5, n_in=n_in, n_out=n_out, n_hidden=n_hidden, activ=activ, 
                            RMS_zero=False, prior=[0.0, 1.0], reg = 10.00)
t_start = time()
loss = FxEstimator.fit_models(train_data_set,epochs= epochs,  batch_size=batch_size, lr = 0.0001, 
                              verbose=False, val_data=valid_data_set)
t_train = time()
my,sy = FxEstimator.predictPos(x)
t_infr  = time()

res = {'seed' : parms['seed'],
       'batch_size': batch_size,
       't_start': t_start,
       't_train': t_train,
       't_infr': t_infr,
       'loss':  loss['mse'],
       'val_loss': loss['val_mse'],
       'epochs' : epochs,
       'loss_obj' : loss
       }

my,sy = FxEstimator.predictPos(x)
fig, (ax1,ax2) = plot_graph4(None, None, x, y, x_train, y_train,my,  sy )

plt.savefig(path + FxEstimator.type + ' network_data.jpg')
FxEstimator.save_model(path + FxEstimator.type + '.save')
res = print_stat2(FxEstimator, data, res)
json.dump(res, path + FxEstimator.type + '_res.save')

print('### Train GP')
# GP
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from time import time
import functools
set_seed(parms['seed'])
from time import time
import functools
import GPy
import numpy as np
np.random.seed(parms['seed'])

# X = np.linspace(0,10,50)[:,None]
# k = GPy.kern.RBF(1)
# y = np.random.multivariate_normal(np.zeros(N),k.K(X)+np.eye(N)*np.sqrt(noise_var)).reshape(-1,1)

set_seed(parms['seed'])
batch_size = 20
epochs = 1

t_start = time()
### defual kernel is RBF, but it can be set like:
### k = GPy.kern.RBF(1)
### gpy = GPy.models.GPRegression(x_train,y_train.reshape((-1,1)), k)
gpy = GPy.models.GPRegression(x_train,y_train.reshape((-1,1)))
gpy.optimize('bfgs')
t_train = time()

def predict_Gpy(x):
    my,var = gpy.predict(x)
    sy = np.sqrt(var)
    return my[:,0], sy[:,0]

gpy.predictPos = predict_Gpy
# x_pred = np.linspace(-6, 6).reshape(-1,1)
my, sy = gpy.predictPos(x = x)
t_infr  = time()

FxEstimator = gpy
FxEstimator.type = 'GPy'
my,sy = gpy.predictPos(x)
fig, (ax1,ax2) = plot_graph4(None, None, x, y, x_train, y_train,my,  sy )

res = {'seed' : parms['seed'],
       'batch_size': batch_size,
       't_start': t_start,
       't_train': t_train,
       't_infr': t_infr,
       'loss': [0.0],
       'val_loss': [0.0],       
       'epochs' : epochs
       }

plt.savefig(path + FxEstimator.type + ' network_data.jpg')
# FxEstimator.save_model(path + FxEstimator.type + '.save')
res = print_stat2(FxEstimator, data, res)
json.dump(res, path + FxEstimator.type + '_res.save')

print('### Train DADEE - 20 ens')
set_seed(parms['seed'])
batch_size = 20
epochs = 1000
n_ens = 20
FxEstimator = DADEE(n_net = n_ens, n_in=n_in, n_out=n_out, n_hidden=n_hidden, activ=activ, 
                            RMS_zero=False, prior=[0.0, 1.0], reg = 10.0)
t_start = time()
loss = FxEstimator.fit_models(train_data_set,epochs= epochs, epochs_var=20, batch_size=batch_size, lr = 0.0001, verbose=False, val_data=valid_data_set)

t_train = time()
my,sy = FxEstimator.predictPos(x)
t_infr  = time()

res = {'seed' : parms['seed'],
       'batch_size': batch_size,
       't_start': t_start,
       't_train': t_train,
       't_infr': t_infr,
       'loss': loss['mse'],
       'val_loss': loss['val_mse'],
       'epochs' : epochs,
       'loss_obj': loss
       }

my,sy = FxEstimator.predictPos(x)
fig, (ax1,ax2) = plot_graph4(None, None, x, y, x_train, y_train,my,  sy )
fig.suptitle(FxEstimator.type + ' network - 20 ens');

plt.savefig(path + FxEstimator.type + ' network_data_20ens.jpg')
FxEstimator.save_model(path + FxEstimator.type + '_20ens.save')
res = print_stat2(FxEstimator, data, res)
json.dump(res, path + FxEstimator.type + '_res20ens.save')