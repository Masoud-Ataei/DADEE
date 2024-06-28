# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Mar 13 2023, 10:26:41) 
# [GCC 9.4.0]
# Embedded file name: /home/masoud/Desktop/Conference_Article_Results/202306_AAAI/Regression_ToyExample/Online-SWAG_with_learning_Sigma_TN/TN_Estimator_Vanilla_Uncertainty.py
# Compiled at: 2023-08-06 12:07:23
# Size of source mod 2**32: 14884 bytes
from tensorflow.keras.optimizers import SGD, Adam
from keras import backend as K
import numpy as np
import json_tricks as json

import keras
import tensorflow as tf
from keras import layers
from keras        import models
import keras.backend as K
    
class prepare_input(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(prepare_input, self).__init__()        
        pass

    def call(self, inputs):
        theta  = inputs[:,0,tf.newaxis]
        stheta = tf.sin(theta)
        ctheta = tf.cos(theta)
        v      = inputs[:,1,tf.newaxis]
        x      = layers.concatenate([stheta,ctheta,v],axis =1) 
        return x

class prepare_input():
    def __init__(self, model):
        super(prepare_input, self).__init__()        
        self.model = model

    def call(self, inputs):
        theta  = inputs[:,0,tf.newaxis]
        stheta = tf.sin(theta)
        ctheta = tf.cos(theta)
        v      = inputs[:,1,tf.newaxis]
        x      = layers.concatenate([stheta,ctheta,v],axis =1) 
        return x
     

class estimate_Fx_NN_Anchor_Ensemble():
    # global coef_loss
    def __init__(self, n_net = 5, n_in = 2, n_out = 12, 
                 n_hidden = (4,12), lr = 0.01,        
                 activ = "tanh", reg = 0.0, coef_loss = 1.0, RMS_zero = True, prior = [0.0, 1.0]):        
        """
        This class will estimate Fx using a NN.
        n_in : number of elements in state, that is same as input of netwrok.
        n_out: number of elements of Fx, output of predict function.        
        n_hidden: number of hidden layer and the number of neurons in each of them (n_layer, n_neurons).
        lr   : learninf rate.
        active: activation of hidden layers.
        --- last layer doesn't hae any activation, due to Fx can be any real number.
        """
        self.type = 'Anchored'
        self.RMS_zero = RMS_zero
        self.lr = lr
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.activation = activ
        self.reg = reg
        self.coef_loss = coef_loss
        n_layer, n_node = n_hidden
        self.prior = prior
        self.n_ens = n_net
        nns = []
        theta_rms = []
        for i in range(n_net):
            nn, theta = self.def_model(n_in = n_in, n_out = n_out,
                                              activ = activ ,lr = lr,
                                              n_layer = n_layer, n_node = n_node, prior = prior, loss_func = self.custom_loss)
            nns.append(nn)
            theta_rms.append(theta)

        self.models_mu = nns
        self.theta_rms = theta_rms

        # self.model_mu = self.def_model(n_in=n_in, n_out=n_out, n_layer=n_layer, n_node=n_node, activ=activ, lr=lr, opt='Adam')
        # self.model_var , _ = self.def_model(n_in=n_in, n_out=n_out, n_layer=n_layer, n_node=n_node, activ=activ, lr=lr, opt='Adam', loss_func = None)
        self.WHist = []
        self.LHist = []

    def save_model(self, path='dump.sav'):
        _dump = {}
        _dump['type'] = self.type
        _dump['n_ens'] = self.n_ens
        _dump['lr'] = self.lr
        _dump['n_in'] = self.n_in
        _dump['n_out'] = self.n_out
        _dump['n_hidden'] = self.n_hidden
        _dump['activation'] = self.activation
        _dump['reg'] = self.reg
        _dump['coef_loss'] = self.coef_loss
        _dump['weights_mu'] =  [model_mu.get_weights() for model_mu in self.models_mu]
        # _dump['weights_var'] = self.model_var.get_weights()
        json.dump(_dump, path)

    def load_model(self, path='dump.sav'):
        _dump = json.load(path)
        self.lr = _dump['lr']
        self.n_in = _dump['n_in']
        self.n_out = _dump['n_out']
        self.n_hidden = _dump['n_hidden']
        self.activation = _dump['activation']
        self.reg = _dump['reg']
        self.coef_loss = _dump['coef_loss']
        weights_mu = _dump['weights_mu']
        # weights_var = _dump['weights_var']
        #self.model_mu.set_weights(weights_mu)
        # self.model_var.set_weights(weights_var)
        for w, m in zip(weights_mu, self.models_mu):
            m.set_weights(w)     
        #self.model_mu.set_weights(weights_mu)
        # self.model_var.set_weights(weights_var)
        self.n_ens       = _dump['n_ens']

    def set_coef_loss(self,val):
        self.coef_loss = val

    # def custom_mse(self, y_true, y_pred):
        
    #     # calculating squared difference between target and predicted values 
    #     loss = K.square(y_pred - y_true)  # (batch_size, ...)
        
    #     # multiplying the values with weights along batch dimension
    #     # loss = loss          # (batch_size, ...)
    #     # print(self.coef_loss, "coef")
    #     # summing both loss values along batch dimension 
    #     loss = K.sum(loss, axis=1)  * self.coef_loss       # (batch_size,)
        
    #     return loss

    def custom_loss(self, y_true, y_pred):     
        # print(self.model_index)   
        model = self.models_mu[self.model_index]
        theta = self.theta_rms[self.model_index]
        loss = self.mse_loss(y_true, y_pred)
        weights = model.trainable_weights
        
        loss_layer = self.reg * tf.reduce_sum([tf.reduce_sum((w-t) ** 2) for w,t in zip(weights,theta)]) / self.N
        # print(loss)
        # print(len(w), len(theta))
        return loss + loss_layer

    def def_model(self, n_in = 2, n_out = 12, n_layer = 4, n_node = 12, 
                  activ = "tanh", lr = 0.01, opt = 'Adam', specnorm = False, prior = [0.0, 1.0], loss_func = None):
        
        inputs = keras.Input(shape=(n_in,), dtype=tf.float32)        

        # x      = inputs[:,2:]        
        # theta  = inputs[:,0,tf.newaxis]
        # stheta = tf.sin(theta)
        # ctheta = tf.cos(theta)
        # v      = inputs[:,1,tf.newaxis]
        # x      = layers.concatenate([stheta,ctheta,v],axis =1)      
        x = inputs
        for _ in range(n_layer):
            x = layers.Dense(n_node, activation=activ,
                            #  kernel_regularizer=tf.keras.regularizers.L2(reg)
                             )(x)        
        
        Fx = layers.Dense(n_out, name='Fx',
                        #   kernel_regularizer=tf.keras.regularizers.L2(reg)
                          )(x)        
        
        nn = tf.keras.Model(inputs=inputs, outputs=Fx)

        # theta = tf.concat([tf.reshape(var, [-1]) for var in nn.trainable_variables], axis=0)
        if self.RMS_zero:
            # theta = tf.zeros(theta.shape, dtype= tf.float32)
            theta = [tf.zeros(v.shape, dtype=tf.float32) for v in nn.trainable_variables]            
        else:
            # theta = tf.random.normal(theta.shape,prior[0],prior[1],dtype=tf.float32)
            theta = [tf.random.normal(v.shape, prior[0],prior[1],dtype=tf.float32) for v in nn.trainable_variables]
        
        if opt == 'SGD':
            opt = SGD(learning_rate=lr)
        elif  opt == 'Adam':
            opt = Adam(learning_rate=lr)
        else:            
            raise 'Choose a optimizer: "SGD" or "Adam"'
        
        if loss_func is None:
            loss_fn   = keras.losses.MeanSquaredError()    
        else:
            loss_fn   = loss_func
        # loss_fn = keras.losses.MeanSquaredError()   
        self.mse_loss  = keras.losses.MeanSquaredError()
        metrics = [loss_fn]                
        metrics += ['mse', 'mae']
        nn.compile(optimizer= opt, loss=loss_fn, metrics=metrics)
        self.model_index = 0 
        return nn, theta

    def reset(self):
        n_layer, n_node = self.n_hidden
        nns = []
        theta_rms = []
        for i in range(self.n_net):            
            nn, theta = self.def_model(n_in = self.n_in, n_out = self.n_out, n_u = self.n_u, n_layer = n_layer, n_node = n_node, activ = self.activation ,lr = self.lr)
            nns.append(nn)
            theta_rms.append(theta)

        self.models_mu = nns
        self.theta_rms = theta_rms
        # self.model_mu  = self.def_model(self.n_in, n_out = self.n_out, n_u = self.n_u, n_layer = n_layer, n_node = n_node, activ = self.activation ,lr = self.lr, opt = 'Adam')
        # self.model_var , _ = self.def_model(self.n_in, n_out = self.n_out, n_u = self.n_u, n_layer = n_layer, n_node = n_node, activ = self.activation ,lr = self.lr, opt = 'Adam')

        # self.all_shapes = [a.shape           for a in self.nn.get_weights()]
        # self.w_size = np.sum([np.prod(shape) for shape in self.all_shapes])
        self.WHist  = []
        self.LHist  = []

    def set_lr(self, lr):
        self.lr = lr
        for model_mu in self.models_mu:
            model_mu .optimizer.learning_rate = lr
        # self.model_var.optimizer.learning_rate = lr


    def fit_models(self, data_set, epochs=10, batch_size=100, lr=0.001, delta=0.01, opt='Adam', const=0.0, verbose = True, val_data = None, preloss = None):
        self.N = len(data_set)
        self.set_lr(lr)
        xin = tf.convert_to_tensor(list(data_set))[:,0,:]
        yin = tf.convert_to_tensor(list(data_set))[:,1,:]
        if val_data is not None:
            xval = tf.convert_to_tensor(list(val_data))[:,0,:]
            yval = tf.convert_to_tensor(list(val_data))[:,1,:]
            val_data = (xval,yval)
        
        losses = []
        for i, model_mu in enumerate(self.models_mu):
            self.model_index = i            
            losses.append(model_mu .fit(xin, yin, validation_data = val_data, batch_size=batch_size , epochs=epochs, verbose=verbose).history)
        
        loss = {}
        for k in losses[0].keys():
            loss[k] = list(np.array([l[k] for l in losses ]).mean(axis=0))
        
        if preloss is not None:
            for k in preloss.keys():
                preloss[k] += loss[k]
            loss = preloss
        self.loss = loss 
        return self.loss
    
    def predictPos(self, xin, parallel=False):
        # xt = xin
        # my = self.model_mu(xt)
        # sy = self.model_var(xt)
        # sy = np.sqrt(sy ** 2)
        # return (my.numpy()[:, 0], sy[:, 0])

        xt = xin #[:,2:]
        # print(xin)
        F  = [] #np.zeros((S, Fsize))

        for model_mu in self.models_mu:
            F.append(model_mu(xt))

        F = np.array(F )
        my = F.mean(axis=0)[:,0]
        vy = F.var (axis=0)[:,0]
        # S  = self.model_var(xt) [:, 0]
        # sy = np.sqrt(np.abs(vy) + np.abs(S))
        sy = np.sqrt(np.abs(vy))
        return (my, sy)
    