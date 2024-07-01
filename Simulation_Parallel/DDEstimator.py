from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
import numpy as np
import json_tricks as json
import keras
from keras import layers
import tensorflow as tf
import edward2 as ed
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

def shuffle(ls):
    if len(ls)>0:
        indx = np.arange(len(ls[0]))
        np.random.shuffle(indx)
        for i,l in enumerate(ls):
            ls[i] = ls[i][indx]
	

def invsqrt_precision(M):
    """Compute ``M^{-0.5}`` as a tridiagonal matrix.

    Parameters
    ----------
    M : torch.Tensor

    Returns
    -------
    M_invsqrt : torch.Tensor
    """
    return _precision_to_scale_tril(M)

def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    #  modified to tf    
    Lf = tf.linalg.cholesky(tf.reverse(P, axis = (-2, -1)))
    L_inv = tf.transpose( tf.reverse(Lf, axis = (-2, -1)))
    Id = tf.eye(P.shape[-1], dtype=P.dtype)
    L = tf.linalg.triangular_solve(L_inv, Id, lower=True)
    return L

class prepare_input(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(prepare_input, self).__init__()        
        pass

    def call(self, inputs):
        theta  = inputs[:,0,tf.newaxis]
        stheta = tf.sin(theta)
        ctheta = tf.cos(theta)
        rest   = inputs[:,1:]
        #v     = inputs[:,2,tf.newaxis]
        x      = layers.concatenate([stheta,ctheta,rest],axis =1) 
        return x

class AddRealDDFx(keras.layers.Layer):
    def __init__(self, n_out, L=2.0, M=1.0):
        super(AddRealDDFx, self).__init__()   
        self.M = M     # Mass of Robot
        self.L = L     # Lenght of Robot
        self.n_out = n_out
        pass

    def call(self, inps):
        inputs, Fx = inps
        theta  = inputs[:,0,tf.newaxis]
        stheta = tf.sin(theta)
        ctheta = tf.cos(theta)
        w      = inputs[:,1,tf.newaxis]      
        v      = inputs[:,2,tf.newaxis]      
        F = tf.split(Fx, num_or_size_splits=self.n_out, axis=1)

        F[0]  = layers.add((F[0], v * ctheta))
        F[3]  = layers.add((F[3], v * stheta))
        F[6]  = layers.add((F[6], w))

        F[10] = layers.Lambda(lambda x:x+tf.constant(1.))(F[10]) #
        F[14] = layers.Lambda(lambda x:x+tf.constant(1./ self.M))(F[14]) #
                
        return tf.concat(F, axis=1)

class AddRealFx(keras.layers.Layer):
    def __init__(self, n_out, L=2.0):
        super(AddRealFx, self).__init__()   
        self.L = L     # Lenght of Robot
        self.n_out = n_out

        pass

    def call(self, inps):
        inputs, Fx = inps
        theta  = inputs[:,0,tf.newaxis]
        stheta = tf.sin(theta)
        ctheta = tf.cos(theta)
        v      = inputs[:,1,tf.newaxis]          
        F = tf.split(Fx, num_or_size_splits=self.n_out, axis=1)

        F[0]  = layers.add((F[0], v * ctheta))
        F[3]  = layers.add((F[3], v * stheta))
        
        F[7]  = layers.Lambda(lambda x:x+tf.constant(1. / self.L))(F[7]) #        
        F[8]  = layers.Lambda(lambda x:x-tf.constant(1. / self.L))(F[8]) #        
        F[10] = layers.Lambda(lambda x:x+tf.constant(1./2))(F[10]) #
        F[11] = layers.Lambda(lambda x:x+tf.constant(1./2))(F[11]) #
                
        return tf.concat(F, axis=1)

class estimate_DDFx():
    def __init__(self, parms_nn):
        """
        This class will estimate Fx using a NN.
        n_in : number of elements in state, that is same as input of netwrok.
        n_out: number of elements of Fx, output of predict function.
        n_u  : number of elements of control signal plus 1. (for u=[w, a] signal would be 3).
        n_hidden: number of hidden layer and the number of neurons in each of them (n_layer, n_neurons).
        lr   : learninf rate.
        active: activation of hidden layers.
        n_ens : number of ensembles 
        seed : for randomness of dropout
        --- last layer doesn't have any activation, due to Fx can be any real number.
        """
        self.type       = parms_nn['type']
        self.L          = parms_nn['car']['Lenght']
        self.M          = parms_nn['car']['Mass']
        self.car_model  = parms_nn['car']['car_model']
        self.lr         = parms_nn['lr']
        self.n_in       = parms_nn['n_in']
        self.n_out      = parms_nn['n_out']
        self.n_x        = parms_nn['n_x']
        self.n_u        = parms_nn['n_u']
        self.n_layer    = parms_nn['n_l']
        self.n_node     = parms_nn['n_node']        
        self.activation = parms_nn['activ']                
        self.seed       = parms_nn['seed']
        self.batch_size = parms_nn['batch_size']
        self.preTrain   = parms_nn['preTrain']
        self.DEUP       = parms_nn['DEUP']
        self.MLLV       = parms_nn['MLLV']
        self.reg        = parms_nn['reg']        
        self.models_mu = []        
        if   self.type =='Baseline':
            # self.train_model, self.predict_model = self.def_snn()
            if self.MLLV:
                raise 'not implemented'
            pass
        elif self.type == 'Ensemble':
            self.n_ens      = parms_nn['n_ens']        
        elif self.type == 'Ancor':             
            self.n_ens      = parms_nn['Ancor']['n_ens']        
            self.prior      = parms_nn['Ancor']['prior']  
            self.reg        = parms_nn['Ancor']['reg']  
            self.mse_loss  = keras.losses.MeanSquaredError()      
            if self.MLLV:
                raise 'not implemented'
        elif self.type == 'SWAG':
            self.swag       = parms_nn['swag']        
            if self.MLLV:
                raise 'not implemented'
        elif self.type == 'LA':
            self.LA       = parms_nn['LA']        
            self.temp     = parms_nn['LA']['temp']
            self.prior    = parms_nn['LA']['prior']
            self.sigma    = parms_nn['LA']['sigma']
            if self.MLLV:
                raise 'not implemented'
        elif self.type == 'MC-D':
            self.drop     = parms_nn['MC-D']['drop']
            inc_hid       =  parms_nn['MC-D']['inc_hid']
            self.n_ens    = parms_nn['MC-D']['n_ens']
            if inc_hid and self.drop>0:
                self.n_node = int(self.n_node / (1-self.drop))
            if self.MLLV:
                raise 'not implemented'
        elif self.type == 'Res':
            self.Width     = parms_nn['Res']['Width']
            self.BatchNorm = parms_nn['Res']['BatchNorm']
            self.ResLayers = parms_nn['Res']['ResLayers']            
            if self.MLLV:
                raise 'not implemented'
        
        else:
            raise Exception("Model type is not acceptable") 
        self.reset()
        
    def reset(self):
        self.models_mu = []        
        if   self.type =='Baseline':
            self.model_mu  = self.def_snn()
            if self.DEUP or self.MLLV:
                self.model_var = self.def_snn(lact_activ='relu')
            
        elif self.type == 'Ensemble':
            for _ in range(self.n_ens):
                model_mu = self.def_snn()
                self.models_mu.append(model_mu)

            if self.DEUP or self.MLLV:
                self.model_var = self.def_snn(lact_activ = 'relu')

            if self.MLLV:
                self.models_cmb = []
                for i,model_mu in enumerate(self.models_mu):
                    self.models_cmb.append( self.cmb_model(i, opt = 'Adam', loss_fn = self.custom_loss))

        elif   self.type =='SWAG':
            self.model_mu  = self.def_snn()
            if self.DEUP or self.MLLV:
                self.model_var = self.def_snn()
            self.all_shapes = [a.shape for a in self.model_mu[0].get_weights()]
            self.w_size = np.sum([np.prod(shape) for shape in self.all_shapes]) 
            T, K = self.swag['T'], self.swag['K']
            w_size  = self.w_size

            self.w_swa     = np.zeros(w_size)
            self.s_diag_sq = np.zeros(w_size)
            self.D_hat     = np.zeros((w_size, K)) # D_hat has K columns
        elif   self.type =='LA':
            self.model_mu  = self.def_snn()
            if self.DEUP or self.MLLV:
                self.model_var = self.def_snn()
            self.P         = len(tf.concat( [ tf.reshape(v, -1) for v in self.model_mu[0].variables], axis = 0))
            self.H         = tf.zeros((self.P,self.P))
        
        elif self.type == 'MC-D':
            self.model_mu  = self.def_snn(drop = self.drop)
            if self.DEUP or self.MLLV:
                self.model_var = self.def_snn(drop = self.drop)
        
        elif self.type == 'Ancor':            
            for _ in range(self.n_ens):
                model_mu = self.def_snn(theta = True, loss_fn=self.custom_loss_ancor)
                self.models_mu.append(model_mu)

            if self.DEUP or self.MLLV:
                self.model_var = self.def_snn(lact_activ = 'relu', theta = True, loss_fn=self.custom_loss_ancor_var)

            if self.MLLV:
                raise 'not implemented'
                self.models_cmb = []
                for i,model_mu in enumerate(self.models_mu):
                    self.models_cmb.append( self.cmb_model(i, opt = 'Adam', loss_fn = self.custom_loss, theta = True))

        elif self.type == 'Res':            
            self.model_mu  = self.def_resnn(Width=self.Width)
            if self.DEUP or self.MLLV:
                self.model_var = self.def_resnn(Width=self.Width, lact_activ='relu')
        else:
            raise Exception("Model type is not acceptable") 
        
        self.WHist  = []
        self.LHist  = []

    def cmb_model(self, indx, opt ='Adam', loss_fn = None):

        inputs = keras.Input(shape=(self.n_in,), dtype=tf.float32)        
        u      = keras.Input(shape=(self.n_u,) , dtype=tf.float32)

        mu  = self.models_mu[indx][0] ([inputs, u])
        var = self.model_var      [0] ([inputs, u])
        out = tf.concat((mu,var), axis = -1)
        nn = tf.keras.Model(inputs=[inputs, u], outputs=out)

        if opt == 'SGD':
            opt = SGD(learning_rate=self.lr)
        elif  opt == 'Adam':
            opt = Adam(learning_rate=self.lr)
        else:            
            raise 'Choose a optimizer: "SGD" or "Adam"'
        
        if loss_fn is None:
            loss_fn   = keras.losses.MeanSquaredError()
        nn.compile(optimizer= opt, loss=loss_fn, metrics=[loss_fn])
        return nn

    def custom_loss(self, y_true, y_pred):     
        my,var = tf.split(y_pred, 2, axis = -1)
        model = self.models_mu[self.model_index][0]
        var   = tf.abs(var) 
        loss = tf.reduce_mean( tf.reduce_sum( (y_true - my) ** 2 / (var + 0.01) / 2 + tf.math.log(var + 0.01) / 2, axis = -1))
        weights = model.trainable_weights
        
        loss_layer = self.reg * tf.reduce_sum([tf.reduce_sum((w) ** 2) for w in weights]) / self.N
        return loss + loss_layer

    def custom_loss_ancor(self, y_true, y_pred):     
        # print(self.model_index)   
        model = self.models_mu[self.model_index]
        theta = model[0].theta
        loss = self.mse_loss(y_true, y_pred)
        weights = model[0].trainable_weights
        
        loss_layer = self.reg * tf.reduce_sum([tf.reduce_sum((w-t) ** 2) for w,t in zip(weights,theta)]) / self.N
        return loss + loss_layer
    
    def custom_loss_ancor_var(self, y_true, y_pred):     
        model = self.model_var
        theta = model[0].theta
        loss = self.mse_loss(y_true, y_pred)
        weights = model[0].trainable_weights
        
        loss_layer = self.reg * tf.reduce_sum([tf.reduce_sum((w-t) ** 2) for w,t in zip(weights,theta)]) / self.N
        return loss + loss_layer
    
    def save_model(self, path = 'dump.sav'):
        _dump = {}
        _dump['lr'        ] = self.lr
        _dump['n_in'      ] = self.n_in
        _dump['n_out'     ] = self.n_out        
        _dump['n_hidden'  ] = self.n_hidden
        _dump['activation'] = self.activation
        _dump['reg'       ] = self.reg
        _dump['coef_loss' ] = self.coef_loss
        _dump['weights_mu'] = self.model_mu.get_weights()
        if self.DEUP or self.MLLV:
            _dump['weights_var'] = self.model_var.get_weights()
        json.dump(_dump, path)

    def load_model(self, path = 'dump.sav'):        
        _dump = json.load(path)
        self.lr         = _dump['lr'        ] 
        self.n_in       = _dump['n_in'      ]   
        self.n_out      = _dump['n_out'     ]          
        self.n_hidden   = _dump['n_hidden'  ]     
        self.activation = _dump['activation']       
        self.reg        = _dump['reg'       ]         
        self.coef_loss  = _dump['coef_loss' ]  
        weights_mu      = _dump['weights_mu' ]            
        self.model_mu .set_weights(weights_mu)
    
        if self.DEUP or self.MLLV:
            weights_var     = _dump['weights_var']        
            self.model_var.set_weights(weights_var)
    def dens_block(self, input_tensor,units, activ = 'relu'):
        """A block that has a dense layer at shortcut.
        # Arguments
            input_tensor: input tensor
            unit: output tensor shape
        # Returns
            Output tensor for the block.
        """
        x = layers.Dense(units)(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activ)(x)

        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activ)(x)

        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)

        shortcut = layers.Dense(units)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation(activ)(x)
        return x

    def identity_block(self, input_tensor,units, activ = 'relu'):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            units:output shape
        # Returns
            Output tensor for the block.
        """
        x = layers.Dense(units)(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activ)(x)

        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activ)(x)

        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation(activ)(x)

        return x

    def def_resnn(self, lact_activ = 'linear', drop = 0.0, theta = False, loss_fn = None, Width = 16):                
        inputs = keras.Input(shape=(self.n_in,), dtype=tf.float32)        
        u      = keras.Input(shape=(self.n_u,) , dtype=tf.float32)

        x = prepare_input()(inputs)
        for _ in range(self.n_layer):
            for l in self.ResLayers:
                if   l == 'I':
                    x = self.dens_block(x,Width, activ = self.activation)
                elif l == 'D':
                    x = self.identity_block(x,Width, activ = self.activation)
                            
        if self.BatchNorm:
            x = layers.BatchNormalization()(x)
 
        x = layers.BatchNormalization()(x)

        Fx   = layers.Dense(self.n_out, name='Fx', activation=lact_activ,
                            kernel_regularizer      = tf.keras.regularizers.L2(self.reg),
                            bias_regularizer        = tf.keras.regularizers.L2(self.reg),
                            # # activity_regularizer=regularizers.L2(1e-5)
                            )(x)   
        if self.preTrain:   
            if self.car_model == 'DD':    
                Fx  = AddRealDDFx(self.n_out, self.L, self.M) ((inputs, Fx))        
            else:
                Fx  = AddRealFx(self.n_out, self.L) ((inputs, Fx))        

        FxM  = layers.Reshape((-1,self.n_u))(Fx)        
        FxU  = layers.multiply([FxM,u])        
        if lact_activ == 'relu':
            xdot = keras.activations.relu(layers.add([FxU[:,:,i] for i in range(3)]))
        else:
            xdot = layers.add([FxU[:,:,i] for i in range(3)])
        
        nn_train   = tf.keras.Model(inputs=[inputs,u], outputs=xdot)
        nn_predict = tf.keras.Model(inputs=inputs, outputs=Fx)
        opt = Adam(learning_rate=self.lr)

        if loss_fn is None:
            nn_train  .compile(optimizer= opt, loss='mse', metrics=['mse'])
            nn_predict.compile(optimizer= opt, loss='mse', metrics=['mse'])
        else:
            nn_train  .compile(optimizer= opt, loss=loss_fn, metrics=['mse', loss_fn])
            nn_predict.compile(optimizer= opt, loss=loss_fn, metrics=['mse', loss_fn])
        
        if theta:
            theta = [tf.random.normal(v.shape, self.prior[0],self.prior[1],dtype=tf.float32) for v in nn_train.trainable_variables]
            theta_numpy = [t.numpy() for t in theta]
            nn_train  .theta = theta
            nn_predict.theta = theta
            nn_train  .theta_numpy = theta_numpy
            nn_predict.theta_numpy = theta_numpy
            
        return nn_train, nn_predict
    
    def def_snn(self, lact_activ = 'linear', drop = 0.0, theta = False, loss_fn = None):                
        inputs = keras.Input(shape=(self.n_in,), dtype=tf.float32)        
        u      = keras.Input(shape=(self.n_u,) , dtype=tf.float32)
        x = prepare_input()(inputs)
        for _ in range(self.n_layer):
            x = layers.Dense(self.n_node, activation=self.activation,
                                        kernel_regularizer      = tf.keras.regularizers.L2(self.reg),
                                        bias_regularizer        = tf.keras.regularizers.L2(self.reg),
                                        # # activity_regularizer=regularizers.L2(1e-5)
                                        )(x)     
            if drop > 0:
                x = layers.Dropout(drop)(x)

        Fx   = layers.Dense(self.n_out, name='Fx', activation=lact_activ,
                            kernel_regularizer      = tf.keras.regularizers.L2(self.reg),
                            bias_regularizer        = tf.keras.regularizers.L2(self.reg),
                            # # activity_regularizer=regularizers.L2(1e-5)
                            )(x)   
        if self.preTrain:   
            if self.car_model == 'DD':    
                Fx  = AddRealDDFx(self.n_out, self.L, self.M) ((inputs, Fx))        
            else:
                Fx  = AddRealFx(self.n_out, self.L) ((inputs, Fx))        

        FxM  = layers.Reshape((-1,self.n_u))(Fx)        
        FxU  = layers.multiply([FxM,u])        
        if lact_activ == 'relu':
            xdot = keras.activations.relu(layers.add([FxU[:,:,i] for i in range(3)]))
        else:
            xdot = layers.add([FxU[:,:,i] for i in range(3)])
        
        nn_train   = tf.keras.Model(inputs=[inputs,u], outputs=xdot)
        nn_predict = tf.keras.Model(inputs=inputs, outputs=Fx)
        opt = Adam(learning_rate=self.lr)

        if loss_fn is None:
            nn_train  .compile(optimizer= opt, loss='mse', metrics=['mse'])
            nn_predict.compile(optimizer= opt, loss='mse', metrics=['mse'])
        else:
            nn_train  .compile(optimizer= opt, loss=loss_fn, metrics=['mse', loss_fn])
            nn_predict.compile(optimizer= opt, loss=loss_fn, metrics=['mse', loss_fn])
        
        if theta:
            theta = [tf.random.normal(v.shape, self.prior[0],self.prior[1],dtype=tf.float32) for v in nn_train.trainable_variables]
            theta_numpy = [t.numpy() for t in theta]
            nn_train  .theta = theta
            nn_predict.theta = theta
            nn_train  .theta_numpy = theta_numpy
            nn_predict.theta_numpy = theta_numpy
            
        return nn_train, nn_predict

    def def_bnn(self):
        def prior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = keras.Sequential([
                tfpl.DistributionLambda(lambda t : tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n) * 0.1)),
            ])
            return prior_model
                
        def posterior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            posterior_model = keras.Sequential([
                tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
                tfpl.MultivariateNormalTriL(n)
            ])
            return posterior_model

        inputs = keras.Input(shape=(self.n_in,), dtype=tf.float32)        
        u      = keras.Input(shape=(self.n_u,), dtype=tf.float32)

        x = prepare_input()(inputs)
        for _ in range(self.n_layer):
            x = tfp.layers.DenseFlipout(self.n_node, activation=self.activation)(x)   

        Fx = layers.Dense(self.n_out, name='Fx')(x)        
        if self.preTrain:
            if self.car_model != 'DD':
                Fx  = AddRealDDFx(self.n_out, self.L) ((inputs, Fx))        
            else:
                Fx  = AddRealFx(self.n_out, self.L, self.M) ((inputs, Fx))        

        FxM = layers.Reshape((-1,self.n_u))(Fx)        
        FxU = layers.multiply([FxM,u])        
        xdot = layers.add([FxU[:,:,i] for i in range(3)])        
        nn_train   = tf.keras.Model(inputs=[inputs,u], outputs=xdot)
        nn_predict = tf.keras.Model(inputs=inputs, outputs=Fx)

        opt = Adam(learning_rate=self.lr)

        loss_fn   = keras.losses.MeanSquaredError()
        nn_train.compile(optimizer= opt  , loss=loss_fn, metrics=[loss_fn])
        nn_predict.compile(optimizer= opt, loss=loss_fn, metrics=[loss_fn])
        return nn_train, nn_predict
    
    def set_lr(self, lr):
        self.lr = lr
        if self.type in ['Baseline', 'MC-D', 'SWAG', 'LA', 'Res']:
            self.model_mu[0].optimizer.learning_rate = lr # nn_train
        elif self.type in ['Ensemble', 'Ancor']:
            for i,model_mu in enumerate(self.models_mu):
                model_mu[0].optimizer.learning_rate = lr # nn_train
        elif self.type == 'BNN':
            self.train_BNN.optimizer.learning_rate = lr # nn_train

        if self.DEUP or self.MLLV:
            self.model_var[0].optimizer.learning_rate = lr # nn_train   

        if self.MLLV:
            if self.type == 'Ensemble':
                for i,model_cmb in enumerate(self.models_cmb):
                    model_cmb.optimizer.learning_rate = lr # nn_train            
            else:
                raise 'not implemented'


    
    def fit_model(self, train_data, valid_data = None, valid_split = 0.0,  epochs = 10, batch_size = 100, lr = 0.01, var_epochs = None, verbose= True, preloss = None):        

        if var_epochs is None:
            var_epochs = epochs

        [xin, uin], xdotin = train_data
        shuffle([xin, uin,xdotin])
        
        if lr > 0.0:
            self.set_lr(lr)

        if self.MLLV:
            raise 'not implemented'        

        xin = xin[:, 2:]
        h = self.model_mu[0] .fit([xin,uin], xdotin, validation_data =valid_data, validation_split = valid_split, batch_size=batch_size , epochs=epochs, verbose=verbose).history
        
        if self.DEUP:
            ein = xdotin - self.model_mu [0]([xin,uin])
            ein = tf.square(ein)
            if valid_data is not None:
                [v_xin, v_uin], v_xdotin = valid_data
                eval = v_xdotin - self.model_mu[0] ([v_xin, v_uin])
                eval = tf.square(eval)
                valid_data = ([v_xin, v_uin],eval)
            h_var = self.model_var[0].fit([xin,uin], ein, validation_data =valid_data, validation_split = valid_split, batch_size=batch_size , epochs=var_epochs, verbose=verbose).history
            for k in h_var.keys():                
                h['var_' + k] = h_var[k]
        
        if preloss is None:
            preloss = h
        else:
            for k in h.keys():
                if k in preloss.keys():
                    preloss[k] += h[k]
                else:
                    preloss[k] = h[k]
        self.loss = h
        return h
    
    def fit_model_mc(self, memory, epochs = 10, batch_size = 100, lr = 0.0, delta = 0.01, opt = 'Adam', const_var = 0.01, const = 0.0, verbose= True):        
        xin, uin, xdotin = memory.get_samples(int(len(memory)))
        NTrain = int(len(memory) * 0.9)
        NVal   = int(len(memory) * 0.1)
        N      = int(len(memory) )
        self.N = NTrain        
        if NVal > 0:
            xinV, uinV, xdotinV = xin[:NVal], uin[:NVal], xdotin[:NVal]
            xin, uin, xdotin = xin[NVal:], uin[NVal:], xdotin[NVal:]
            
        if lr > 0.0:
            self.set_lr(lr)

        if self.MLLV:
            raise 'not implemented'        
        xin = xin[:, 2:]        
        h = self.model_mu[0] .fit([xin,uin], xdotin, batch_size=batch_size , epochs=epochs, verbose=verbose).history['loss']
        loss_mu = h[-1]
        # print(loss_mu[-1])
        self.loss = np.mean(loss_mu)
        loss_var = 0.0
        if self.DEUP:
            yhat  = [self.model_mu [0]([xin,uin], training = True) for _ in range(self.n_ens)]
            ymean = tf.reduce_mean         (yhat, axis = 0)
            yvar  = tf.math.reduce_variance(yhat, axis = 0)
            self.val_error0 = tf.reduce_mean( tf.square(ymean - xdotin))
            
            ein = (xdotin - ymean) ** 2 - yvar
            ein = tf.math.maximum(ein, tf.zeros(ein.shape))            
            
            h   = self.model_var[0].fit([xin,uin], ein, batch_size=batch_size , epochs=epochs, verbose=verbose).history['loss']
            loss_var = h[-1]
            self.loss = np.mean(loss_mu) + loss_var

        if self.DEUP or self.MLLV:
            self.loss_val = 0.0

            if NVal > 0:
                xinV = xinV[:, 2:]        
                
                yhat  = [self.model_mu [0]([xinV,uinV], training = True) for _ in range(self.n_ens)]
                ymean = tf.reduce_mean         (yhat, axis = 0)
                yvar  = tf.math.reduce_variance(yhat, axis = 0)
                  
                self.val_error0 = tf.reduce_mean( tf.square(ymean - xdotinV))            
                ein = (xdotinV - ymean) ** 2 - yvar
                ein = tf.math.maximum(ein, tf.zeros(ein.shape))     
                var = self.model_var[0]([xinV,uinV], trianing = True)
                self.val_error1 = tf.reduce_mean( tf.square(var - ein))
                self.loss_val  = (self.val_error0  +self.val_error1 ).numpy()

        else:
            self.loss_val = 0.0
            if NVal > 0:
                xinV = xinV[:, 2:]        
                yhat  = [self.model_mu [0]([xinV,uinV], training = True) for _ in range(self.n_ens)]
                ymean = tf.reduce_mean         (yhat, axis = 0)
                yvar  = tf.math.reduce_variance(yhat, axis = 0)
                self.val_error0 = tf.reduce_mean( tf.square(ymean - xdotinV))
                ein = (xdotinV - ymean) ** 2 - yvar                
                self.loss_val  = (tf.reduce_mean(ein) ).numpy() + self.val_error0.numpy()
                
        return self.loss, self.loss_val

    def fit_models(self, train_data, valid_data = None, valid_split = 0.0, epochs = 10, batch_size = 100, lr = 0.01, var_epochs = None, verbose= True, preloss = None):        
        if var_epochs is None:
            var_epochs = epochs

        [xin, uin], xdotin = train_data
        xin = xin[:, 2:]
        shuffle([xin, uin,xdotin])

        if lr > 0.0:
            self.set_lr(lr)

        if self.MLLV:
            
            h  = None
            for i, model_cmb in enumerate( self.models_cmb):
                self.model_index = i
                h_ = model_cmb.fit([xin,uin], xdotin, validation_data =valid_data, validation_split = valid_split, batch_size=batch_size , epochs=epochs, verbose=verbose).history
                if h is None:
                    h = {}
                    for k in h_.keys():
                        h[k]  = np.array(h_[k])
                else:
                    for k in h_.keys():
                        h[k] += (np.array(h_[k]) * i)                
                        h[k]  = h[k] / (i+1)
            for k in h.keys():
                h[k]  = list(h[k])
        else:            
            h  = None
            for i,model_mu in enumerate( self.models_mu):
                self.model_index = i
                h_ = model_mu[0] .fit([xin,uin], xdotin, batch_size=batch_size , epochs=epochs, verbose=verbose).history
                if h is None:
                    h = {}
                    for k in h_.keys():
                        h[k]  = np.array(h_[k])
                else:
                    for k in h_.keys():
                        h[k] += (np.array(h_[k]) * i)                
                        h[k]  = h[k] / (i+1)
            for k in h.keys():
                h[k]  = list(h[k])

        if self.DEUP:
            ### 3
            yhat  = [model_mu [0]([xin,uin]) for model_mu in self.models_mu]
            ymean = tf.reduce_mean         (yhat, axis = 0)
            yvar  = tf.math.reduce_variance(yhat, axis = 0)
            
            ein = (xdotin - ymean) ** 2 - yvar
            ein = tf.math.maximum(ein, tf.zeros(ein.shape))            
            
            if valid_data is not None:
                [v_xin, v_uin], v_xdotin = valid_data

                v_yhat  = [model_mu [0]([v_xin,v_uin]) for model_mu in self.models_mu]
                v_ymean = tf.reduce_mean         (v_yhat, axis = 0)
                v_yvar  = tf.math.reduce_variance(v_yhat, axis = 0)
                
                v_ein = (v_xdotin - v_ymean) ** 2 - v_yvar
                v_ein = tf.math.maximum(v_ein, tf.zeros(v_ein.shape))     
                valid_data = ([v_xin, v_uin],v_ein)
            h_var   = self.model_var[0].fit([xin,uin], ein, batch_size=batch_size , epochs=var_epochs, verbose=verbose).history
            for k in h_var.keys():                
                h['var_' + k] = h_var[k]
        
        if preloss is None:
            preloss = h
        else:
            for k in h.keys():
                if k in preloss.keys():
                    preloss[k] += h[k]
                else:
                    preloss[k] = h[k]
        self.loss = h
        return h

    def w_to_flat(self, weights):
        flattened = np.hstack([w.flatten() for w in weights])
        return flattened

    def flat_to_w(self, flat):
        all_shapes = self.all_shapes
        weights = []
        start = 0
        for i, shape in enumerate(all_shapes):
            a = (flat[start:start + np.prod(shape)].reshape)(*shape)
            start = start + np.prod(shape)
            weights.append(a)
        else:
            return weights
        
    def get_w_rand2(self, S=-1):
        w_swa  = self.w_swa
        s_diag = self.s_diag_sq ** 2
        D_hat  = self.D_hat
        K      = self.swag['K']
        S_low = D_hat @ D_hat.T
        Sigma = 0.5 * s_diag + S_low / 2 / (K - 1)
        if S <= 0:
            S = self.swag['S']
        Ws = []
        json.dump([w_swa, Sigma], 'temp_w_rand.log' )
        w_rand = np.random.multivariate_normal(mean=w_swa, cov=Sigma, size=S)
        for i, w in enumerate(w_rand):
            Ws.append(self.flat_to_w(w))
        else:
            return Ws
        
    def fit_swag(self, memory, epochs_burn = 1, epochs_var = 10, batch_size = 64, lr = 0.0, delta = 0.01, verbose = True):
        xin, uin, xdotin = memory.get_samples(int(len(memory)))
        NTrain = int(len(memory) * 0.9)
        NVal   = int(len(memory) * 0.1)
        N      = int(len(memory) )
        self.N = NTrain
        
        if NVal > 0:
            xinV, uinV, xdotinV = xin[:NVal], uin[:NVal], xdotin[:NVal]
            xin, uin, xdotin    = xin[NVal:], uin[NVal:], xdotin[NVal:]

        if lr > 0.0:
            self.set_lr(lr)
            
        xin = xin[:, 2:]        
        if self.MLLV:
            raise 'need implementation'
        
            T, K = self.swag['T'], self.swag['K']
            w_size  = self.w_size
            w_swa   = np.zeros(w_size)        
            w_swa2  = np.zeros(w_size)
            w_bar   = np.zeros(w_size)        
            D_hat   = np.zeros((w_size, K)) # D_hat has K columns
            Ws      = np.zeros((w_size, K)) # D_hat has K columns
            loss = []
            for t in range(T):
                k = K - (T - t )
                ## train one epoch
                ## estimator.train ...           

                # h = self.nn.fit(x, y, batch_size = batch_size, epochs = 1, verbose = 0).history            
                
                loss.append(self.model_cmb[0] .fit([xin,uin], xdotin, batch_size=batch_size , epochs=epochs_burn, verbose=verbose).history['loss'])

                weights     = self.model_mu[0].get_weights()
                w_flattened = self.w_to_flat(weights)
                w_swa  += (w_flattened / T)  # avg (w) 1 to T
                w_swa2 += (np.power(w_flattened,2) / T) # avg (w^2) 1 to T
                w_bar   = ((w_bar * t) + w_flattened) / (t+1) # avg (w) 1 to i
                if k >= 0 and k< K:
                    Di         = w_flattened - w_bar # wi - w_bar
                    D_hat[:,k] = Di # put each Di in correspond column
                    Ws   [:,k] = w_flattened
                # self.WHist.append(w_flattened)
                # self.LHist.append(loss[-1])
            # w_swa, and w_swa2 is correct
            self.w_swa      = w_swa
            # self.s_diag     = np.diag(w_swa2 - np.power(w_swa,2))
            self.s_diag_sq  = np.diag( np.sqrt( w_swa2 - np.power(w_swa,2)+ 1e-5) )        
            # self.s_low_rank = D_hat @ D_hat.T /(K-1)
            self.D_hat      = D_hat
            self.Ws         = Ws

        else:            
            T, K = self.swag['T'], self.swag['K']
            w_size  = self.w_size
            w_swa   = np.zeros(w_size)        
            w_swa2  = np.zeros(w_size)
            w_bar   = np.zeros(w_size)        
            D_hat   = np.zeros((w_size, K)) # D_hat has K columns
            Ws      = np.zeros((w_size, K)) # D_hat has K columns
            loss = []
            for t in range(T):
                k = K - (T - t )
                ## train one epoch
                ## estimator.train ...           

                loss.append(self.model_mu[0] .fit([xin,uin], xdotin, batch_size=batch_size , epochs=epochs_burn, verbose=verbose).history['loss'][-1])

                weights     = self.model_mu[0].get_weights()
                w_flattened = self.w_to_flat(weights)
                w_swa  += (w_flattened / T)  # avg (w) 1 to T
                w_swa2 += (np.power(w_flattened,2) / T) # avg (w^2) 1 to T
                w_bar   = ((w_bar * t) + w_flattened) / (t+1) # avg (w) 1 to i
                if k >= 0 and k< K:
                    Di         = w_flattened - w_bar # wi - w_bar
                    D_hat[:,k] = Di # put each Di in correspond column
                    Ws   [:,k] = w_flattened
                
            # w_swa, and w_swa2 is correct
            self.w_swa      = w_swa
            self.s_diag_sq  = np.diag( np.sqrt( w_swa2 - np.power(w_swa,2)+ 1e-5) )        
            self.D_hat      = D_hat
            self.Ws         = Ws
        
        self.loss = np.mean(loss)
        if self.DEUP:
            backup_W = self.model_mu[0].get_weights()
            yhat = []
            w_rand = self.get_w_rand2()
            self.w_rand = w_rand       
            for i, wi_rand in enumerate(self.w_rand):
                self.model_mu[0].set_weights(wi_rand)
                _yhat = self.model_mu[0]([xin, uin])
                yhat.append(_yhat)

            self.model_mu[0].set_weights(backup_W)

            ymean = tf.reduce_mean    (yhat, axis = 0)
            ystd  = tf.math.reduce_std(yhat, axis = 0)
            ein = (xdotin - ymean) ** 2 - ystd ** 2
            ein = tf.math.maximum(ein, tf.zeros(ein.shape))
            ein = tf.sqrt(ein)
            
            h   = self.model_var[0].fit([xin,uin], ein, batch_size=batch_size , epochs=epochs_var, verbose=verbose).history
            loss_var = h['loss'][-1]
            self.loss = np.mean(loss) + loss_var

        if self.DEUP or self.MLLV:
            self.loss_val = 0.0
            if NVal > 0 :
                xinV = xinV[:,2:]
                backup_W = self.model_mu[0].get_weights()
                yhat = []
                w_rand = self.get_w_rand2()
                self.w_rand = w_rand
            
                for i, wi_rand in enumerate(self.w_rand):
                    self.model_mu[0].set_weights(wi_rand)
                    _yhat = self.model_mu[0]([xinV, uinV])
                    yhat.append(_yhat)
                self.model_mu[0].set_weights(backup_W)

                ymean = tf.reduce_mean         (yhat, axis = 0)
                yvar  = tf.math.reduce_variance(yhat, axis = 0)
                self.val_error0 = tf.reduce_mean( tf.square(ymean - xdotinV))
                ein = (xdotinV - ymean) ** 2 - yvar
                ein = tf.math.maximum(ein, tf.zeros(ein.shape))     
                var = self.model_var[0]([xinV,uinV])
                self.val_error1 = tf.reduce_mean( tf.square(var - ein))
                self.loss_val  = (self.val_error0  +self.val_error1 ).numpy()

        else:
            self.loss_val = 0.0
            if NVal > 0 :
                xinV = xinV[:,2:]
                backup_W = self.model_mu[0].get_weights()
                yhat = []
                w_rand = self.get_w_rand2()
                self.w_rand = w_rand
            
                for i, wi_rand in enumerate(self.w_rand):
                    self.model_mu[0].set_weights(wi_rand)
                    _yhat = self.model_mu[0]([xinV, uinV])
                    yhat.append(_yhat)
                self.model_mu[0].set_weights(backup_W)

                ymean = tf.reduce_mean         (yhat, axis = 0)
                yvar  = tf.math.reduce_variance(yhat, axis = 0)
                self.val_error0 = tf.reduce_mean( tf.square(ymean - xdotinV))
                ein = (xdotinV - ymean) ** 2 - yvar                
                self.loss_val  = (tf.reduce_mean(ein) ).numpy() + (self.val_error0 ).numpy()
        return self.loss, self.loss_val
    
    def Jaccobian_Xdot_theta(self, x, u):
        Jth = []

        with tf.GradientTape() as g:  
            g.reset()
            out = self.model_mu[0]([x,u])

        g0 = g.jacobian(out, self.model_mu[0].variables)
        Jth = tf.concat([tf.reshape(g_, (g_.shape[0],g_.shape[1],-1)) for g_ in g0], axis = -1)
        return Jth, out

    def Jaccobian_Fx_theta(self, x):
        Jth = []
        with tf.GradientTape(persistent=True) as g:  
            g.reset()
            out = self.model_mu[1](x)
            for i in range(x.shape[0]):
                Jth.append([])
                for io in range(self.n_out):
                    dmodel_dtheta = g.gradient(out[i,io], self.model_mu[1].variables)  # (4*x^3 at x = 3)
                    dl = [tf.reshape(dm, (-1))   for dm in dmodel_dtheta]            
                    Jth[-1].append(tf.concat(dl, axis = 0))
        Jth = tf.convert_to_tensor(Jth) #, out
        return Jth, out

    def curv_closure(self, x, u, y):
        loss_fn   = keras.losses.MeanSquaredError()
        Js, f = self.Jaccobian_Xdot_theta(x, u)
        loss = 0.5 * loss_fn(f, y)  * len(y)      
        H_ggn = tf.einsum('mkp,mkq->pq', Js, Js)
        return loss, H_ggn

    def fit_curv(self, x, u, y, batch_size = 100):
        P = self.P
        H = tf.zeros((P,P))
        loss = 0
        end = 0
        N = x.shape[0]
        for start in range(0, N, batch_size):
            end = start + batch_size
            if end > N:
                end = N
            x_batch_train = x[start:end]
            u_batch_train = u[start:end]
            y_batch_train = y[start:end]
            loss_batch, H_batch = self.curv_closure(x_batch_train, u_batch_train, y_batch_train)
            loss += loss_batch
            H += H_batch
        return loss, H
    
    def fit_LA(self, memory, epochs = 10, batch_size = 100, lr = 1e-3, loss_coef = 1.0, const_var = 0.01, const = 0.0, verbose=True):        
        xin, uin, xdotin = memory.get_samples(int(len(memory)))
        NTrain = int(len(memory) * 0.9)
        NVal   = int(len(memory) * 0.1)
        N      = int(len(memory) )
        self.N = NTrain
        if NVal > 0:
            xinV, uinV, xdotinV = xin[:NVal], uin[:NVal], xdotin[:NVal]
            xin, uin, xdotin    = xin[NVal:], uin[NVal:], xdotin[NVal:]

        if lr > 0.0:
            self.set_lr(lr)

        _xin = xin
        xin = xin[:, 2:]        
        if self.MLLV:
            raise 'need implementaion'
            h = self.model_cmb[0] .fit([xin,uin], xdotin, batch_size=batch_size , epochs=epochs, verbose=verbose).history        
        else:
            h = self.model_mu[0] .fit([xin,uin], xdotin, batch_size=batch_size , epochs=epochs, verbose=verbose).history['loss']
            loss_mu = h[-1]

        self.loss = loss_mu

        nll_loss, self.H = self.fit_curv(xin, uin, xdotin, batch_size)
        loss_var = 0.0
        if self.DEUP:
            # ### 3
            ymean , ystd = self.predictXdot(_xin, uin, s = -1)
            ein = (xdotin - ymean) ** 2 - ystd ** 2
            ein = tf.math.maximum(ein, tf.zeros(ein.shape))

            h   = self.model_var[0].fit([xin,uin], ein, batch_size=batch_size , epochs=epochs, verbose=verbose).history['loss']
            loss_var = h[-1]    
            self.loss = loss_mu + loss_var
        
        if self.DEUP or self. MLLV:
            self.loss_val = 0.0
            if NVal > 0:

                ymean , ystd = self.predictXdot(xinV, uinV, s = -1)
                yvar  = ystd ** 2
                self.val_error0 = tf.reduce_mean( tf.square(ymean - xdotinV))
                ein = (xdotinV - ymean) ** 2 - yvar
                ein = tf.math.maximum(ein, tf.zeros(ein.shape))     
                var = self.model_var[0]([xinV[:,2:],uinV])
                self.val_error1 = tf.reduce_mean( tf.square(var - ein))
                self.loss_val  = (self.val_error0  +self.val_error1 ).numpy()
        else:
            self.loss_val = 0.0
            if NVal > 0:
                
                ymean , ystd = self.predictXdot(xinV, uinV, s = -1)
                yvar  = ystd ** 2
                self.val_error0 = tf.reduce_mean( tf.square(ymean - xdotinV))
                ein = (xdotinV - ymean) ** 2 - yvar
                ein = tf.math.maximum(ein, tf.zeros(ein.shape))                     
                self.loss_val  = (tf.reduce_mean(ein) ).numpy() + self.val_error0.numpy()

        return self.loss, self.loss_val
        

    def predictXdot(self, xin, uin, s = 0):     
        xt = xin[:,2:]                
        if self.type == 'Baseline':
            my = self.model_mu [0] ([xt, uin])
            my = my.numpy()
            if self.DEUP or self.MLLV:
                vy = self.model_var[0] ([xt, uin])
                sy = np.sqrt(vy)
            else:
                sy = my * 0.0

            if s > 0:
                json.dump([my,sy], 'temp_predictXdot.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
                # return my
            else:
                return (my,sy)

        if self.type == 'Res':
            my = self.model_mu [0] ([xt, uin])
            my = my.numpy()
            if self.DEUP or self.MLLV:
                vy = self.model_var[0] ([xt, uin])
                sy = np.sqrt(vy)
            else:
                sy = my * 0.0

            if s > 0:
                json.dump([my,sy], 'temp_predictXdot.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
            else:
                return (my,sy)

        if self.type == 'Ensemble' or self.type == 'Ancor':           
            F  = [] #np.zeros((S, Fsize))        
            for model in self.models_mu:
                F0 = model[0]([xt, uin])            
                F  .append(F0)

            F = np.array(F )            
            my = F.mean(axis=0)
            vy = F.var (axis=0)
            if self.DEUP or self.MLLV:
                S = self.model_var[0] ([xt, uin]).numpy()
                sy = np.sqrt(np.abs(vy) + np.abs(S))
                self.S = S
            else:
                ### 2
                sy = np.sqrt(np.abs(vy))
            
            self.F = F

            if s > 0:
                json.dump([my,sy], 'temp_predictXdot.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
                # return my
            elif s == -1:
                return F
            else:
                return (my,sy)
        
        if self.type == 'SWAG':            
            backup_W = self.model_mu[0].get_weights()
            F = []
        
            w_rand = self.get_w_rand2()
            self.w_rand = w_rand
        
            for i, wi_rand in enumerate(self.w_rand):
                self.model_mu[0].set_weights(wi_rand)
                F0 = self.model_mu[0]([xt, uin])
                F.append(F0)

            self.model_mu[0].set_weights(backup_W)
            F = np.array(F)

            if self.DEUP or self.MLLV:
                S  = self.model_var[0]([xt, uin]).numpy()
                sy = np.sqrt(F.std(axis=0) ** 2 + S ** 2)
                self.S = S
            else:
                sy = np.sqrt(F.std(axis=0) ** 2)

            my = F.mean(axis=0)

            self.F = F
            
            
            if s > 0:
                json.dump([my,sy], 'temp_predictXdot.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
            elif s == -1:
                return F
            else:
                return (my,sy)
        if self.type == 'LA':
            Js, my = self.Jaccobian_Xdot_theta(xt, uin)

            posterior_precision = self.posterior_precision()
            scale = invsqrt_precision(posterior_precision)
            posterior_covariance = scale @ tf.transpose( scale)
            f_var = tf.einsum('ncp,pq,nkq->nck', Js, posterior_covariance, Js)
            
            f_var = tf.linalg.diag_part(f_var)      

            if s == -1:
                return (my.numpy(), np.sqrt(f_var.numpy()))
            
            my = my.numpy()
            
            if self.DEUP or self.MLLV:
                S  = self.model_var[0]([xt, uin]).numpy()
                sy = tf.math.sqrt(f_var + S ** 2).numpy()
            else:
                sy = tf.math.sqrt(f_var).numpy()
            
            if s > 0:
                json.dump([my,sy], 'temp_predictXdot.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
            else:
                return (my,sy)

        if self.type == 'MC-D':           
            F  = [] #np.zeros((S, Fsize))        
            for _ in range(self.n_ens):
                F0 = self.model_mu[0]([xt, uin], training = True)            
                F  .append(F0)

            F = np.array(F )            
            my = F.mean(axis=0)
            vy = F.var (axis=0)
            if self.DEUP or self.MLLV:
                S = self.model_var[0] ([xt, uin], training = True).numpy()
                sy = np.sqrt(np.abs(vy) + np.abs(S))
                self.S = S
            else:
                ### 2
                sy = np.sqrt(np.abs(vy))
                        
            self.F = F

            if s > 0:
                json.dump([my,sy], 'temp_predictXdot.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
            elif s == -1:
                return F
            else:
                return (my,sy)

        raise Exception("Model type is not acceptable") 

    def H_factor(self):
        sigma2 = self.sigma ** 2
        return 1 / sigma2 / self.temp

    def prior_precision_diag(self):
        return self.prior * tf.ones(self.P)
    
    def posterior_precision(self):
        posterior_precision = self.H_factor() * self.H + tf.linalg.diag(self.prior_precision_diag())   # prior_precision_diag_logdet(prior_prec, P)
        return posterior_precision
    
    def check_nan(self, x):
        if np.isnan(x).any():
            print(x)
            json.dump('nan.log', x)
            return True
        return False
    
    def predictFx(self, xin, s = 1):    
        x  = xin 
        xt = xin[:,2:]        
        xin = xin[:,2:]     
        if self.type == 'Baseline':
            my = self.model_mu [1] (xt)
            my = my.numpy()

            if self.DEUP or self.MLLV:
                vy = self.model_var[1] (xt)
                sy = np.sqrt(vy)
            else:
                sy = my * 0.0

            if s > 0:
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]

            else:
                return (my,sy)
        
        if self.type == 'Res':
            my = self.model_mu [1] (xt)
            my = my.numpy()

            if self.DEUP or self.MLLV:
                vy = self.model_var[1] (xt)
                sy = np.sqrt(vy)
            else:
                sy = my * 0.0

            if s > 0:
                json.dump([my,sy], 'temp_predictFx.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
            else:
                return (my,sy)        

        if self.type == 'Ensemble' or self.type == 'Ancor':
            F  = [] #np.zeros((S, Fsize))        
            for model in self.models_mu:
                F0 = model[1](xt)            
                F  .append(F0)

            F = np.array(F )
            
            my = F.mean(axis=0)
            vy = F.var (axis=0)
            if self.DEUP or self.MLLV:
                S = self.model_var[1] (xt).numpy()            
                sy = np.sqrt(np.abs(vy) + np.abs(S))
                self.S = S
            else:
                ### 2
                sy = np.sqrt(np.abs(vy))
            
            self.F = F
            
            if s > 0:
                json.dump([my,sy], 'temp_predictFx.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
            elif s == -1:
                return F
            else:
                return (my,sy)
        
        if self.type == 'SWAG':            
            backup_W = self.model_mu[0].get_weights()
            F = []
        
            w_rand = self.get_w_rand2()
            self.w_rand = w_rand
        
            for i, wi_rand in enumerate(self.w_rand):
                self.model_mu[0].set_weights(wi_rand)
                F0 = self.model_mu[1](xt)
                F.append(F0)

            self.model_mu[0].set_weights(backup_W)
            F = np.array(F)            
            self.F = F
            my = F.mean(axis=0)
            vy = F.var (axis=0)
            if self.DEUP or self.MLLV:
                S = self.model_var[1] (xt).numpy()            
                sy = np.sqrt(np.abs(vy) + np.abs(S))
                self.S = S
            else:
                sy = np.sqrt(np.abs(vy)) 

        
            self.F = F
            
            if s > 0:
                json.dump([my,sy], 'temp_predictFx.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
            elif s == -1:
                return F
            else:
                return (my,sy)
        
        if self.type == 'LA':
            Js, my = self.Jaccobian_Fx_theta(xin)

            posterior_precision = self.posterior_precision()
            scale = invsqrt_precision(posterior_precision)
            posterior_covariance = scale @ tf.transpose( scale)
            f_var = tf.einsum('ncp,pq,nkq->nck', Js, posterior_covariance, Js)
            
            f_var = tf.linalg.diag_part(f_var)        
            my = my.numpy()
            if self.DEUP or self.MLLV:
                S = self.model_var[1] (xt).numpy()            
                sy = np.sqrt(f_var + S ** 2)
                self.S = S
            else:
                sy = np.sqrt(f_var)      

            if s == -1:
                return (my, np.sqrt(f_var.numpy()))
            
            if s > 0:
                json.dump([my,sy], 'temp_predictFx.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
            else:
                return (my,sy)

        if self.type == 'MC-D':           
            F  = [] #np.zeros((S, Fsize))        
            for _ in range(self.n_ens):
                F0 = self.model_mu[1](xt, training = True)            
                F  .append(F0)

            F = np.array(F )
            
            my = F.mean(axis=0)
            vy = F.var (axis=0)
            if self.DEUP or self.MLLV:
                S = self.model_var[1] (xt, training = True).numpy()            
                sy = np.sqrt(np.abs(vy) + np.abs(S))
                self.S = S
            else:
                ### 2
                sy = np.sqrt(np.abs(vy))
            
            self.F = F
            
            if s > 0:
                json.dump([my,sy], 'temp_predictFx.log')
                return np.array([np.random.multivariate_normal(my_, np.diag(sy_), s) for my_,sy_ in zip(my,sy)]).astype(np.float32) [:,:,:]
                # return my
            elif s == -1:
                return F
            else:
                return (my,sy)
        
        raise Exception("Model type is not acceptable") 

    def evaluate_train(self, xin, uin, xdot):        
        if self.type == 'Baseline':
            return [self.train_model.evaluate([xin[:,2:],uin], xdot)]
        elif self.type == 'Ensemble':
            return [m.evaluate([xin[:,2:],uin], xdot) for m in  self.train_models]

        elif self.type == 'BNN':
            n_sample = self.n_sample
            return [self.train_BNN.evaluate([xin[:,2:],uin], xdot) for _ in  range(n_sample)]

    #TODO
    def evaluate_predict(self, xin, Fx):        
        return [m.evaluate(xin[:,2:], Fx) for m in  self.predict_models]

    def predict_mean_Fx_gradient(self, xin):      
        if self.type == 'Baseline' or self.type == 'Res':
            res = self.take_gradient(xin)
            return np.mean(res, axis = 0)
        elif self.type == 'Ensemble'  or self.type == 'Ancor':
            res = self.take_gradient_ens(xin)
            return np.mean(res, axis = 0)
        elif self.type == 'SWAG':
            res = self.take_gradient(xin)
            return np.mean(res, axis = 0)     
        elif self.type == 'LA':
            res = self.take_gradient(xin)
            return np.mean(res, axis = 0)
        elif self.type == 'MC-D':
            res = self.take_gradient_mc(xin)
            return np.mean(res, axis = 0)
                
        else:
            raise Exception("Model type is not acceptable") 
        
    def take_gradient(self, xin):
        x = tf.Variable(xin[:,2:])   
        self.n_ens = 1             

        grad_res = np.zeros((self.n_ens, x.shape[0], self.n_out,self.n_x))
        
        for i in range(self.n_out):
            with tf.GradientTape() as t:
                t.reset()
                z = self.model_mu[1] (x)[:,i]

            g = t.gradient(z, x)
            grad_res[0,:,i,2:] = g.numpy()        
        
        return grad_res
        
    def take_gradient_mc(self, xin):
        x = tf.Variable(xin[:,2:])   
        self.n_ens = 1     
    
        grad_res = np.zeros((self.n_ens, x.shape[0], self.n_out,self.n_x))
        
        for i in range(self.n_out):
            with tf.GradientTape() as t:
                t.reset()
        
                z = self.model_mu[1] (x, training = True)[:,i]

            g = t.gradient(z, x)
            grad_res[0,:,i,2:] = g.numpy()        
        return grad_res
        
    def take_gradient_ens(self, xin):        
        x = tf.Variable(xin[:,2:])        
        grad_res = np.zeros((self.n_ens, x.shape[0], self.n_out,self.n_x))
        
        for n in range(self.n_ens):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                outputs = self.models_mu[n][1](x)

                # Compute the gradient of the outputs with respect to the inputs for each instance
                gradients = [tape.gradient(outputs[:,i], x) for i in range(self.n_out)]
            grad_res[n,0,:,2:] = tf.stack(gradients)[:,0,:].numpy()
        return grad_res

    def take_gradient_bnn(self, xin):
        x = tf.Variable(xin[:,2:])        
        grad_res = np.zeros((x.shape[0], self.n_out, self.n_x))

        for i in range(self.n_out):
            with tf.GradientTape() as t:
                t.reset()
                # t.watch(x)
                z = self.predict_BNN(x)[:,i]

            g = t.gradient(z, x)
            grad_res[:,i,2:] = g.numpy()        
        return grad_res

    def take_gradient_(self, xin):
        x = tf.Variable(xin[:,2:])        
        grad_res = np.zeros((self.n_ens, x.shape[0], self.n_out,4))
        for n in range(self.n_ens):
            with tf.GradientTape() as t:
                t.reset()
                z = self.predict_models[n](x)            
            grad_res[n,:,:,2:] = np.array([d[:,i,:] for i,d in enumerate(t.jacobian(z,x).numpy())])
        return grad_res    
    
    def get_CBC(self, CF, xt : np.ndarray, obstacle):
        if self.type == 'Baseline' or self.type == 'Res':
            n_ens = 5
            F   = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient(np.array([xt]))
            
            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, _Fx in enumerate(F):
                CBC2[i,:] = CF.CBC2(xt, obstacle, _Fx, JF)
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]
            
        elif self.type == 'Ensemble'  or self.type == 'Ancor':
            n_ens = self.n_ens
            F  = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient_ens(np.array([xt]))
            
            
            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, (_Fx, _JF) in enumerate(zip(F, JF)):
                CBC2[i,:] = CF.CBC2(xt, obstacle, _Fx, _JF) #JF[0])
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print("CBCsigma has is nan: F , JF, CBC2Sigma ==")
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]

        elif self.type == 'SWAG':
            n_ens = self.swag['S']

            F   = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient(np.array([xt]))

            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, _Fx in enumerate(F):
                CBC2[i,:] = CF.CBC2(xt, obstacle, _Fx, JF)
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]
        
        elif self.type == 'LA':
            n_ens = self.LA['S']

            F   = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient(np.array([xt]))

            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, _Fx in enumerate(F):
                CBC2[i,:] = CF.CBC2(xt, obstacle, _Fx, JF)
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]
        
        elif self.type == 'MC-D':
            n_ens = self.n_ens

            F   = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient_mc(np.array([xt]))

            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, _Fx in enumerate(F):
                CBC2[i,:] = CF.CBC2(xt, obstacle, _Fx, JF)
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]
        
        raise 'need implementation'
        raise Exception("Model type is not acceptable") 

    def get_CBC_out(self, CF, xt : np.ndarray, obstacle):
        if self.type == 'Baseline' or self.type == 'Res':
            # assert nSample > 0
            n_ens = 5
            F   = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient(np.array([xt]))
            
            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, _Fx in enumerate(F):
                CBC2[i,:] = CF.CBC2_out(xt, obstacle, _Fx, JF)
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]

        elif self.type == 'Ensemble'  or self.type == 'Ancor':
            n_ens = self.n_ens
            F  = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient_ens(np.array([xt]))
            
            
            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, (_Fx, _JF) in enumerate(zip(F, JF)):
                CBC2[i,:] = CF.CBC2_out(xt, obstacle, _Fx, _JF) #JF[0])
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print("CBCsigma has is nan: F , JF, CBC2Sigma ==")
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]

        elif self.type == 'SWAG':
            n_ens = self.swag['S']

            F   = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient(np.array([xt]))

            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, _Fx in enumerate(F):
                CBC2[i,:] = CF.CBC2_out(xt, obstacle, _Fx, JF)
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]
        
        elif self.type == 'LA':
            n_ens = self.LA['S']

            F   = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient(np.array([xt]))

            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, _Fx in enumerate(F):
                CBC2[i,:] = CF.CBC2_out(xt, obstacle, _Fx, JF)
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]
        
        elif self.type == 'MC-D':
            n_ens = self.n_ens

            F   = self.predictFx(np.array([xt]), n_ens)[0,:,:]
            JF  = self.take_gradient_mc(np.array([xt]))

            CBC2 = np.zeros((n_ens, 3)) # CBC has 3 elements
            for i, _Fx in enumerate(F):
                CBC2[i,:] = CF.CBC2_out(xt, obstacle, _Fx, JF)
            CBC2mean  = CBC2.sum(axis=0) / n_ens
            CBC2D     = CBC2- CBC2mean
            CBC2sigma = CBC2D.T @ CBC2D / n_ens

            if np.isnan(CBC2sigma).any():
                print(F)
                print(JF)
                print(CBC2sigma)
                
            return [CBC2mean, CBC2sigma]
        
        raise 'need implementation'
        raise Exception("Model type is not acceptable") 

