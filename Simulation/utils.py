import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from statistics import NormalDist
from scipy.stats import norm

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)    
    tf.config.experimental.enable_op_determinism()

### confidence metric
def conf(y,my,sy, dp = 0.1):
    """
    example:
    bin_v,conf_v,a = conf(y, my, sy, dp = 0.01)
    """
    conf = []
    ps = list(np.arange(0.0,1.0, dp)) + [1.0 - 4.6525e-4]
    ds = [NormalDist(mu=0.0, sigma=1.0).inv_cdf(0.5+p/2) for p in ps]
    y, my, sy = np.array(y), np.array(my), np.array(sy)
    n = my.shape[0]

    for d in ds:
        miny = my - d * sy
        maxy = my + d * sy
        
        _conf = np.sum(np.all( np.bitwise_and(miny <= y , y <= maxy), axis = -1)) / n        
        conf.append( _conf)

    s = 0.0
    ### Area Error, Calibration
    # s = sum([abs( (ps[i] + ps[i+1] - conf[i] - conf[i+1]) / 2 * dp) for i, (b,c) in enumerate(zip(ps[:-1], conf[:-1]))])
    for i, (b,c) in enumerate(zip(ps[:-1], conf[:-1])):
        s += abs( (ps[i] + ps[i+1] - conf[i] - conf[i+1]) / 2 * dp)

    return np.array(ps), np.array(conf), s

def Uncertainty_metrics(y, my, sy, _bin = None, _conf = None, _area = 0.0, dp = 0.1):
    """
    Example : 
    _bin,_conf,_area, MLP, RL2E, RMSCE = Uncertainty_metrics(_yin, _my, _sy)

    """
    if _bin is None:
        _bin,_conf,_area = conf(y, my, sy, dp = dp)    
    
    # _bin_v,_conf_v,_area = conf(_yin, _my, _sy, dp = 0.1)
    # RMSCE = np.sqrt(np.sum((np.array(_bin_v) - np.array(_conf_v)) ** 2)/ (len(_bin_v))) 
    sy += 1e-3
    MLP   = np.sum(norm.pdf(y, my, sy)) / np.prod(y.shape)        
    RL2E  = np.sqrt(np.sum((my  - y) ** 2) / np.sum(y **2 ))
    RMSCE = np.sqrt(np.mean((_bin[1:] - _conf[1:]) ** 2))
    return _bin,_conf,_area, MLP, RL2E, RMSCE

### Create synthetic data
def f(nsample=1000, xlim = [-3.0, 3.0], nlim = [-0.5, 0.5]):
    x = np.random.uniform(xlim[0],xlim[1], nsample) 
    y = (x ** 3 ) / 5.0 - 1.0 *  x + np.random.uniform(nlim[0],nlim[1], nsample)    
    return x, y

def plot_graph(FxEstimator, conf, x, y, x_train = None,y_train = None, my = None, sy = None):
    """
    fig, (ax1,ax2) = plot_graph(LA, conf, x, y, x_train, y_train)
    fig.suptitle('LA network (fine_tune, burning 5)');
    """
    if FxEstimator == None:
        pass
    else:
        my,sy = FxEstimator.predictPos(x)
    # my, sy = my.numpy()[:,0], sy.numpy()[:,0]
    con = conf(y.numpy().ravel(), my, sy)
    # print(con)
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(8,3)
    ax1.scatter(x,y, s = 2, label = 'True')
    if x_train == None:
        pass
    else:
        ax1.scatter(x_train,y_train,s = 2, color='red', label = 'train')

    indx = np.argsort(x, axis = 0).ravel()
    ax1.fill_between(x.numpy()[indx][:,0] , my[indx] -3*sy[indx],my[indx]+3*sy[indx],alpha = 0.2,color='blue',  label = '3 SD')
    ax1.fill_between(x.numpy()[indx][:,0] , my[indx] -2*sy[indx],my[indx]+2*sy[indx],alpha = 0.5,color='blue',  label = '2 SD')
    ax1.fill_between(x.numpy()[indx][:,0] , my[indx] -1*sy[indx],my[indx]+1*sy[indx],alpha = 0.8,color='blue',  label = '1 SD')
    ax1.plot(x.numpy()[indx][:,0],my[indx], label = 'Mean')
    ax1.legend()

    ### Conf for updated weights
    ax2.scatter(con[0],con[1])
    ax2.plot([0,1.0],[0,1.0])
    ax2.set_xlabel('Confidence(%)')
    ax2.set_ylabel('Accuracy')
    return fig, (ax1,ax2)


def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory
