# taken from Actively Learning Level-Sets of Composite Functions
# Brent Bryan, Jeff Schneider
# ICML 2008
# maximizing 1.96 var(x) - abs(mean(x) - threshold)
import tensorflow as tf 
from tensorflow_probability import distributions as tfd
import numpy as np 
import time 


import utils 



def straddle(x, 
    ls, sigmas, sigma0s, 
    X, Y, 

    xdim, n_hyp, 
    
    ymaxs, 
    invKs, 
    dtype=tf.float32):
    """
    X: n x xdim
    Y: n x 1
    ls: nh x xdim
    sigmas: nh x 1 signal variances
    sigma0s: nh x 1 noise variances
        where nh is the number of hyperparameters
    invKs: nh x n x n
    ymaxs: nh x n_maxs
    """

    nx = tf.shape(x)[0]
    nmax = tf.shape(ymaxs)[1]

    straddle_val = tf.constant(0.0, dtype=dtype)

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        
        invK = invKs[i,...]
        ymax = tf.reshape(ymaxs[i,:], (1,nmax))
        # (1,nmax)

        f_mean, f_var = utils.compute_mean_var_f(x, 
                                X, Y, 
                                l, sigma, sigma0, 
                                invK, dtype=dtype)
        f_mean = tf.reshape(f_mean, (nx,1))
        f_var = tf.reshape(f_var, (nx,))

        straddle_i = 1.96 * f_var - tf.reduce_mean( tf.abs(f_mean - ymax), axis=1 )
        # (nx,)

        straddle_val = straddle_val + tf.squeeze(straddle_i) / tf.constant(n_hyp, dtype=dtype)

    return straddle_val
