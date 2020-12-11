# DARE strategy 
# taken from Decentralized Active Robotic Exploration and Mapping for
# Probabilistic Field Classification in Environmental Sensing
# Kian Hsiang Low, Jie Chen, John M. Dolan, Steve Chien, and David R. Thompson
# AAMAS 2012
# minimizing: abs(threshold - mean(x)) / std(x)
import tensorflow as tf 
from tensorflow_probability import distributions as tfd
import numpy as np 
import time 


import utils 



def dare(x, 
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

    dare_val = tf.constant(0.0, dtype=dtype)

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
        f_var = tf.reshape(f_var, (nx,1))
        f_std = tf.sqrt(f_var)
        # (nx,1)

        dare_i = tf.abs(f_mean - ymax) / f_std
        # (nx, nmax)
        dare_i = - tf.reduce_mean(dare_i, axis=1)
        # (nx,)
        # negating as we are maximizing instead of minimizing as in DARE paper

        dare_val = dare_val + tf.squeeze(dare_i) / tf.constant(n_hyp, dtype=dtype)

    return dare_val


