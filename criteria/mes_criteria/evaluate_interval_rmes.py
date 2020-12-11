import tensorflow as tf 
from tensorflow_probability import distributions as tfd
import numpy as np 
import time 


import utils 



"""
provide:
    GP hypers: l, sigma, sigma0
               Xsamples, ysamples
    samples of ymax
    criterion: select next x
"""

def interval_rmes(x, 
    ls, sigmas, sigma0s, 
    X, Y, 

    xdim, n_hyp, 
    
    ymaxs, 
    invKs, 
    nsamples=100, 
    dtype=tf.float32):
    """
    a more efficient formula than evaluate_interval_rmes.py
        as it uses simplified formula
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

    normal_dist = tfd.Normal(loc=tf.zeros(nx, dtype=dtype), 
                            scale=tf.ones(nx, dtype=dtype))
    one_normal = tfd.Normal(loc=tf.constant(0., dtype=dtype), 
                            scale=tf.constant(1., dtype=dtype))

    rmes = tf.constant(0.0, dtype=dtype)

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        
        invK = invKs[i,...]
        ymax = ymaxs[i,:]
        # (nmax,)

        f_mean, f_var = utils.compute_mean_var_f(x, 
                                X, Y, 
                                l, sigma, sigma0, 
                                invK, dtype=dtype)
        f_mean = tf.reshape(f_mean, (nx,1))
        f_var = tf.reshape(f_var, (nx,1))
        # f_mean.shape = nx x 1
        # f_var.shape = nx x 1

        y_var = f_var + sigma0
        y_std = tf.sqrt(y_var)

        ysamples = normal_dist.sample(sample_shape=(nsamples))
        # (nsamples, nx)

        ysamples = tf.transpose(ysamples) * y_std + f_mean
        # (nx, nsamples)
        ysamples = tf.expand_dims(ysamples, axis=0)
        # (1, nx, nsamples)
        
        # function g shape:
        # (nmax, nx, nsamples)
        ymax = tf.expand_dims(
                tf.expand_dims(ymax, axis=1),
                axis=1)
        # (nmax, 1, 1)

        f_mean = tf.expand_dims(f_mean, axis=0)
        # (1, nx, 1)
        f_var = tf.expand_dims(f_var, axis=0)
        # (1, nx, 1)
        y_var = tf.expand_dims(y_var, axis=0)
        # (1, nx, 1)
        y_std = tf.expand_dims(y_std, axis=0)
        f_std = tf.sqrt(f_var)

        log_upper_cdf = one_normal.log_cdf(
            (ymax - f_mean) / f_std
        )
        log_lower_cdf = one_normal.log_cdf(
            - (ymax - f_mean) / f_std
        )
        # (nmax, nx, 1)
        
        log_upper_g = one_normal.log_cdf(
            (y_var * ymax - sigma0 * f_mean - f_var * ysamples)
            / (tf.sqrt(sigma0) * f_std * y_std)
        ) 

        log_upper_g = tf.cast(log_upper_g, dtype=dtype)
        log_upper_w = log_upper_g - log_upper_cdf
        # (nmax,  nx, nsamples)

        log_lower_g = one_normal.log_cdf(
            -(y_var * ymax - sigma0 * f_mean - f_var * ysamples)
            / (tf.sqrt(sigma0) * f_std * y_std)
        )

        log_lower_g = tf.cast(log_lower_g, dtype=dtype)
        log_lower_w = log_lower_g - log_lower_cdf
        # (nmax, nx, nsamples)

        log_g = tf.stack([log_upper_g, log_lower_g])
        log_w = tf.stack([log_upper_w, log_lower_w])
        # (2, nmax, nx, nsamples)

        rmes_i = tf.reduce_mean( 
                tf.reduce_mean(
                    tf.reduce_sum(tf.exp(log_g) * log_w, axis=0), # average over upper, lower
                    axis = 0), # average over nmax
                axis = 1) # average over nsample
        # (nx,)

        rmes = rmes + tf.squeeze(rmes_i) / tf.constant(n_hyp, dtype=dtype)

    return rmes

