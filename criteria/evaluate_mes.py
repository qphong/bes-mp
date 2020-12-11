import tensorflow as tf 
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

def mes(x, 
        ls, sigmas, sigma0s, 
        Xsamples, Ysamples, 

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
    mes = tf.constant(0.0, dtype=dtype)
    norm = tf.distributions.Normal(
                loc=tf.constant(0., dtype=dtype), 
                scale=tf.constant(1., dtype=dtype))

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        
        invK = invKs[i,...]

        f_mean, f_var = utils.compute_mean_var_f(x, Xsamples, Ysamples, l, sigma, sigma0, invK, dtype=dtype)
        f_std = tf.sqrt(f_var)

        f_mean = tf.reshape(f_mean, shape=(1, nx))
        f_std = tf.reshape(f_std, shape=(1,nx))

        ent_f = utils.evaluate_norm_entropy(f_std, dtype=dtype)
        # (1,nx)

        ymax = tf.reshape(ymaxs[i,:], shape=(nmax,1))
        # (nmax, 1)

        standardized_ymax = (ymax - f_mean) / f_std 
        # (nmax, nx)

        logcdf_ymax = norm.log_cdf(standardized_ymax)
        cdf_ymax = tf.exp(logcdf_ymax)

        cond_ent_f = tf.reduce_mean( tf.constant(0.5, dtype=dtype) 
                            + tf.log( tf.cast(tf.sqrt(2.0 * np.pi), dtype=dtype) * f_std) 
                            + logcdf_ymax 
                            - standardized_ymax * norm.prob(standardized_ymax) / 2.0 / cdf_ymax,
                            axis=0)
        # (nmax,)

        mes = mes + tf.squeeze(ent_f - cond_ent_f) / tf.constant(n_hyp, dtype=dtype)

    return mes 
