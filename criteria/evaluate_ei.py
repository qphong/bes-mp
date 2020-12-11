import tensorflow as tf 
import tensorflow_probability as tfp 
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
def ei(x, xdim, n_hyp, Xsamples, Ysamples, ls, sigmas, sigma0s, invKs, fmax, dtype=tf.float32):
    """
    X: n x xdim
    Y: n x 1
    ls: nh x xdim
    sigmas: nh x 1 signal variances
    sigma0s: nh x 1 noise variances
        where nh is the number of hyperparameters
    invKs: nh x n x n
    fmax could be the maximum samples (observations) so far
               or the maximum posterior mean
    """

    norm = tfp.distributions.Normal(loc=tf.constant(0., dtype=dtype), scale=tf.constant(1., dtype=dtype))

    ei_val = tf.constant(0.0, dtype=dtype)

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        
        invK = invKs[i,...]

        f_mean, f_var = utils.compute_mean_var_f(x, Xsamples, Ysamples, l, sigma, sigma0, invK, dtype=dtype)
        f_std = tf.sqrt(f_var)

        # consider the distribution of f, not of y
        diff = f_mean - fmax
        pos_diff = tf.clip_by_value(diff, clip_value_min=0.0, clip_value_max=np.infty)
        standard_diff = diff / f_std
        ei_val = ei_val + tf.squeeze( pos_diff + f_std * norm.prob(standard_diff) - tf.abs(diff) * norm.cdf(standard_diff) )

    ei_val = ei_val / tf.constant(n_hyp, dtype=dtype)
    return ei_val


