"""
multi-BES
a threshold is defined as a vector [b0, b1, b2, ..., b_{k+1}]
where b0      = -infty,
      b_{k+1} = infty,
      and b_i < b_{i+1}
"""


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

def mnes(x, 
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
    ymaxs: nh x n_maxs x 2
        k+2 elements of ymax are to define
        the k+1 intervals (including -infty->, and ->infty)
        nbound = k+2
    """

    nx = tf.shape(x)[0]
    # nmax = tf.shape(ymaxs)[1]
    nbound = tf.shape(ymaxs)[2] # excluding -infty and infty

    normal_dist = tfd.Normal(loc=tf.zeros(nx, dtype=dtype), 
                            scale=tf.ones(nx, dtype=dtype))
    one_normal = tfd.Normal(loc=tf.constant(0., dtype=dtype), 
                            scale=tf.constant(1., dtype=dtype))

    mnes_val = tf.constant(0.0, dtype=dtype)

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        
        invK = invKs[i,...]
        ymax = ymaxs[i,:]
        # (nmax,nbound)

        f_mean, f_var = utils.compute_mean_var_f(x, 
                                X, Y, 
                                l, sigma, sigma0, 
                                invK, dtype=dtype)
        f_mean = tf.reshape(f_mean, (nx,1))
        f_var = tf.reshape(f_var, (nx,1))
        f_std = tf.sqrt(f_var)
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
        
        """
        (nmax, nx, nsamples, nbound)
        """

        ysamples = tf.expand_dims(ysamples, axis=3)
        # (1, nx, nsamples, 1)

        ymax = tf.expand_dims(
                tf.expand_dims(ymax, axis=1),
                axis=1)
        # (nmax, 1, 1, nbound)

        f_mean = tf.reshape(f_mean, shape=(1, nx, 1, 1))
        f_var  = tf.reshape(f_var, shape=(1, nx, 1, 1))
        f_std  = tf.reshape(f_std, shape=(1, nx, 1, 1))
        y_var  = tf.reshape(y_var, shape=(1, nx, 1, 1)) 
        y_std  = tf.reshape(y_std, shape=(1, nx, 1, 1))


        # including the 2 ending segments
        # [-infty,a0], [ak,infty]
        interval_logprob_given_yD = tf.concat([
            one_normal.log_cdf(
                                (ymax[:,:,:,0:1] - f_mean) / f_std
                            ),
            one_normal.log_cdf(
                                -(ymax[:,:,:,1:] - f_mean) / f_std
                            )
            ], axis=3)
        # (nmax, nx, 1, nbound)

        # including the 2 ending segments
        # [-infty,a0], [ak,infty]
        interval_logprob_given_yD_yx = tf.concat([
            one_normal.log_cdf(
            (y_var * ymax[:,:,:,0:1] - sigma0 * f_mean - f_var * ysamples)
            / (tf.sqrt(sigma0) * f_std * y_std)
        ),
            one_normal.log_cdf(
            -(y_var * ymax[:,:,:,1:] - sigma0 * f_mean - f_var * ysamples)
            / (tf.sqrt(sigma0) * f_std * y_std)
        )
            ], axis=3)
        # (nmax, nx, nsamples, nbound)


        # including the middle segment
        interval_prob_given_yD_mid = one_normal.cdf(
                                (ymax[:,:,:,1:] - f_mean) / f_std
                            ) - one_normal.cdf(
                                (ymax[:,:,:,0:1] - f_mean) / f_std
                            )
        interval_prob_given_yD_mid = tf.clip_by_value(
                        interval_prob_given_yD_mid,
                        clip_value_min=1e-300,
                        clip_value_max = 1.0)

        interval_logprob_given_yD_mid = tf.log(interval_prob_given_yD_mid)        

        interval_prob_given_yD_yx_mid = \
                one_normal.cdf(
                    (y_var * ymax[:,:,:,1:] - sigma0 * f_mean - f_var * ysamples)
                    / (tf.sqrt(sigma0) * f_std * y_std)
                ) - one_normal.cdf(
                    (y_var * ymax[:,:,:,0:1] - sigma0 * f_mean - f_var * ysamples)
                    / (tf.sqrt(sigma0) * f_std * y_std)
                )
        interval_prob_given_yD_yx_mid = tf.clip_by_value(
                interval_prob_given_yD_yx_mid,
                clip_value_min=1e-300,
                clip_value_max=1.0)
        interval_logprob_given_yD_yx_mid = tf.log(interval_prob_given_yD_yx_mid)

        interval_logprob_given_yD = tf.concat([
            interval_logprob_given_yD,
            interval_logprob_given_yD_mid
            ], axis=3)

        interval_logprob_given_yD_yx = tf.concat([
            interval_logprob_given_yD_yx,
            interval_logprob_given_yD_yx_mid
            ], axis=3)

        log_w = interval_logprob_given_yD_yx - interval_logprob_given_yD
        # (nmax, nx, nsamples, nbound)

        # tmp = tf.where(
        #         interval_logprob_given_yD_yx < -1e200,
        #         tf.zeros_like(log_w, dtype=dtype), 
        #         tf.exp(interval_logprob_given_yD_yx) * log_w)

        tmp = tf.exp(interval_logprob_given_yD_yx) * log_w

        mnes_i = tf.reduce_mean(
                tf.reduce_mean(
                    tf.reduce_sum(tmp, axis=3), # average over interval index
                    axis=0), # average over nmax
                axis = 1) # average over nsamples
        # (nx,)

        mnes_val = mnes_val + tf.squeeze(mnes_i) / tf.constant(n_hyp, dtype=dtype)

    return mnes_val


