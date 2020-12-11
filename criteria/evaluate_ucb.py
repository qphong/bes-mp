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
def ucb(x, 
    xdim, n_hyp, 
    Xsamples, Ysamples, 
    ls, sigmas, sigma0s, 
    invKs, 
    iter,
    dtype=tf.float32, 
    ntestpoint=1000):
    """
    X: n x xdim
    Y: n x 1
    ls: nh x xdim
    sigmas: nh x 1 signal variances
    sigma0s: nh x 1 noise variances
        where nh is the number of hyperparameters
    invKs: nh x n x n
    iter: the current iteration number
    """

    ucb_val = tf.constant(0.0, dtype=dtype)

    u = 3. + tf.log(tf.square(tf.cast(iter, dtype=dtype)) * np.pi**2 / 6.)
    
    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        
        invK = invKs[i,...]

        f_mean, f_var = utils.compute_mean_var_f(x, Xsamples, Ysamples, l, sigma, sigma0, invK, dtype=dtype)
        f_std = tf.sqrt(f_var)

        noise_mean = 0.0

        post_var = f_var + sigma0
        
        ucb = tf.sqrt(tf.cast(2. * (u + np.log(ntestpoint)),
                               dtype=dtype)
                       * post_var)

        ucb_val = ucb_val + tf.squeeze(f_mean + ucb)

    ucb_val = ucb_val / tf.constant(n_hyp, dtype=dtype)
    return ucb_val


"""
based on gpoptimization matlab package: http://econtal.perso.math.cnrs.fr/software/
u = opt.u + log(iter^2 * pi^2 / 6)
ucb = sqrt(2*(u + log(n)) * s2)
n: number of test points? how to get this number?
        maybe just set it to the maximum number of iterations
s2: posterior variance
"""
