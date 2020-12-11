import tensorflow as tf 
import numpy as np 
import scipy as sp 
import time 
import scipy.stats as spst
import sys 
import utils

import matplotlib.pyplot as plt 



# draw random features, and their weights
def draw_random_init_weights_features(
            xdim, n_funcs, n_features, 
            
            xx, # (nobs, xdim)
            yy, # (nobs, 1)
            
            l, sigma, sigma0, 
            # (1,xdim), (), ()
            
            dtype=tf.float32, 
            name='random_features'):
    """
    sigma, sigma0: scalars
    l: 1 x xdim
    xx: n x xdim
    yy: n x 1
    n_features: a scalar
    different from draw_random_weights_features,
        this function set W, b, noise as Variable that is initialized randomly
        rather than sample W, b, noise from random function
    """

    n = tf.shape(xx)[0]

    xx = tf.tile( tf.expand_dims(xx, axis=0), multiples=(n_funcs,1,1) )
    yy = tf.tile( tf.expand_dims(yy, axis=0), multiples=(n_funcs,1,1) )
    idn = tf.tile(tf.expand_dims(tf.eye(n, dtype=dtype), axis=0), multiples=(n_funcs,1,1))

    # draw weights for the random features.
    W = tf.get_variable(name="{}_W".format(name), 
                shape=(n_funcs, n_features,xdim), 
                dtype=dtype, 
                initializer=tf.random_normal_initializer()) \
        * tf.tile( tf.expand_dims(tf.sqrt(l), axis=0), 
                   multiples=(n_funcs,n_features,1) )
    # n_funcs x n_features x xdim

    b = 2.0 * np.pi \
        * tf.get_variable(
            name="{}_b".format(name), 
            shape=(n_funcs,n_features,1), 
            dtype=dtype, 
            initializer=tf.random_uniform_initializer(minval=0., maxval=1.))
    # n_funcs x n_features x 1

    # compute the features for xx.
    Z = tf.cast(tf.sqrt(2.0 * sigma / n_features), dtype=dtype)\
        * tf.cos( tf.matmul(W, xx, transpose_b=True)
                + tf.tile(b, multiples=(1,1,n) ))
    # n_funcs x n_features x n

    # draw the coefficient theta.
    noise = tf.get_variable(
                name="{}_noise".format(name), 
                shape=(n_funcs,n_features,1), 
                dtype=dtype, 
                initializer=tf.random_normal_initializer())
    # n_funcs x n_features x 1

    def true_clause():
        Sigma = tf.matmul(Z, Z, transpose_a=True) + sigma0 * idn
        # n_funcs x n x n of rank n or n_features

        mu = tf.matmul(tf.matmul(Z, utils.multichol2inv(Sigma, n_funcs, dtype=dtype)), yy)
        # n_funcs x n_features x 1

        e, v = tf.linalg.eigh(Sigma)
        e = tf.expand_dims(e, axis=-1)
        # n_funcs x n x 1

        r = tf.reciprocal(tf.sqrt(e) * (tf.sqrt(e) + tf.sqrt(sigma0)))
        # n_funcs x n x 1

        theta = noise \
            - tf.matmul(Z, 
                        tf.matmul(v, 
                                r * tf.matmul(v, 
                                                tf.matmul(Z, noise, transpose_a=True), 
                                                transpose_a=True))) \
            + mu
        # n_funcs x n_features x 1

        return theta 


    def false_clause():
        Sigma = utils.multichol2inv( tf.matmul(Z, Z, transpose_b=True) / sigma0 
                        + tf.tile(tf.expand_dims(tf.eye(n_features, dtype=dtype), axis=0), multiples=(n_funcs,1,1)), 
                        n_funcs, dtype=dtype)

        mu = tf.matmul(tf.matmul(Sigma, Z), yy) / sigma0

        theta = mu + tf.matmul(tf.cholesky(Sigma), noise)
        return theta


    theta = tf.cond(
                pred=tf.less(n, n_features),
                true_fn=true_clause,
                false_fn=false_clause
            )

    return theta, W, b


def make_function_sample(x, n_features, sigma, theta, W, b, dtype=tf.float32):
    fval = tf.squeeze( tf.sqrt(2.0 * sigma / n_features) \
                * tf.matmul(theta,
                            tf.cos( tf.matmul(W, 
                                              x, 
                                              transpose_b=True) 
                                    + b ), 
                            transpose_a=True) )
    return fval


def duplicate_function_with_multiple_inputs(f, n_inits, xmin=-np.infty, xmax=np.infty, dtype=tf.float32):

    xs_list = [None] * n_inits
    fvals = [None] * n_inits

    for i in range(n_inits):
        xs_list[i] = tf.get_variable(shape=(1,xdim), dtype=dtype, name='{}_{}'.format(name, i),
                                constraint=lambda x: tf.clip_by_value(x, xmin, xmax))
        fvals[i] = f(xs_list[i])

    fvals = tf.squeeze(tf.stack(fvals))
    xs = tf.stack(xs_list)
    return xs, xs_list, fvals


# find maximum of a function with multiple initializers
# a function is a tensor, so this function can be used in the above function
def find_maximum_with_multiple_init_tensor(xs_list, fvals, n_inits, xdim, optimizer, dtype=tf.float32):
    """
    # xmin=-np.infty, xmax=np.infty,
    xs: list of size n_inits of (1,xdim)
    fvals: (n_inits,): function value with inputs are xs tensor
    initializers: n_inits x xdim
    """
    # initializers: n x d
    # func: a tensor function 
    #     input:  tensor n x d 
    #     output: tensor n x 1
    # n_inits: scalar (not a tensor)
    """
    returns:
        vals: shape = (n_inits,)
        invars: shape = (n_inits,xdim)
        maxval: scalar
        maxinvar: shape= (xdim,)
    """

    trains = [None] * n_inits

    for i in range(n_inits):
        trains[i] = optimizer.minimize(-fvals[i], var_list=[xs_list[i]])

    max_idx = tf.argmax(fvals)
    return trains, max_idx


def find_maximum_list_of_funcs(xdim, n_inits, n_funcs, xs, xs_list, fvals, optimizer, dtype=tf.float32):
    """
    xs: shape=(n_funcs, n_inits, xdim)
    xs_list: list of n_funcs lists of size n_inits of tensor (1,xdim)
    fvals: tensor of shape (n_funcs, n_inits)
    #initializers: (n_funcs, n_inits, xdim)
    """
    train_all = []
    max_val_all = [None] * n_funcs
    max_input_all = [None] * n_funcs
    max_idx_all = []

    for i in range(n_funcs):
        trains, max_idx = find_maximum_with_multiple_init_tensor(xs_list[i], fvals[i,...], n_inits, xdim, dtype=dtype, optimizer=optimizer)

        train_all.extend(trains)
        max_idx_all.append(max_idx)

        max_input_all[i] = xs[i,max_idx,...]
        max_val_all[i] = fvals[i,max_idx]

    max_val_arr = tf.reshape(tf.stack(max_val_all), shape=(n_funcs,))
    max_input_arr = tf.reshape(tf.stack(max_input_all), shape=(n_funcs,xdim))
    max_idx_arr = tf.reshape(tf.stack(max_idx_all), shape=(n_funcs,))

    return train_all, max_val_arr, max_input_arr, max_idx_arr


def gen_fval_xs(funcs, n_inits, xdim, xmin, xmax, dtype=tf.float32, name='test'):
    """
    if funcs is a list of functions
        return xs: nfuncs x n_inits x xdim
               xs_list: list of nfuncs lists of n_inits tensors of size (1,xdim)
               fvals: nfuncs x n_inits
    else:
        return xs: n_inits x xdim
               xs_list: list of n_inits tensors of size (1,xdim)
               fvals: n_inits,
    """
    if isinstance(funcs, list):
        print("List of functions")

        n_funcs = len(funcs)
        xs_list = [[tf.get_variable(shape=(1,xdim), dtype=dtype, name='{}_{}_{}'.format(name, i, j),
                                    constraint=lambda x: tf.clip_by_value(x, xmin, xmax)) for i in range(n_inits)] for j in range(n_funcs)]

        xs = []
        for i in range(n_funcs):
            xs.append( tf.stack(xs_list[i]) )
        xs = tf.stack(xs)

        fvals = []
        for i in range(n_funcs):
            fvals_i = []
            for j in range(n_inits):
                fvals_i.append( tf.squeeze(funcs[i](xs_list[i][j])) )

            fvals.append( tf.squeeze(tf.stack(fvals_i)) )

        fvals = tf.stack(fvals)

    else: # funcs is a function
        print("A function")
        xs_list = [tf.get_variable(shape=(1,xdim), dtype=dtype, name='test_func_mul_init_{}'.format(i),
                                    constraint=lambda x: tf.clip_by_value(x, xmin, xmax)) for i in range(n_inits)]

        fvals = [funcs(x) for x in xs_list]

        xs = tf.reshape(tf.concat(xs_list, axis=0), shape=(n_inits, xdim))
        fvals = tf.squeeze(tf.concat(fvals, axis=0))

    return xs, xs_list, fvals



# draw random features, and their weights
def draw_random_init_weights_features_np(
            xdim, n_funcs, n_features, 
            
            xx, # (nobs, xdim)
            yy, # (nobs, 1)
            
            l, sigma, sigma0):
            # (1,xdim), (), ()
    """
    sigma, sigma0: scalars
    l: 1 x xdim
    xx: n x xdim
    yy: n x 1
    n_features: a scalar
    different from draw_random_weights_features,
        this function set W, b, noise as Variable that is initialized randomly
        rather than sample W, b, noise from random function
    """
    n = xx.shape[0]
    l = l.reshape(1,xdim)
    
    xx = np.tile( xx.reshape(1,n,xdim), reps=(n_funcs,1,1) )
    yy = np.tile( yy.reshape(1,n,1), reps=(n_funcs,1,1) )
    idn = np.tile( np.eye(n).reshape(1,n,n), reps=(n_funcs,1,1) )

    # draw weights for the random features.
    W = np.random.randn(n_funcs, n_features, xdim) \
        * np.tile(np.sqrt(l).reshape(1,1,xdim), 
                  reps=(n_funcs, n_features, 1))
    # n_funcs x n_features x xdim

    b = 2.0 * np.pi * np.random.rand(n_funcs, n_features, 1)
    # n_funcs x n_features x 1

    # compute the features for xx.
    Z = np.sqrt(2.0 * sigma / n_features) \
        * np.cos( np.matmul(W, np.transpose(xx, (0,2,1)))
                  + np.tile(b, reps=(1,1,n)) )
    # n_funcs x n_features x n

    # draw the coefficient theta.
    noise = np.random.randn(n_funcs, n_features, 1)
    # n_funcs x n_features x 1

    if n < n_features:
        Sigma = np.matmul(np.transpose(Z, (0,2,1)), Z) \
                + sigma0 * idn
        # n_funcs x n x n

        while True:
            try:
                invSigma = np.linalg.inv(Sigma)
                break
            except:
                # non-invertible Sigma
                jitter = 1e-4
                print("optfunc:draw_random_init_weights_features_np: non-invertible Sigma. add jitter {}".format(jitter))
                Sigma = Sigma + jitter

        mu = np.matmul( np.matmul(Z, invSigma ), yy)
        # n_funcs x n_features x 1

        e, v = np.linalg.eig(Sigma)
        # n_funcs, n
        # n_funcs, n, n

        e = e.reshape(n_funcs, n, 1)
        # n_funcs x n x 1

        r = 1.0 / (np.sqrt(e) * (np.sqrt(e) + np.sqrt(sigma0)))
        # n_funcs x n x 1

        theta = noise \
                - np.matmul(Z, 
                            np.matmul(v, 
                                      r * np.matmul(np.transpose(v, (0,2,1)), 
                                                    np.matmul(np.transpose(Z,(0,2,1)), 
                                                              noise) 
                                                    )
                                     )
                            ) \
                + mu
        # n_funcs x n_features x 1
    else:
        Sigma = np.linalg.inv(
            np.matmul(Z, np.transpose(Z,(0,2,1))) / sigma0
            + np.tile( np.eye(n_features).reshape(1,n_features,n_features), reps=(n_funcs,1,1) )
        )
        mu = np.matmul( np.matmul(Sigma,Z), yy ) / sigma0

        theta = mu + np.matmul(np.linalg.cholesky(Sigma), noise)

    return theta, W, b


# for testing draw_random_init_weights_features_np
def make_function_sample_np(x, n_features, sigma, theta, W, b):
    fval = np.squeeze(
        np.sqrt(2.0 * sigma / n_features)
        * np.matmul(theta.T,
                    np.cos(
                        np.matmul(W, x.T)
                        + b
                        )
                    ) )
    # x must be a 2d tensor
    # return: n_funcs x tf.shape(x)[0]
    #      or (tf.shape(x)[0],) if n_funcs = 1

    return fval
