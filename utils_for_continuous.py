
import numpy as np 
import tensorflow as tf 
import scipy as sp 
import time 
import scipy.stats as spst
import utils 
import optfunc
import functions


def func_with_2d_input(func, x, xdim):
    x = tf.reshape(x, shape=(-1,xdim))
    return func(x)


def optimize_continuous_function(xdim, func,
                candidate_xs, top_init_k, 
                parallel_iterations=5,
                xmin=-np.infty,
                xmax=np.infty,
                dtype=tf.float32,
                name='optimize_cont_func',
                debug=False,
                multiple_func=None):
    # requires: func's input is a vector of 1d
    #           func's output is a scalar
    #   func can receive a vector of 2d
    #        returns a 1d output
    rand_candidate_xs = tf.random.uniform(shape=(top_init_k,xdim), 
                        minval=xmin, maxval=xmax, dtype=dtype)
    candidate_xs = tf.concat([candidate_xs, rand_candidate_xs], axis=0)


    ncandidate = tf.shape(candidate_xs)[0]

    candidate_fs_tfarr = tf.TensorArray(dtype=dtype, size=ncandidate)
    
    _, candidate_fs_tfarr = tf.while_loop(
        cond = lambda i,_: i < ncandidate,
        body = lambda i, candidate_fs_tfarr: (
                        i+1, 
                        candidate_fs_tfarr.write(i, func(candidate_xs[i,:]))
                ),
        loop_vars = (0, candidate_fs_tfarr),
        parallel_iterations = parallel_iterations
    )

    candidate_fs = tf.squeeze(candidate_fs_tfarr.stack())

    _, top_k_idxs = tf.math.top_k(candidate_fs, 
                k = top_init_k)

    top_init_xs = tf.gather(candidate_xs, top_k_idxs)

    xs = tf.get_variable(shape=(top_init_k, xdim), 
                dtype=dtype, 
                name='{}_xs'.format(name),
                constraint=lambda x: tf.clip_by_value(x, xmin, xmax))
    assign = tf.assign(xs, top_init_xs)

    if multiple_func is None:
        fs = func(xs)
    else:
        fs = multiple_func(xs)

    train = tf.train.AdamOptimizer().minimize(
                        -tf.reduce_sum(fs), 
                        var_list=[xs])

    max_idx = tf.argmax( tf.squeeze(fs) )

    maximizer = tf.gather(xs, max_idx)
    maximizer = tf.reshape(maximizer, shape=(1,xdim))

    maximum = tf.gather(fs, max_idx)

    if debug:
        return assign, train, maximizer, maximum, xs, fs
    
    return assign, train, maximizer, maximum


def sample_function(
        xdim, n_hyp, 
        n_funcs, n_features, 
        ls, sigmas, sigma0s, 
        # (nhyp, xdim), (nhyp,), (nhyp,)
        Xsamples, # (nobs, xdim)
        Ysamples, # (nobs, 1)
        dtype=tf.float32):
    thetas_all = []
    Ws_all = []
    bs_all = []

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]

        thetas, Ws, bs = optfunc.draw_random_init_weights_features(
                xdim, n_funcs, n_features, 
                
                Xsamples, 
                Ysamples, 
                
                l, sigma, sigma0, 
                
                dtype=dtype, 
                name='random_features_{}'.format(i))
        # thetas.shape = (n_funcs, n_features, 1)
        # Ws.shape = (n_funcs, n_features, xdim)
        # bs.shape = (n_funcs, n_features, 1)

        thetas_all.append(thetas)
        Ws_all.append(Ws)
        bs_all.append(bs)

        thetas_all = tf.stack(thetas_all)
        Ws_all = tf.stack(Ws_all)
        bs_all = tf.stack(bs_all)

    return thetas_all, Ws_all, bs_all
    # thetas_all.shape = (nhyp, n_funcs, n_features, 1)
    # Ws_all.shape = (nhyp, n_funcs, n_features, xdim)
    # bs_all.shape = (nhyp, n_funcs, n_features, 1)


def sample_xmaxs_fmaxs(
            xdim, n_hyp, n_funcs, n_features, 
            ls, sigmas, sigma0s, 
            thetas_all, # (nhyp, n_funcs, n_features, 1)
            Ws_all, # (nhyp, n_funcs, n_features, xdim)
            bs_all, # (nhyp, n_funcs, n_features, 1)
            initializers, # ( n_init, xdim)
            top_init_k, # scalar
            xmin, xmax, 
            dtype=tf.float32, 
            parallel_iterations = 5,
            get_xs = False,
            name='sample_maxs'):
    # Requires: n_init >= top_init_k
    assigns = []
    trains = []
    max_xs = []
    max_fs = []
    xs_all = []

    n_init = tf.shape(initializers)[0]

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]

        # Find the top k to initialize the optimization
        top_k_inits_tfarr = tf.TensorArray(dtype=dtype, size=n_funcs)
        
        def body(j, top_k_inits_tfarr):
            fvals = tf.squeeze( tf.sqrt(2.0 * sigma / n_features) \
                * tf.matmul(thetas_all[i,j,...],
                        tf.cos( tf.matmul(Ws_all[i,j,...], 
                                    initializers, 
                                    transpose_b=True) 
                                + bs_all[i,j,...] ), 
                        transpose_a=True) )
            fvals = tf.reshape(fvals, shape=(n_init,))
            # (n_init,)

            _, idxs = tf.math.top_k(fvals, k=top_init_k)
            # (top_init_k,)
            
            return j+1, top_k_inits_tfarr.write(j, 
                            tf.gather(initializers, idxs))
        
        _, top_k_inits_tfarr = tf.while_loop(
            cond = lambda j,_: j < n_funcs,
            body = body,
            loop_vars = (0, top_k_inits_tfarr),
            parallel_iterations = parallel_iterations
        )
        top_k_inits = top_k_inits_tfarr.stack()
        # (n_funcs, top_init_k, xdim)

        # Optimize function samples
        name = '{}_{}'.format(name, i)

        xs = tf.get_variable(shape=(n_funcs, top_init_k, xdim), 
                             dtype=dtype, 
                             name='{}_xs'.format(name),
                             constraint=lambda x: tf.clip_by_value(x, xmin, xmax))
        assign = tf.assign(xs, top_k_inits)

        fvals = tf.reshape(tf.sqrt(2.0 * sigma / n_features) \
                    * tf.matmul(thetas_all[i,...],
                            tf.cos( tf.matmul(Ws_all[i,...], 
                                xs, 
                                transpose_b=True) 
                                + bs_all[i,...] ), 
                            transpose_a=True), 
                shape=(n_funcs, top_init_k))
        # (n_funcs, top_init_k)

        sum_fvals = tf.reduce_sum(fvals)
        # scalar

        train = tf.train.AdamOptimizer().minimize(-sum_fvals, var_list=[xs])

        idx = tf.argmax(fvals, axis=1)
        # (n_funcs,)

        max_x = []
        for j in range(n_funcs):
            max_x.append(tf.gather(xs[j,...], idx[j]))
        max_x = tf.stack(max_x)
        # (n_funcs, xdim)

        max_f = tf.reduce_max(fvals, axis=1)
        # (n_funcs,)

        assigns.append(assign)
        trains.append(train)
        max_xs.append(max_x)
        max_fs.append(max_f)
        
        if get_xs:
            xs_all.append(xs)

    max_xs = tf.stack(max_xs)
    # (nhyp, n_funcs, xdim)
    max_fs = tf.stack(max_fs)
    # (nhyp, n_funcs)

    if get_xs:
        xs_all = tf.stack(xs_all)
        # (nhyp, n_funcs, top_init_k, xdim)
        return assigns, trains, max_xs, max_fs, xs_all, top_k_inits

    return assigns, trains, max_xs, max_fs
    # list of nhyp elements


# for debugging
def get_function_samples(xs, # (nx, xdim)
        xdim, n_hyp, n_funcs, n_features, 
        ls, sigmas, sigma0s, 
        thetas_all, # (nhyp, n_funcs, n_features, 1)
        Ws_all, # (nhyp, n_funcs, n_features, xdim)
        bs_all, # (nhyp, n_funcs, n_features, 1)
        dtype=tf.float32):

    fvals_all = []

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]

        # Find the top k to initialize the optimization
        top_k_inits = []

        fvals = tf.reshape(tf.sqrt(2.0 * sigma / n_features) \
                    * tf.matmul(thetas_all[i,...],
                        tf.cos( tf.matmul(Ws_all[i,...], 
                                    tf.tile(
                                        tf.expand_dims(
                                            xs, 
                                            axis=0), 
                                        multiples=(n_funcs,1,1)
                                    ),
                                    transpose_b=True) 
                                + bs_all[i,...] ), 
                        transpose_a=True), 
                    shape=(n_funcs, -1))
        # (n_funcs, nx)

        fvals_all.append(fvals)

    return fvals_all
    # (nhyp, n_funcs, nx)
