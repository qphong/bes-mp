import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
import scipy as sp 
import scipy.stats as spst
import sys

clip_min = 1e-100
print("clip_min = {}".format(clip_min))


def perturb(xs, duplicate_resolution, xmin, xmax):
    """
    xs: n x d
    if exist i,j dist(xs[i,:], xs[j,:]) <= duplicate_resolution
        perturb xs[i,:] such that dist(xs[i,:], xs[j,:]) > duplicate_resolution
            and xs[i,:] >= xmin and xs[i,:] <= xmax
    """
    n = xs.shape[0]
    d = xs.shape[1]

    is_duplicated = True

    while is_duplicated:

        is_duplicated = False 

        for i in range(n):
            for j in range(i+1,n):
                diff = xs[i,:] - xs[j,:]
                rms = np.sqrt(np.sum(diff * diff))

                while rms <= duplicate_resolution or np.any(xs[j,:] < xmin) or np.any(xs[j,:] > xmax):
                    # modify xs[j]
                    xs[j,:] = xs[i,:] + np.random.rand(1,d) * duplicate_resolution * 4.0 - duplicate_resolution * 2.0
                    
                    diff = xs[i,:] - xs[j,:]
                    rms = np.sqrt(np.sum(diff * diff))

                    is_duplicated = True

    return xs



def sqrtm(mat):
    # return tf.linalg.sqrtm(mat)
    # only valid for positive symmetric matrix
    s, u, _ = tf.svd(mat, full_matrices=True)
    return u * tf.sqrt(s) @ tf.transpose(u)

def get_uniform_random_vect(size, dim, xmin, xmax):
    xs = np.random.rand(size,dim) * (xmax - xmin) + xmin
    return xs


def evaluate_norm_entropy(s, dtype=tf.float32):
    return tf.cast(0.5 * tf.log(2*np.pi), dtype=dtype) + tf.log(s) + tf.constant(0.5, dtype=dtype)


def chol2inv(mat, dtype=tf.float32):

    n = tf.shape(mat)[0]

    _, mat, _ = tf.while_loop(
        cond = lambda i, mat, eigs: tf.reduce_min(eigs) < 1e-9,
        body = lambda i, mat, eigs: [0.0, 
                mat + tf.eye(n, dtype=dtype) * 1e-4, 
                tf.linalg.eigvalsh(mat + tf.eye(n, dtype=dtype) * 1e-4)],
        loop_vars = (0.0, mat, tf.linalg.eigvalsh(mat)) )

    # lower = tf.linalg.cholesky(mat)
    invlower = tf.matrix_solve(tf.linalg.cholesky(mat), 
                               tf.eye(n, dtype=dtype))
    invmat = tf.transpose(invlower) @ invlower
    return invmat


def multichol2inv(mat, n_mat, dtype=tf.float32):
    # lower = tf.linalg.cholesky(mat)
    invlower = tf.matrix_solve(tf.linalg.cholesky(mat), 
                               tf.tile(tf.expand_dims(tf.eye(tf.shape(mat)[1], dtype=dtype), 
                                                      axis=0), 
                                       multiples=(n_mat,1,1) ) )
    invmat = tf.matmul(invlower, invlower, transpose_a=True)
    return invmat    


def computeKnm(X, Xbar, l, sigma, dtype=tf.float32):
    """
    X: n x d
    l: d
    """
    n = tf.shape(X)[0]
    m = tf.shape(Xbar)[0]

    X = X * tf.sqrt(l)
    Xbar = Xbar * tf.sqrt(l)

    Q = tf.tile(tf.reduce_sum( tf.square(X), axis=1 , keepdims=True ), multiples=(1,m))
    Qbar = tf.tile(tf.transpose(tf.reduce_sum(tf.square(Xbar), axis=1, keepdims=True )), multiples=(n,1)) 

    dist = Qbar + Q - 2 * X @ tf.transpose(Xbar)
    knm = sigma * tf.exp( -0.5 * dist )
    return knm


def computeKmm(X, l, sigma, nd=2, dtype=tf.float32):
    """
    X: (...,n,d)
    nd = len(tf.shape(X))
    l: (1,d)
    sigma: signal variance
    return (...,n,n)
    """
    n = tf.shape(X)[-2]
    X = X * tf.sqrt( tf.reshape(l, shape=(1,-1)) )
    # (...,n,d)
    Q = tf.reduce_sum( tf.square(X), axis=-1, keepdims=True )
    # (...,n,1)

    transpose_idxs = np.array(list(range(nd)))
    transpose_idxs[-2] = nd-1
    transpose_idxs[-1] = nd-2

    dist = Q + tf.transpose(Q, perm=transpose_idxs) - 2 * X @ tf.transpose(X, perm=transpose_idxs)

    kmm = sigma * tf.exp(-0.5 * dist)

    return kmm


def computeNKmm(X, l, sigma, sigma0, dtype=tf.float32):
    """
    X: n x d
    l: 1 x d
    sigma: signal variance
    sigma0: noise variance
    """
    # cond = tf.less(sigma0, 1e-2)
    # tf.where(cond)
    """
    if sigma0 >= 1e-2:
        perturb = 1e-10
    else:
        perturb = 1e-4
    """
    # print("Add jitter for computeNKmm")
    # return computeKmm(X, l, sigma) + tf.eye(tf.shape(X)[0], dtype=dtype) * (sigma0 + sigma * tf.constant(1e-10, dtype=dtype))
    print("No jitter for computeNKmm")
    return computeKmm(X, l, sigma, dtype=dtype) + tf.eye(tf.shape(X)[0], dtype=dtype) * sigma0


def compute_mean_var_f(x, Xsamples, Ysamples, l, sigma, sigma0, 
                    NKInv=None, fullcov=False, dtype=tf.float32):
    """
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    Ysamples: m x 1
    Xsamples: m x d
    x: n x d

    return: mean: n x 1
            var : n x 1
    """
    if NKInv is None:
        NK = computeNKmm(Xsamples, l, sigma, sigma0, dtype=dtype)
        NKInv = chol2inv( NK, dtype=dtype )

    kstar = computeKnm(x, Xsamples, l, sigma, dtype=dtype)
    mean = tf.squeeze(kstar @ (NKInv @ Ysamples))

    if fullcov:
        Kx = computeKmm(x, l, sigma, dtype=dtype)
        var = Kx - kstar @ NKInv @ tf.transpose(kstar)
        diag_var = tf.linalg.diag_part(var)
        diag_var = tf.clip_by_value(diag_var, clip_value_min=clip_min, clip_value_max=np.infty)
        var = tf.linalg.set_diag(var, diag_var)
    else:
        var = sigma - tf.reduce_sum( (kstar @ NKInv) * kstar, axis=1 )
        var = tf.clip_by_value(var, clip_value_min=clip_min, clip_value_max=np.infty)

    return mean, var


def computeKmm_np(X, l, sigma):
    n = X.shape[0]
    xdim = X.shape[1]
    
    l = l.reshape(1,xdim)

    X = X * np.sqrt(l)

    Q = np.tile(
        np.sum( X * X, axis=1, keepdims=True ),
        reps=(1,n)
    )
    dist = Q + Q.T - 2 * X.dot(X.T)

    kmm = sigma * np.exp(-0.5 * dist)
    return kmm 


def computeKnm_np(X, Xbar, l, sigma):
    """
    X: n x d
    l: d
    """
    n = np.shape(X)[0]
    m = np.shape(Xbar)[0]
    xdim = np.shape(X)[1]

    l = l.reshape(1,xdim)
    
    X = X * np.sqrt(l)
    Xbar = Xbar * np.sqrt(l)

    Q = np.tile( 
        np.sum( X*X, axis=1, keepdims=True),
        reps = (1,m))
    Qbar = np.tile(
        np.sum( Xbar*Xbar, axis=1, keepdims=True).T,
        reps=(n,1))

    dist = Qbar + Q - 2 * X.dot(Xbar.T)
    knm = sigma * np.exp(-0.5 * dist)
    return knm


def compute_mean_f_np(x, Xsamples, Ysamples, l, sigma, sigma0):
    """
    x: n x xdim
    Xsample: m x xdim
    Ysamples: m x 1
    return mean: n x 1

    l: 1 x xdim
    sigma, sigma0: scalar
    """
    m = Xsamples.shape[0]
    xdim = Xsamples.shape[1]
    x = x.reshape(-1,xdim)
    n = x.shape[0]

    Ysamples = Ysamples.reshape(m,1)

    NKmm = computeKmm_np(Xsamples, l, sigma) + np.eye(m) * sigma0
    invNKmm = np.linalg.inv(NKmm)

    kstar = computeKnm_np(x, Xsamples, l, sigma)
    mean = kstar.dot(invNKmm.dot(Ysamples))

    return mean.reshape(n,)


def computeNKmm_multiple_data(nxs, Xsamples, xs, l, sigma, sigma0, dtype=tf.float32, inverted=False):
    """
    xs: shape = (nxs,xdim)
    compute covariance matrix of [Xsamples, x] for x in xs
        where Xsamples include noise
              x does not include noise
    return shape (nxs, n_data+1, n_data+1)
        where n_data = tf.shape(Xsamples)[0]
    """
    n_data = tf.shape(Xsamples)[0]
    noise_mat = tf.eye(n_data, dtype=dtype) * sigma0
    noise_mat = tf.pad(noise_mat, [[0,1], [0,1]], "CONSTANT")

    ret = []
    for i in range(nxs):
        X_concat = tf.concat([Xsamples, tf.expand_dims(xs[i,:],0) ], axis=0)
        NKmm = computeKmm(X_concat, l, sigma, dtype=dtype) + noise_mat

        if inverted:
            invNKmm = chol2inv(NKmm, dtype=dtype)
            ret.append(invNKmm)
        else:
            ret.append(NKmm)

    return tf.stack(ret)



def compute_mean_var_f_multiple_data(n_xs, n_ys_per_x, x, Xsamples, Ysamples, xs, fs, 
                    l, sigma, sigma0, NKInvs=None, fullcov=False, dtype=tf.float32):
    """
    x: nx x d
        
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    sigma: scalar
    Xsamples: n x d
    Ysamples: n x 1

    fs: n_xs x n_ys_per_x
    xs: n_xs x d

    NKInvs: n_xs x (n+1) x (n+1)
            NOTE: the first n element contain noise variance
                  the last element doesn't contain noise variance

    return: mean: n_xs x n_ys_per_x x nx
            var:  n_xs x nx if fullcov == False
                  n_xs x nx x nx if fullcov == True
    """
    if fullcov:
        Kx = computeKmm(x, l, sigma, 
                    dtype=dtype, inverted=False)

    if NKInvs is None:
        NKInvs = computeNKmm_multiple_data(n_xs, Xsamples, xs, 
                        l, sigma, sigma0, dtype=dtype, inverted=True)

    var_all = []
    mean_all = []

    for i in range(n_xs):
        NKInv = NKInvs[i,...]
        X_concat = tf.concat([Xsamples, tf.expand_dims(xs[i,:],0)], axis=0)

        kstar = computeKnm(x, X_concat, l, sigma)

        if fullcov:
            var = Kx - kstar @ NKInv @ tf.transpose(kstar)
            diag_var = tf.linalg.diag_part(var)
            diag_var = tf.clip_by_value(diag_var, clip_value_min=clip_min, clip_value_max=np.infty)
            var = tf.linalg.set_diag(var, diag_var)
        else:
            var = sigma - tf.reduce_sum( (kstar @ NKInv) * kstar, axis=1 )
            var = tf.clip_by_value(var, clip_value_min=clip_min, clip_value_max=np.infty)

        var_all.append(var)

        mean_i= []
        for j in range(n_ys_per_x):            
            mean = tf.squeeze(kstar @ (NKInv @ 
                            tf.concat([Ysamples, tf.reshape(fs[i,j], 
                                      shape=(1,1)) ], axis=0) ))
            mean_i.append(mean)
        mean_all.append(tf.stack(mean_i))
    
    print("utils.compute_mean_var_f_multiple_data: clip value of var at {}!".format(clip_min))
    var_all = tf.stack(var_all)
    mean_all = tf.stack(mean_all)

    return mean_all, var_all



def compute_mean_f(x, 
        xdim, n_hyp, 
        Xsamples, Ysamples, 
        ls, sigmas, sigma0s, 
        NKInvs, 
        dtype=tf.float32):
    """
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    Ysamples: n x 1
    Xsamples: n x d
    x: 1 x d

    return: mean: n x 1
            var : n x 1
    """

    mean = tf.constant(0.0, dtype=dtype)

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        NKInv = NKInvs[i]

        kstar = computeKnm(x, Xsamples, l, sigma)
        mean = mean + tf.squeeze(kstar @ (NKInv @ Ysamples)) / tf.constant(n_hyp, dtype=dtype)
    return mean


def find_top_k(ys, k):
    _, idxs = tf.math.top_k(tf.squeeze(ys), k, sorted=False)
    idxs = tf.reshape(idxs, (-1,))
    return idxs


def get_initializers(func, ngroups, groups, n_inits):
    # get top n_inits[i] points from groups[i]
    # n_inits[i]: scalar
    # groups[i]: array of shape n x xdim
    # func(x) -> scalar
    # func is a function need maximization
    # requires: all n_inits > 0
    top_k_inits = []

    for i in range(ngroups):
        inits = utils.find_top_k(func, groups[i], n_inits[i])
        top_k_inits.append(inits)

    return tf.concat(top_k_inits, axis=0)


def merge_2dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


# def find_top_ks(sess, func, feed_dict, input_key, init_groups, ks, batchsize):
def find_top_ks(func, init_groups, ks, batchsize):
    """
    init_groups: list of lists of inputs
    ks: list of k values
    batchsize: batchsize for evaluating func at 1 sess.run
    return: concatenation of top ks[i] of init_groups[i] for all i
    func = lambda x: sess.run(f, feed_dict=merge_2dicts(train_dict, {'initializers': x}))
    # need to test
    """
    res = []
    xdim = init_groups[0].shape[1]
    
    for gi,g in enumerate(init_groups):
        remain_size = g.shape[0] % batchsize
        if remain_size:
            padding_size = batchsize - (g.shape[0] % batchsize)
            padding = np.tile(g[-1,:].reshape(1,-1), reps=(padding_size,1))
            g = np.concatenate([g, padding], axis=0)
        else:
            padding_size = 0

        vals = []
        for i in range(0, g.shape[0], batchsize):
            # feed_dict[input_key] = g[i:i+batchsize,:]
            # val = sess.run(func, feed_dict=feed_dict)
            val = func(g[i:i+batchsize,:])
            vals.append(val)

        vals = np.squeeze(np.concatenate(vals))

        if padding_size:
            idxs = np.argsort(vals[:-padding_size])[-ks[gi]:]
        else:
            idxs = np.argsort(vals)[-ks[gi]:]

        res.append(g[idxs,:].reshape(-1,xdim))
    res = np.concatenate(res,axis=0)
    return res


def get_duplicate_mask_np(xs, resolution=1e-5):
    """
    duplicate_mask[i] = 1 if xs[i,:] is already in xs[:i,:]
    """
    n = xs.shape[0]

    duplicate_mask = np.zeros(n)

    for i in range(n):
        if duplicate_mask[i] == 1.0:
            # already duplicated
            continue 

        for j in range(i+1,n):
            adiff = xs[j,:] - xs[i,:]
            dist = np.sqrt( np.sum(adiff * adiff) )
            if dist <= resolution:
                duplicate_mask[j] = 1.0

    return duplicate_mask


def remove_duplicates_np(xs, resolution=1e-5):
    invalid_tests = get_duplicate_mask_np(xs, resolution)

    remove_idxs = np.where(invalid_tests == 1.0)[0]
    xs = np.delete(xs, remove_idxs, axis=0)
    return xs


def precomputeInvKs(xdim, nhyp, ls, 
                    sigmas, sigma0s, 
                    Xsamples, 
                    dtype):

    invKs = []
    for i in range(nhyp):

        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]

        NK = computeNKmm(Xsamples, l, sigma, sigma0, dtype=dtype)

        invK = chol2inv(NK, dtype=dtype)
        invKs.append(invK)

    invKs = tf.stack(invKs)
    return invKs


def eval_invKmaxsams(xdim, nhyp, nmax, 
                    ls, sigmas, sigma0s, 
                    Xsamples, 
                    maxima, 
                    dtype=tf.float32):
    # only required for PES criterion
    invKmaxsams = []
    for i in range(nhyp):

        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]

        invKmaxsams_i = []
        for j in range(nmax):
            xmax_xsam = tf.concat([tf.reshape(maxima[i,j,...], shape=(1,xdim)), Xsamples], axis=0)
            NKmaxsam = computeNKmm(xmax_xsam, l, sigma, sigma0, dtype=dtype)
            invKmaxsam = chol2inv(NKmaxsam, dtype=dtype)
            invKmaxsams_i.append(invKmaxsam)

        invKmaxsams.append(tf.stack(invKmaxsams_i))
    invKmaxsams = tf.stack(invKmaxsams)
    # nhyp x nmax x (Xsamples.shape[0] + maxima.shape[0]) x (Xsamples.shape[0] + maxima.shape[0])

    return invKmaxsams


