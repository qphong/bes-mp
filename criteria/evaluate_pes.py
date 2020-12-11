import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np 
import time 
import sys

import utils 


def pes(xs, xdim, n_max, n_hyp, Xsamples, Ysamples, ls, sigmas, sigma0s, xmaxs, invKs, invKmaxsams, max_observed_y, dtype=tf.float32, n_x=1):
    # invKmaxsams shape: (n_hyp, n_max, n_x+1, n_x+1)
    # TODO: fix xmaxs is of shape(n_hyp, n_max, xdim)
    
    pes_vals = [tf.constant(0.0, dtype=dtype) for i in range(n_x)]
    norm = tfp.distributions.Normal(loc=tf.constant(0., dtype=dtype), scale=tf.constant(1., dtype=dtype))

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        
        invK = invKs[i,...]

        mean_fmaxs_C2, var_fmaxs_C2, _, _ = imposeC2(xdim, n_max, Xsamples, Ysamples, xmaxs[i,...], max_observed_y, l, sigma, sigma0, invK, dtype)

        f_mean_all, f_var_all = utils.compute_mean_var_f(xs, Xsamples, Ysamples, l, sigma, sigma0, invK, dtype=dtype)

        zero_const = tf.constant(0.0, dtype=dtype)

        for idx in range(n_x):
            x = tf.reshape(xs[idx,...], shape=(1,xdim))

            ent_y = utils.evaluate_norm_entropy(tf.sqrt(tf.squeeze(f_var_all[idx]) + sigma0), dtype=dtype)

            cond_ent_y = zero_const

            for j in range(n_max):
                invKmaxsam = invKmaxsams[i,j,...]
                mean_fmax_C2 = mean_fmaxs_C2[j]
                var_fmax_C2 = var_fmaxs_C2[j]
                xmax = tf.reshape(xmaxs[i,j,...], shape=(1,xdim))

                _, var_f_C2_C3 = imposeC3(xdim, x, Xsamples, Ysamples, xmax, mean_fmax_C2, var_fmax_C2, l, sigma, sigma0, invK, invKmaxsam, norm, dtype)

                cond_ent_y = cond_ent_y + utils.evaluate_norm_entropy(tf.sqrt(var_f_C2_C3 + sigma0), dtype=dtype)

            cond_ent_y = cond_ent_y / tf.constant(n_max, dtype=dtype)
            pes_vals[idx] = pes_vals[idx] + (ent_y - cond_ent_y) / tf.constant(n_hyp, dtype=dtype)

    return tf.squeeze(tf.stack(pes_vals))



def imposeC2(xdim, n_max, Xsamples, Ysamples, xmaxs, max_observed_y, l, sigma, sigma0, invK, dtype=tf.float32):
    # C2: fmax is larger than past observations (> max_observed_y)
    # xmaxs: shape(n_max x xdim)
    # max_observed_y: scalar
    # l: shape(1,xdim)
    # sigma, sigma0: scalars

    max_observed_y = tf.squeeze(max_observed_y)
    
    norm = tfp.distributions.Normal(loc=tf.constant(0., dtype=dtype), scale=tf.constant(1., dtype=dtype))

    fmax_mean, fmax_var = utils.compute_mean_var_f(xmaxs, Xsamples, Ysamples, l, sigma, sigma0, invK)

    tmp = tf.sqrt( tf.constant(1.0, dtype=dtype) + fmax_var / sigma0)

    z = (fmax_mean - max_observed_y) / tf.sqrt(sigma0) / tmp

    pdf_over_cdf_z = tf.exp(norm.log_prob(z) - norm.log_cdf(z))

    mean_fmax_C2 = fmax_mean + fmax_var * pdf_over_cdf_z / ( tf.sqrt(sigma0) * tmp )
    var_fmax_C2 = tf.clip_by_value( fmax_var - fmax_var * fmax_var * pdf_over_cdf_z / (sigma0 + fmax_var) * (z + pdf_over_cdf_z), clip_value_min = 1e-9, clip_value_max=np.infty )

    return mean_fmax_C2, var_fmax_C2, fmax_mean, fmax_var
    # both are of shape (n_max,)



def imposeC3(xdim, x, Xsamples, Ysamples, xmax, mean_fmax_C2, var_fmax_C2, l, sigma, sigma0, invK, invKmaxsam, norm, dtype=tf.float32):
    # C3: f(x) is smaller than f_max
    # xmax: shape(1,xdim)
    # mean_fmax_C2, var_fmax_C2: scalars
    # IMPORTANT: only use with a single x
    #     x.shape=(1,xdim)
    # invK: include noise
    # invKmaxsam = inv( K_{xmax Xsamples, xmax Xsamples} + sigma_n^2 * [0;I] )
    """
    TODO: if x is close to xmax (wrt the lengthscale): return the mean_fmax_C2, var_fmax_C2
    """

    xmax = tf.reshape(xmax, shape=(1,xdim))
    x = tf.reshape(x, shape=(-1,xdim))

    mean_fmax_C2 = tf.reshape(mean_fmax_C2, shape=(1,1))
    var_fmax_C2 = tf.reshape(var_fmax_C2, shape=(1,1))

    # distribution of 
    # [f;fmax] given Xsamples, Ysamples, C2
    # let f2 = [fmax; f]
    #     x2 = [xmax; x]

    xmaxsam = tf.concat([xmax, Xsamples], axis=0)
    Kx_xmaxsam = utils.computeKnm(x, xmaxsam, l, sigma)
    tmp = Kx_xmaxsam @ invKmaxsam
    # in the notes page 3:
    # tmp[0] = a
    # tmp[1:] = B

    a = tmp[0,0]

    mean_f_C2 = tf.reshape(tmp @ tf.concat([ mean_fmax_C2, Ysamples ], axis=0), shape=(1,1))
    mean_f2_C2 = tf.reshape(tf.concat([mean_fmax_C2, mean_f_C2], axis=0), shape=(2,1))

    Kx = utils.computeKmm(x, l, sigma)
    var_f_given_fmax = Kx - tf.matmul(tmp, Kx_xmaxsam, transpose_b=True)
    m00 = var_fmax_C2
    m01 = a * var_fmax_C2
    m11 = var_f_given_fmax + a*a*var_fmax_C2
    var_f2_C2 = tf.reshape(tf.concat([m00, m01, m01, m11], axis=0), shape=(2,2))


    # let Z = fmax - f
    # distribution of Z given Xsamples, Ysamples, C2
    # mean_z, var_z
    c = tf.constant([[1.0], [-1.0]], dtype=dtype)
    mean_z = tf.squeeze(tf.matmul(c, mean_f2_C2, transpose_a=True))
    var_z = tf.squeeze(tf.matmul(c, var_f2_C2 @ c, transpose_a=True))

    alpha = mean_z / tf.sqrt(var_z)
    beta = tf.exp(norm.log_prob(alpha) - norm.log_cdf(alpha))

    mean_f_C2_C3 = tf.squeeze(mean_fmax_C2 - mean_z - tf.sqrt(var_z) * beta)
    var_f2_C2_C3 = var_f2_C2 - beta * (alpha + beta) * ((var_f2_C2 @ c) @ tf.matmul(c, var_f2_C2, transpose_a=True)) / var_z
    var_f_C2_C3 = tf.squeeze(var_f2_C2_C3[1,1])

    return mean_f_C2_C3, var_f_C2_C3



