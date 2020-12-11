import sys
sys.path.insert(0, './criteria')
sys.path.insert(0, './criteria/mes_criteria')

import os
import argparse

DEBUG = False
use_GPU_for_sample_functions = False

parser = argparse.ArgumentParser(description='Implicit Level Set Estimation')
parser.add_argument('-g', '--gpu', help='gpu device index for tensorflow',
                    required=False,
                    type=str,
                    default='0')
parser.add_argument('-c', '--criterion', help='BO acquisition function',
                    required=False,
                    type=str,
                    default='mnes2')
parser.add_argument('-n', '--noisevar', help='noise variance',
                    required=False,
                    type=float,
                    default=0.09)
parser.add_argument('-q', '--numqueries', help='number/budget of queries',
                    required=False,
                    type=int,
                    default=80)
parser.add_argument('-r', '--numruns', help='number of random experiments',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-s', '--numhyps', help='number of sampled hyperparameters',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-m', '--nmax', help='number of function samples',
                    required=False,
                    type=int,
                    default=5)
parser.add_argument('-u', '--nfeature', help='number of features to sample functions',
                    required=False,
                    type=int,
                    default=100)
parser.add_argument('-a', '--nparal', help='number of parallel iterations',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-y', '--nysample', help='number of y samples to evaluate acquisition',
                    required=False,
                    type=int,
                    default=50)
parser.add_argument('--ntrain', help='number of optimizing iterations',
                    required=False,
                    type=int,
                    default=500)
parser.add_argument('--ninit', help='number of initial observations',
                    required=False,
                    type=int,
                    default=2)
parser.add_argument('--function', help='function to optimize',
                    required=False,
                    type=str,
                    default='func_1d_4modes')
parser.add_argument('--alpha', help='searching for max_value - alpha',
                    required=False,
                    type=float,
                    default=1.0)
parser.add_argument('-t', '--dtype', help='type of float: float32 or float64',
                    required=False,
                    type=str,
                    default='float64')


args = parser.parse_args()

# print all arguments
print('================================')
for arg in vars(args):
    print(arg, getattr(args, arg))
print('================================')

gpu_device_id = args.gpu

criterion = args.criterion


folder = args.function
if not os.path.exists(folder):
    os.makedirs(folder)

folder = "{}/alpha_{}".format(folder, args.alpha)
if not os.path.exists(folder):
    os.makedirs(folder)

folder = "{}/noise_var_{}".format(folder, args.noisevar)
if not os.path.exists(folder):
    os.makedirs(folder)
    
folder = '{}/{}'.format(folder, args.criterion)
if not os.path.exists(folder):
    os.makedirs(folder)

nquery = args.numqueries
nrun = args.numruns
nhyp = args.numhyps

nmax = args.nmax
nfeature = args.nfeature

alpha = args.alpha

nysample = args.nysample
parallel_iterations = args.nparal

ntrain = args.ntrain
n_initial_training_x = args.ninit
func_name = args.function

print("nrun: {}".format(nrun))
print("nquery: {}".format(nquery))
print("nhyp: {}".format(nhyp))
print("nmax (nfunc): {}".format(nmax))
print("n_initial_training_x: {}".format(n_initial_training_x))
print("Function: {}".format(func_name))
print("Alpha: {}".format(alpha))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device_id


import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp
import scipy as sp 
import time 
import scipy.stats as spst


import matplotlib.pyplot as plt 


import utils 
import utils_for_continuous
import optfunc
import functions

import evaluate_mnes2
import evaluate_mnes3


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = False
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.44

if args.dtype == 'float32':
    dtype = tf.float32
    nptype = np.float32
elif args.dtype == 'float64':
    dtype = tf.float64
    nptype = np.float64
else:
    raise Exception("Unknown dtype: {}".format(args.dtype))


duplicate_resolution=0.001
print("duplicate_resolution = {} to avoid degenerate matrix if 2 testing points are too close to each other".format(duplicate_resolution))


def evaluate_criterion(xs, nx,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    required_placeholders,
                    dtype=tf.float32):
    xdim = crit_params['xdim']
    nhyp = crit_params['nhyp']
    alpha = crit_params['alpha']

    Xsamples = required_placeholders['X']
    Ysamples = required_placeholders['Y']
 
    invKs = required_placeholders['invKs']
    opt_fsample_maxima = required_placeholders['opt_fsample_maxima']
    # (nhyp, n_maxima)

    ymaxs = tf.expand_dims( opt_fsample_maxima, axis=-1)
    # (nhyp, n_maxima, 1)
    levels = ymaxs - alpha
    # (nhyp, n_maxima, 1)

    if criterion == 'mnes2':
        bounds = levels[:,:,0]

        vals = evaluate_mnes2.mnes(xs, 
            ls, sigmas, sigma0s, 
            Xsamples, Ysamples, 

            xdim, nhyp, 
            
            bounds, 
            invKs, 
            nsamples=crit_params['nysample'], 
            dtype=dtype)

    elif criterion == 'mnes3':
        bounds = tf.concat([levels, ymaxs], axis=2, name='bounds')

        vals = evaluate_mnes3.mnes(xs, 
            ls, sigmas, sigma0s, 
            Xsamples, Ysamples, 

            xdim, nhyp, 
            
            bounds, 
            invKs, 
            nsamples=crit_params['nysample'], 
            dtype=dtype)

    else:
        raise Exception("Unknown criterion: {}".format(criterion))

    return vals


def get_required_placeholders(criterion, crit_params,
                dtype,
                is_debug_mode = False):
    nhyp = crit_params['nhyp']
    xdim = crit_params['xdim']
    nmax = crit_params['nmax']
    nfeature = crit_params['nfeature']

    X_plc = tf.placeholder(dtype=dtype, shape=(None, xdim), name='X_plc')
    Y_plc = tf.placeholder(dtype=dtype, shape=(None, 1), name='Y_plc')
    validation_X_plc = tf.placeholder(dtype=dtype, shape=(None,xdim), name='validation_X_plc')

    max_observed_y_plc = tf.placeholder(dtype=dtype, shape=(), 
        name = 'max_observed_y_plc')

    # estimate fmax optimistic with UCB
    optimistic_maximum_plc = tf.placeholder(dtype=dtype, shape=(),
        name = "optimistic_maximum_plc")

    invKs_plc = tf.placeholder(shape=(nhyp,None,None), dtype=dtype, name='invKs_plc')
    # (nhyp, nobs, nobs)
    opt_fsample_maxima_plc = tf.placeholder(dtype=dtype, shape=(nhyp, None), name='opt_fsample_maximizers_plc')
    # (nhyp,nmax)
    opt_fsample_maximizers_plc = tf.placeholder(dtype=dtype, shape=(nhyp, None, xdim), name='opt_fsample_maxima_plc')
    # (nhyp,nmax,xdim)

    opt_meanf_maximizer_plc = tf.placeholder(dtype = dtype, shape=(1,xdim), name = 'opt_meanf_maximizer_plc')
    opt_meanf_maximum_plc = tf.placeholder(dtype = dtype, shape=(), name = 'opt_meanf_maximum_plc')
    # assuming same for all nhyp

    test_xs_plc = tf.placeholder(dtype=dtype, 
                shape=(None, crit_params['xdim']), 
                name='test_xs')
    # (ntest,xdim)s
    max_probs_plc = tf.placeholder(dtype=dtype, shape=( crit_params['nhyp'], None), name='max_probs_plc')
    # (nhyp, nmax)
    post_mean_tests_plc = tf.placeholder(dtype=dtype, 
            shape=( crit_params['nhyp'], None, None), 
            name='post_mean_tests_plc')
    # (nhyp, nmax, ntest)
    post_cov_tests_plc = tf.placeholder(dtype=dtype, 
            shape=( crit_params['nhyp'], None, None, None), 
            name='post_cov_tests_plc')
    # (nhyp, nmax, ntest, ntest)

    invpNKs_plc = tf.placeholder(dtype=dtype, 
            shape=( crit_params['nhyp'], None, None), 
            name='invpNKs_plc')
    # (nhyp, nobs+ntest, nobs+ntest)

    opt_meanf_candidate_xs_plc = tf.placeholder(dtype = dtype, 
                            shape = (None,xdim),
                            name = 'opt_meanf_candidate_xs')
    opt_fsample_candidate_xs_plc = tf.placeholder(dtype = dtype,
                            shape = (None,xdim),
                            name = 'opt_fsample_candidate_xs')
    opt_crit_candidate_xs_plc = tf.placeholder(dtype = dtype,
                            shape = (None,xdim),
                            name = 'opt_crit_candidate_xs')
    
    thetas_plc = tf.placeholder(dtype = dtype,
        shape = (nhyp, nmax, nfeature, 1),
        name = 'thetas')
    Ws_plc = tf.placeholder(dtype = dtype,
        shape = (nhyp, nmax, nfeature, xdim),
        name = 'Ws')
    bs_plc = tf.placeholder(dtype = dtype,
        shape = (nhyp, nmax, nfeature, 1),
        name = 'bs')

    iteration_plc = tf.placeholder(dtype=dtype, shape=(), name='iteration_plc')

    required_placeholders = {
        'iteration': iteration_plc,
        'X': X_plc,
        'Y': Y_plc,
        'validation_X': validation_X_plc,

        'opt_meanf_candidate_xs': opt_meanf_candidate_xs_plc,
        'opt_fsample_candidate_xs': opt_fsample_candidate_xs_plc,
        'opt_crit_candidate_xs': opt_crit_candidate_xs_plc,
        'invKs': invKs_plc,

        'thetas': thetas_plc,
        'Ws': Ws_plc,
        'bs': bs_plc,

        'opt_fsample_maximizers': opt_fsample_maximizers_plc,
        'opt_fsample_maxima': opt_fsample_maxima_plc,


        'opt_meanf_maximizer': opt_meanf_maximizer_plc
    }

    if criterion in ['mes', 'rmes', 'avg_bes_mp', 'marginal_bes_mp', 'dare_mp', 'straddle_mp', 'maxent_mp', 'sampletmes']:
        required_placeholders['opt_fsample_maxima'] = opt_fsample_maxima_plc
    
    elif criterion in ['avg_bes_mean', 'meanftmes', 'marginal_bes_mean', 'dare_mean', 'straddle_mean', 'maxent_mean']:
        required_placeholders['opt_meanf_maximum'] = opt_meanf_maximum_plc

    elif criterion in ['avg_bes_ucb', 'otmes']:
        required_placeholders['optimistic_maximum'] = optimistic_maximum_plc

    elif criterion == 'ei':
        required_placeholders['max_observed_y'] = max_observed_y_plc
    
    elif criterion == 'eim':
        required_placeholders['opt_meanf_maximum'] = opt_meanf_maximum_plc

    elif criterion == 'ucb':
        pass
    
    elif criterion == 'pes':
        invKmaxsams_plc = tf.placeholder(dtype=dtype, 
                shape=(nhyp, None, None, None), # nhyp, nmax, nobs+1, nobs+1
                name='invKmaxsams_plc')

        required_placeholders['opt_fsample_maximizer'] = opt_fsample_maximizers_plc
        required_placeholders['invKmaxsams'] = invKmaxsams_plc
        required_placeholders['max_observed_y'] = max_observed_y_plc

    elif criterion in ['ftl', 'lftl']:
        required_placeholders['test_xs'] = test_xs_plc
        required_placeholders['max_probs'] = max_probs_plc
        required_placeholders['post_mean_tests'] = post_mean_tests_plc
        required_placeholders['post_cov_tests'] = post_cov_tests_plc
        required_placeholders['invKs'] = invKs_plc
        required_placeholders['invpNKs'] = invpNKs_plc

    elif criterion == 'sftl':
        post_test_samples_plc = tf.placeholder(dtype=dtype, 
                                shape=(nhyp, None, None, None), 
                                name='post_test_samples_plc')
        post_test_masks_plc = tf.placeholder(dtype=tf.bool,
                                shape=(nhyp, None, None), 
                                name='post_test_masks_plc')

        required_placeholders['test_xs'] = test_xs_plc
        required_placeholders['max_probs'] = max_probs_plc
        required_placeholders['post_test_samples'] = post_test_samples_plc
        required_placeholders['post_test_masks'] = post_test_masks_plc
        required_placeholders['invKs'] = invKs_plc
        required_placeholders['invpNKs'] = invpNKs_plc

    return required_placeholders


def get_intermediate_tensors(criterion, crit_params,
            required_placeholders,
            ls, sigmas, sigma0s,
            dtype,
            is_debug_mode=False):
    xdim = crit_params['xdim']
    nhyp = crit_params['nhyp']
    nmax = crit_params['nmax']
    nfeature = crit_params['nfeature']
    xmin = crit_params['xmin']
    xmax = crit_params['xmax']
    opt_meanf_top_init_k = crit_params['opt_meanf_top_init_k']
    opt_fsample_top_init_k = crit_params['opt_fsample_top_init_k']
    opt_crit_top_init_k = crit_params['opt_crit_top_init_k']

    X_plc = required_placeholders['X']
    Y_plc = required_placeholders['Y']

    intermediate_tensors = {}

    max_observed_y = tf.reduce_max(Y_plc)
    intermediate_tensors['max_observed_y'] = max_observed_y

    invKs = utils.precomputeInvKs(xdim, nhyp, 
                ls, sigmas, sigma0s, 
                X_plc, dtype)
    # nhyp x nobs x nobs
    intermediate_tensors['invKs'] = invKs


    meanf, varf =  utils.compute_mean_var_f(required_placeholders['validation_X'],
                                X_plc, Y_plc,
                                tf.reshape(ls[0,...], shape=(1,-1)), 
                                sigmas[0,...], sigma0s[0,...],
                                NKInv=invKs[0,...], fullcov=False,
                                dtype=dtype)
    stdf = tf.sqrt(varf)

    intermediate_tensors['meanf'] = meanf 
    intermediate_tensors['stdf'] = stdf 


    # optimize mean function
    opt_meanf_func = lambda x: utils.compute_mean_f(
                                tf.reshape(x, shape=(-1,xdim)),
                                xdim, nhyp,
                                X_plc, Y_plc,
                                ls, sigmas, sigma0s,
                                required_placeholders['invKs'],
                                dtype=dtype)

    opt_meanf_assign, opt_meanf_train, opt_meanf_maximizer, opt_meanf_maximum \
        = utils_for_continuous.optimize_continuous_function(xdim, 
                opt_meanf_func,
                required_placeholders['opt_meanf_candidate_xs'],
                opt_meanf_top_init_k,
                parallel_iterations=parallel_iterations,
                xmin = xmin,
                xmax = xmax,
                dtype= dtype,
                name = 'opt_meanf')
    
    intermediate_tensors['opt_meanf_assign'] = opt_meanf_assign
    intermediate_tensors['opt_meanf_train'] = opt_meanf_train
    intermediate_tensors['opt_meanf_maximizer'] = opt_meanf_maximizer
    intermediate_tensors['opt_meanf_maximum'] = opt_meanf_maximum

    thetas_all, Ws_all, bs_all = utils_for_continuous.sample_function(
                        xdim, nhyp, nmax, nfeature,
                        ls, sigmas, sigma0s,
                        X_plc, Y_plc,
                        dtype=dtype)
    intermediate_tensors['thetas'] = thetas_all
    intermediate_tensors['Ws'] = Ws_all
    intermediate_tensors['bs'] = bs_all

    # optimize function samples
    opt_fsample_assigns, opt_fsample_trains, \
    opt_fsample_maximizers, opt_fsample_maxima, \
    _, opt_fsample_top_k_inits \
        = utils_for_continuous.sample_xmaxs_fmaxs(
                xdim, nhyp, nmax, nfeature,
                ls, sigmas, sigma0s, 

                required_placeholders['thetas'],
                required_placeholders['Ws'],
                required_placeholders['bs'],

                required_placeholders['opt_fsample_candidate_xs'],
                opt_fsample_top_init_k,

                xmin, xmax,
                get_xs=True,
                dtype=dtype,
                parallel_iterations=parallel_iterations,
                name='sample_xmaxs_fmaxs')

    intermediate_tensors['opt_fsample_assigns'] = opt_fsample_assigns
    intermediate_tensors['opt_fsample_trains'] = opt_fsample_trains
    intermediate_tensors['opt_fsample_maximizers'] = opt_fsample_maximizers
    intermediate_tensors['opt_fsample_maxima'] = opt_fsample_maxima
    # for debugging
    intermediate_tensors['opt_fsample_top_k_inits'] = opt_fsample_top_k_inits

    # optimize acquisition function
    opt_fsample_maximizers = required_placeholders['opt_fsample_maximizers'] # (nhyp, nmax, xdim)
    opt_fsample_maxima = required_placeholders['opt_fsample_maxima'] # (nhyp, nmax)

    opt_crit_func = lambda x: evaluate_criterion(
                    tf.reshape(x, shape=(-1,xdim)), 
                    1,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    required_placeholders,
                    dtype=dtype)

    opt_crit_multiple_func = lambda x: evaluate_criterion(
                    tf.reshape(x, shape=(-1,xdim)), 
                    opt_crit_top_init_k,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    required_placeholders,
                    dtype=dtype)

    opt_crit_assign, opt_crit_train, \
    opt_crit_maximizer, opt_crit_maximum \
        = utils_for_continuous.optimize_continuous_function(
        xdim, opt_crit_func, 
        required_placeholders['opt_crit_candidate_xs'],
        opt_crit_top_init_k,
        parallel_iterations = parallel_iterations,
        xmin = xmin,
        xmax = xmax,
        dtype = dtype,
        name = 'optimize_crit',
        multiple_func = opt_crit_multiple_func)

    intermediate_tensors['opt_crit_assign'] = opt_crit_assign
    intermediate_tensors['opt_crit_train'] = opt_crit_train
    intermediate_tensors['opt_crit_maximizer'] = opt_crit_maximizer
    intermediate_tensors['opt_crit_maximum'] = opt_crit_maximum

    return intermediate_tensors



def get_placeholder_values(sess, 
        criterion, crit_params,
        required_placeholders,
        intermediate_tensors,
        ls, sigmas, sigma0s,
        X_np, Y_np,
        candidate_xs,
        # for computing log loss
        validation_xs, # (m,xdim) np array
        validation_fs, # (m,1) np array
        level, # scalar
        # # #
        init_random,
        init_npoint,
        init_seed=None,
        previous_opt_meanf_maximizer=None, # (1,xdim)
        iteration=1,
        dtype=tf.float32,
        is_debug_mode=False):

    xdim = crit_params['xdim']
    nmax = crit_params['nmax']
    ntrain = crit_params['ntrain']
    nfeature = crit_params['nfeature']
    xmin = crit_params['xmin']
    xmax = crit_params['xmax']

    values = {
            'iteration': iteration,
            'query_x': None}

    if 'max_observed_y' in intermediate_tensors:
        max_observed_y_np = sess.run(
            intermediate_tensors['max_observed_y'],
            feed_dict = { required_placeholders['Y']: Y_np }
        )

        values['max_observed_y'] = max_observed_y_np

    if 'invKs' in intermediate_tensors:
        invKs_np = sess.run(intermediate_tensors['invKs'], 
            feed_dict = {
                required_placeholders['X']: X_np
            })
        values['invKs'] = invKs_np
    
    if previous_opt_meanf_maximizer is None:
        previous_opt_meanf_maximizer = functions.get_initializers(
                                        xdim, 
                                        xmin, xmax, 
                                        name = "random", 
                                        random=True, npoint=1, 
                                        seed=None)

    # Optimize for marginal_bes_mpt guess
    if candidate_xs['opt_meanf'] is None:
        opt_meanf_candidate_xs_np = functions.get_initializers(
                                        xdim, 
                                        xmin, xmax, 
                                        name = "opt_meanf", 
                                        random = init_random, 
                                        npoint = init_npoint, 
                                        seed = init_seed)

        opt_meanf_candidate_xs_np = np.concatenate([opt_meanf_candidate_xs_np, previous_opt_meanf_maximizer], axis=0)
    else:
        opt_meanf_candidate_xs_np = candidate_xs['opt_meanf']

    sess.run(intermediate_tensors['opt_meanf_assign'],
        feed_dict = {
            required_placeholders['opt_meanf_candidate_xs']: opt_meanf_candidate_xs_np,
            required_placeholders['X']: X_np,
            required_placeholders['Y']: Y_np,
            required_placeholders['invKs']: values['invKs']
        })

    for _ in range(crit_params['ntrain']):
        sess.run(intermediate_tensors['opt_meanf_train'],
            feed_dict = {
                required_placeholders['X']: X_np,
                required_placeholders['Y']: Y_np,
                required_placeholders['invKs']: values['invKs']
            })

    opt_meanf_maximizer_np, opt_meanf_maximum_np \
        = sess.run([
                intermediate_tensors['opt_meanf_maximizer'],
                intermediate_tensors['opt_meanf_maximum'] ],
            feed_dict = {
                required_placeholders['X']: X_np,
                required_placeholders['Y']: Y_np,
                required_placeholders['invKs']: values['invKs']
            })

    values['opt_meanf_maximizer'] = opt_meanf_maximizer_np
    values['opt_meanf_maximum'] = opt_meanf_maximum_np


    if 'opt_fsample_maximizer' in required_placeholders or 'opt_fsample_maxima' in required_placeholders:

        if use_GPU_for_sample_functions:
            # sample functions
            # move the computation of thetas, Ws, bs
            # to numpy instead of tensorflow 
            # to avoid numerical error (return None of tf.linalg.eigh)
            print("use GPU to sample functions.")
            thetas_np, Ws_np, bs_np = sess.run(
                [ intermediate_tensors['thetas'],
                intermediate_tensors['Ws'],
                intermediate_tensors['bs'] ],
                feed_dict = {
                    required_placeholders['X']: X_np,
                    required_placeholders['Y']: Y_np
                })

        else:
            print("use CPU to sample functions.")
            thetas_np = np.zeros([nhyp, nmax, nfeature, 1])
            Ws_np = np.zeros([nhyp, nmax, nfeature, xdim])
            bs_np = np.zeros([nhyp, nmax, nfeature, 1])

            for hyp_idx in range(nhyp):
                thetas_np[hyp_idx,...], Ws_np[hyp_idx,...], bs_np[hyp_idx,...] \
                    = optfunc.draw_random_init_weights_features_np(
                        xdim, nmax, nfeature,
                        X_np, Y_np,
                        ls[hyp_idx], sigmas[hyp_idx], sigma0s[hyp_idx])

        # optimize functions
        # assign initial values
        
        if candidate_xs['opt_fsample'] is None:
            opt_fsample_candidate_xs_np = functions.get_initializers(
                                            xdim, 
                                            xmin, xmax, 
                                            name = "opt_fsample", 
                                            random = init_random, 
                                            npoint = init_npoint, 
                                            seed = init_seed + 123)

            opt_fsample_candidate_xs_np = np.concatenate([opt_fsample_candidate_xs_np, opt_meanf_maximizer_np], axis=0)
        else:
            opt_fsample_candidate_xs_np = candidate_xs['opt_fsample']
        
        sess.run(intermediate_tensors['opt_fsample_assigns'],
            feed_dict = {
                required_placeholders['thetas']: thetas_np,
                required_placeholders['Ws']: Ws_np,
                required_placeholders['bs']: bs_np,
                required_placeholders['opt_fsample_candidate_xs']: opt_fsample_candidate_xs_np
            })

        for xx in range(ntrain):
            sess.run(intermediate_tensors['opt_fsample_trains'],
                feed_dict = {
                    required_placeholders['thetas']: thetas_np,
                    required_placeholders['Ws']: Ws_np,
                    required_placeholders['bs']: bs_np
                })
      
        opt_fsample_maximizers_np, opt_fsample_maxima_np \
            = sess.run([
                    intermediate_tensors['opt_fsample_maximizers'],
                    intermediate_tensors['opt_fsample_maxima'] ],
                feed_dict = {
                        required_placeholders['thetas']: thetas_np,
                        required_placeholders['Ws']: Ws_np,
                        required_placeholders['bs']: bs_np
                })

        values['opt_fsample_maximizers'] = opt_fsample_maximizers_np
        values['opt_fsample_maxima'] = opt_fsample_maxima_np


    # evaluating the log loss performance metric
    """
        we don't know the true level, use the max mean as an estimate of the max-value
        then predict the level
    """
    estimated_levels = (values['opt_fsample_maxima'][0,:] - alpha).reshape(1,-1)
    # (1,nmax)
   
    # - E_x log E_{estimated_level} p(correct_label(x) | estimated_level)
    meanf_val, stdf_val = sess.run([intermediate_tensors['meanf'], 
                                    intermediate_tensors['stdf']],
                        feed_dict = {
                            required_placeholders['X']: X_np,
                            required_placeholders['Y']: Y_np,
                            required_placeholders['validation_X']: validation_xs
                        })

    sign = (level - validation_fs) / np.abs(validation_fs - level)
    
    sign = sign.reshape(-1,1) # (n,1)
    meanf_val = meanf_val.reshape(-1,1) # (n,1)
    stdf_val = stdf_val.reshape(-1,1) # (n,1)

    logloss = - np.mean( 
                sp.misc.logsumexp( # expectation over estimated_levels
                    spst.norm.logcdf(
                        sign 
                        * (estimated_levels - meanf_val) / stdf_val)
                    - np.log( estimated_levels.shape[1] ),
                    axis=1) )
    
    values['logloss'] = logloss 

    return values 




tf.reset_default_graph()

f_info = getattr(functions, func_name)()
print("Information of function:")
for k in f_info:
    if k != 'xs':
        print("{}: {}".format(k, f_info[k]))
    else:
        print("xs.shape: {}".format(f_info['xs'].shape))

f = f_info['function'] 
xmin = f_info['xmin']
xmax = f_info['xmax']
candidate_xs_to_optimize_np = f_info['xs']

init_random = f_info['init_random']
if init_random:
    init_npoint = 1000
else:  
    init_npoint = f_info['npoint_per_dim']

xdim = f_info['xdim']

true_l = f_info['RBF.lengthscale']
true_sigma = f_info['RBF.variance']
true_sigma0 = f_info['noise.variance']
true_sigma0 = args.noisevar
print("Override true_sigma0 = {}".format(true_sigma0))
true_maximum = f_info['maximum']

ls_np = true_l.reshape(-1,xdim).astype(nptype)
sigmas_np = np.array([ true_sigma ], dtype=nptype)
sigma0s_np = np.array([ true_sigma0 ], dtype=nptype)

seed = 1

print("True GP hyperparameters: l:{} sigma:{} sigma0(noise var):{}".format(true_l, true_sigma, true_sigma0))
print("nhyp: {}".format(nhyp))
print("nrun:{}, nqueries:{}".format(nrun, nquery))
print("____________________________________________")

validation_random = False
validation_npoints = 7001

if xdim == 2:
    validation_random = False
    validation_npoints = 81
elif xdim >= 3:
    validation_random = True
    validation_npoints = 7001

validation_xs = functions.get_initializers(
                                xdim, 
                                xmin, xmax, 
                                name = "validation", 
                                random = validation_random, 
                                npoint = validation_npoints, 
                                seed = 123)
validation_fs = f(validation_xs)
true_level = f_info['maximum'] - alpha

ls_toload = tf.get_variable(dtype=dtype, shape=(nhyp,xdim), name='ls')
sigmas_toload = tf.get_variable(dtype=dtype, shape=(nhyp,), name='sigmas') 
sigma0s_toload = tf.get_variable(dtype=dtype, shape=(nhyp,), name='sigma0s')

crit_params = {'nhyp': nhyp,
               'xdim': xdim,
               'nmax': nmax,
               'nfeature': nfeature,

               'xmin': xmin,
               'xmax': xmax,

               'alpha': alpha,

               'nysample': nysample,
               'parallel_iterations': parallel_iterations,
               
               'opt_meanf_top_init_k': 5,
               'opt_fsample_top_init_k': 10,
               'opt_crit_top_init_k': 7,
               
               'ntrain': ntrain,
               'n_min_sample': 2}

print("crit_params: {}".format(crit_params))


required_placeholder_keys = {
    'mnes2': ['X', 'Y', 'opt_fsample_maxima', 'invKs'],
    'mnes3': ['X', 'Y', 'opt_fsample_maxima', 'invKs']}

required_placeholders = get_required_placeholders(criterion, crit_params, dtype, is_debug_mode=False)

intermediate_tensors = get_intermediate_tensors(criterion, crit_params,
            required_placeholders,
            ls_toload, sigmas_toload, sigma0s_toload,
            dtype,
            is_debug_mode=False)

log_losses = np.zeros([nrun, nquery+1])

all_xx = np.zeros([nrun, nquery + n_initial_training_x, xdim])
all_ff = np.zeros([nrun, nquery + n_initial_training_x]) 
all_yy = np.zeros([nrun, nquery + n_initial_training_x])
# samples of maximum values
all_maximum_samples = np.zeros([nrun, nquery, nmax])


with tf.Session(config=gpu_config) as sess:
    for nr in range(nrun):
        rseed = seed + nr
        print("tf and np random seed: {}".format(rseed))
        np.random.seed(rseed)
        tf.set_random_seed(rseed)

        Xsamples_np = np.random.rand(n_initial_training_x,xdim) * (xmax - xmin) + xmin
        Fsamples_np = f(Xsamples_np).reshape(-1,1).astype(nptype)
        Ysamples_np = (Fsamples_np + np.random.randn(Xsamples_np.shape[0],1) * np.sqrt(true_sigma0)).astype(nptype)

        print("")

        previous_opt_meanf_maximizer = None 

        mean_f_const = 0.0
        min_npoint_opt_hyper = 12
        opt_hyp_every = 3
        last_opt_hyp_iter = -100

        for nq in range(nquery):

            startime_query = time.time()

            # for randomly drawing different functions
            sess.run(tf.global_variables_initializer())

            print("")
            print("{}:{}.=================".format(nr, nq))
            print("  X: {}".format(Xsamples_np.T))
            print("  Y: {}".format(Ysamples_np.T))

            if Xsamples_np.shape[0] < min_npoint_opt_hyper:
                pass
            elif (nq - last_opt_hyp_iter) > opt_hyp_every:
                last_opt_hyp_iter = nq

            ls_toload.load(ls_np, sess)
            sigmas_toload.load(sigmas_np, sess)
            sigma0s_toload.load(sigma0s_np, sess)
            print("")

            candidate_xs = {
                'opt_meanf': None, # candidate_xs_to_optimize_np,
                'opt_fsample': None, # candidate_xs_to_optimize_np,
                'opt_crit': None # candidate_xs_to_optimize_np
            }

            while True:
                # repeat if query_x is nan
                placeholder_values = get_placeholder_values(sess,
                            criterion, crit_params,
                            required_placeholders,
                            intermediate_tensors,
                            ls_np, sigmas_np, sigma0s_np,
                            Xsamples_np, Ysamples_np - mean_f_const,
                            candidate_xs,
                            # for computing log loss
                            validation_xs, # (m,xdim) np array
                            validation_fs, # (m,1) np array
                            true_level, # scalar 
                            # # #
                            init_random,
                            init_npoint,
                            init_seed = seed + nr + nq + np.random.randint(100000),
                            previous_opt_meanf_maximizer = previous_opt_meanf_maximizer,
                            iteration=nq+1,
                            dtype=dtype,
                            is_debug_mode=False)

                previous_opt_meanf_maximizer = placeholder_values['opt_meanf_maximizer'].reshape(1,xdim)

                print("Logloss: {}".format(placeholder_values['logloss']))
                log_losses[nr,nq] = placeholder_values['logloss']
                print("All logloss: {}".format(log_losses[nr,:nq+1]))

                if placeholder_values['query_x'] is not None:
                    opt_crit_maximizer_np = placeholder_values['query_x']
                    opt_crit_maximum_np = None
                
                else:

                    feed_dict = {
                        required_placeholders['X']: Xsamples_np,
                        required_placeholders['Y']: Ysamples_np - mean_f_const,
                        required_placeholders['validation_X']: validation_xs
                    }

                    for key in required_placeholder_keys[criterion]:
                        if key not in ['X', 'Y']:
                            feed_dict[ required_placeholders[key] ] = placeholder_values[key]
                    
                    rand_xs_np = functions.get_initializers(
                                xdim, 
                                xmin, xmax, 
                                name = "opt_crit", 
                                random = True, 
                                npoint = 50, 
                                seed = seed + nr + nq)
                                
                    if candidate_xs['opt_crit'] is None:
                        print("Init optimize crit with meanf_maximizer and meanf_maximizer and 50 random inputs")
                        opt_crit_candidate_xs_np = np.concatenate([rand_xs_np, placeholder_values['opt_meanf_maximizer'] ], axis=0)

                    else:
                        print("Preload the opt_crit_candidate_xs")
                        opt_crit_candidate_xs_np = candidate_xs['opt_crit']

                    print("opt_crit_candidate_xs_np.shape = {}".format(opt_crit_candidate_xs_np.shape))
                    feed_dict[ required_placeholders['opt_crit_candidate_xs'] ] = opt_crit_candidate_xs_np

                    sess.run(intermediate_tensors['opt_crit_assign'],
                        feed_dict = feed_dict)

                    for _ in range(crit_params['ntrain']):
                        sess.run(intermediate_tensors['opt_crit_train'],
                            feed_dict = feed_dict)
                    
                    opt_crit_maximizer_np, opt_crit_maximum_np \
                        = sess.run([
                        intermediate_tensors['opt_crit_maximizer'],
                        intermediate_tensors['opt_crit_maximum'] ],
                        feed_dict = feed_dict)

                query_x = opt_crit_maximizer_np.reshape(-1,xdim)
                query_f = f(query_x).reshape(-1,1) 
                query_y = query_f + np.random.randn(query_x.shape[0],1) * np.sqrt(true_sigma0)


                if 'opt_fsample_maximizers' in placeholder_values:
                    print("maximizer samples: {}".format(
                        placeholder_values['opt_fsample_maximizers']))
                    print("maxima: {}".format(placeholder_values['opt_fsample_maxima']))

                    all_maximum_samples[nr,nq,:] = placeholder_values['opt_fsample_maxima']

                print("QUERY: {}".format(query_x))
                print("end query in {:.4f}s".format(time.time() - startime_query))
                sys.stdout.flush()

                if not np.any(np.isnan(query_x)):
                    break
                else:
                    print("Repeat finding QUERY due to NAN!")

            Xsamples_np = np.concatenate([Xsamples_np, query_x], axis=0)
            Fsamples_np = np.concatenate([Fsamples_np, query_f], axis=0)
            Ysamples_np = np.concatenate([Ysamples_np, query_y], axis=0)

        candidate_xs = {
            'opt_meanf': None, # candidate_xs_to_optimize_np,
            'opt_fsample': None, # candidate_xs_to_optimize_np,
            'opt_crit': None
        }

        placeholder_values = get_placeholder_values(sess,
                    criterion, crit_params,
                    required_placeholders,
                    intermediate_tensors,
                    ls_np, sigmas_np, sigma0s_np,
                    Xsamples_np, Ysamples_np - mean_f_const,
                    candidate_xs,
                    # for computing log loss
                    validation_xs, # (m,xdim) np array
                    validation_fs, # (m,1) np array
                    true_level, # scalar 
                    # # #
                    init_random,
                    init_npoint,
                    init_seed = seed + nr + nq,
                    previous_opt_meanf_maximizer = previous_opt_meanf_maximizer,
                    iteration=nquery,
                    dtype=dtype,
                    is_debug_mode=False)

        print("meanf maximizes at {} ({})".format(
                    placeholder_values['opt_meanf_maximizer'], 
                    placeholder_values['opt_meanf_maximum']))

        print("Logloss: {}".format(placeholder_values['logloss']))
        log_losses[nr,nquery] = placeholder_values['logloss']
        print("All logloss: {}".format(log_losses[nr,...]))

        all_xx[nr,...] = Xsamples_np
        all_ff[nr,...] = Fsamples_np.squeeze()
        all_yy[nr,...] = Ysamples_np.squeeze()

        np.save('{}/{}_xx.npy'.format(folder, criterion), all_xx)
        np.save('{}/{}_ff.npy'.format(folder, criterion), all_ff)
        np.save('{}/{}_yy.npy'.format(folder, criterion), all_yy)
        np.save('{}/{}_loglosses.npy'.format(folder, criterion), log_losses)
        np.save('{}/{}_maxima.npy'.format(folder, criterion), all_maximum_samples)
