import sys
sys.path.insert(0, './criteria')
sys.path.insert(0, './criteria/mes_criteria')

import os
import argparse


DEBUG = False
use_GPU_for_sample_functions = False

parser = argparse.ArgumentParser(description='Level set estimation')
parser.add_argument('-g', '--gpu', help='gpu device index for tensorflow',
                    required=False,
                    type=str,
                    default='0')
parser.add_argument('-c', '--criterion', help='BO acquisition function',
                    required=False,
                    type=str,
                    default='bes')
parser.add_argument('-n', '--noisevar', help='noise variance',
                    required=False,
                    type=float,
                    default=0.09)
parser.add_argument('-q', '--numqueries', help='number/budget of queries',
                    required=False,
                    type=int,
                    default=20)
parser.add_argument('-r', '--numruns', help='number of random experiments',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-s', '--numhyps', help='number of sampled hyperparameters',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-a', '--nparal', help='number of parallel iterations',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-y', '--nysample', help='number of y samples to evaluate acquisition',
                    required=False,
                    type=int,
                    default=5)
parser.add_argument('--ntrain', help='number of optimizing iterations',
                    required=False,
                    type=int,
                    default=100)
parser.add_argument('--ninit', help='number of initial observations',
                    required=False,
                    type=int,
                    default=2)
parser.add_argument('--function', help='ground truth function',
                    required=False,
                    type=str,
                    default='func_2d_largels_rmes')
parser.add_argument('--level', help='level value to search for (i.e., threshold)',
                    required=False,
                    type=float,
                    default=0.0)
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


folder = "level_set_estimation"

if not os.path.exists(folder):
    os.makedirs(folder)

folder = "{}/{}".format(folder, args.function)

if not os.path.exists(folder):
    os.makedirs(folder)

folder = "{}/noise_var_{}".format(folder, args.noisevar)
if not os.path.exists(folder):
    os.makedirs(folder)
    
folder = '{}/{}'.format(folder, args.criterion)
if not os.path.exists(folder):
    os.makedirs(folder)


level = args.level
nquery = args.numqueries
nrun = args.numruns
nhyp = args.numhyps

nysample = args.nysample
parallel_iterations = args.nparal

ntrain = args.ntrain
n_initial_training_x = args.ninit
func_name = args.function

print("nrun: {}".format(nrun))
print("nquery: {}".format(nquery))
print("nhyp: {}".format(nhyp))
print("n_initial_training_x: {}".format(n_initial_training_x))
print("Function: {}".format(func_name))
print("Level: {}".format(level))

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
import evaluate_interval_rmes
import evaluate_dare 
import evaluate_straddle


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
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


def evaluate_criterion(xs, nx,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    level,
                    required_placeholders,
                    dtype=tf.float32):
    xdim = crit_params['xdim']
    nhyp = crit_params['nhyp']
    
    Xsamples = required_placeholders['X']
    Ysamples = required_placeholders['Y']
    nobs = tf.shape(Xsamples)[0]
    
    if criterion == 'bes':
        invKs = required_placeholders['invKs']

        # marginal version
        vals = evaluate_interval_rmes.interval_rmes(xs,
                    ls, sigmas, sigma0s,
                    Xsamples, Ysamples,  
                    
                    xdim, nhyp, 
                    
                    tf.ones(shape=(1,1), dtype=dtype) * tf.constant(level, dtype=dtype),
                    invKs, 
                    nsamples=crit_params['nysample'],
                    dtype=dtype)

    elif criterion == 'dare':
        invKs = required_placeholders['invKs']

        vals = evaluate_dare.dare(xs,
            ls, sigmas, sigma0s,
            Xsamples, Ysamples,  
            
            xdim, nhyp, 
            
            tf.ones(shape=(1,1), dtype=dtype) * tf.constant(level, dtype=dtype),  
            invKs, 
            dtype=dtype)

    elif criterion == 'straddle':
        invKs = required_placeholders['invKs']

        vals = evaluate_straddle.straddle(xs,
            ls, sigmas, sigma0s,
            Xsamples, Ysamples,  
            
            xdim, nhyp, 
            
            tf.ones(shape=(1,1), dtype=dtype) * tf.constant(level, dtype=dtype),  
            invKs, 
            dtype=dtype)

    else:
        raise Exception("Unknown criterion: {}".format(criterion))

    return vals


def get_required_placeholders(criterion, crit_params,
                dtype,
                is_debug_mode = False):

    nhyp = crit_params['nhyp']
    xdim = crit_params['xdim']

    parallel_iterations = crit_params['parallel_iterations']

    X_plc = tf.placeholder(dtype=dtype, shape=(None, xdim), name='X_plc')
    Y_plc = tf.placeholder(dtype=dtype, shape=(None, 1), name='Y_plc')
    validation_X_plc = tf.placeholder(dtype=dtype, shape=(None, xdim), name='validation_X_plc')

    invKs_plc = tf.placeholder(shape=(nhyp,None,None), dtype=dtype, name='invKs_plc')
    # (nhyp, nobs, nobs)

    opt_crit_candidate_xs_plc = tf.placeholder(dtype = dtype,
                            shape = (None,xdim),
                            name = 'opt_crit_candidate_xs')
    
    required_placeholders = {
        'X': X_plc,
        'Y': Y_plc,
        'validation_X': validation_X_plc,
        'opt_crit_candidate_xs': opt_crit_candidate_xs_plc,
        'invKs': invKs_plc
    }

    return required_placeholders


def get_intermediate_tensors(criterion, crit_params,
            required_placeholders,
            ls, sigmas, sigma0s,
            level,
            dtype,
            is_debug_mode=False):

    xdim = crit_params['xdim']
    nhyp = crit_params['nhyp']
    xmin = crit_params['xmin']
    xmax = crit_params['xmax']
    opt_crit_top_init_k = crit_params['opt_crit_top_init_k']

    X_plc = required_placeholders['X']
    Y_plc = required_placeholders['Y']

    intermediate_tensors = {}

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


    # optimize acquisition function
    opt_crit_func = lambda x: evaluate_criterion(
                    tf.reshape(x, shape=(-1,xdim)), 
                    1,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    level,
                    required_placeholders,
                    dtype=dtype)

    opt_crit_multiple_func = lambda x: evaluate_criterion(
                    tf.reshape(x, shape=(-1,xdim)), 
                    opt_crit_top_init_k,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    level,
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
        iteration=1,
        dtype=tf.float32,
        is_debug_mode=False):

    xdim = crit_params['xdim']
    ntrain = crit_params['ntrain']
    xmin = crit_params['xmin']
    xmax = crit_params['xmax']

    values = {
            'iteration': iteration,
            'query_x': None}

    if 'invKs' in intermediate_tensors:
        invKs_np = sess.run(intermediate_tensors['invKs'], 
            feed_dict = {
                required_placeholders['X']: X_np
            })
        values['invKs'] = invKs_np
    

    # evaluating the log loss function
    meanf_val, stdf_val = sess.run([intermediate_tensors['meanf'], 
                                    intermediate_tensors['stdf']],
                        feed_dict = {
                            required_placeholders['X']: X_np,
                            required_placeholders['Y']: Y_np,
                            required_placeholders['validation_X']: validation_xs
                        })

    sign = (level - validation_fs) / np.abs(validation_fs - level)
    
    sign = sign.reshape(-1,)
    meanf_val = meanf_val.reshape(-1,)
    stdf_val = stdf_val.reshape(-1,)

    logloss = - np.mean( spst.norm.logcdf(sign * (level - meanf_val) / stdf_val) )
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


init_random = f_info['init_random']
if init_random:
    init_npoint = 1000
else:  
    init_npoint = f_info['npoint_per_dim']

xdim = f_info['xdim']


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


true_l = f_info['RBF.lengthscale']
true_sigma = f_info['RBF.variance']
true_sigma0 = f_info['noise.variance']
true_sigma0 = args.noisevar
print("Override true_sigma0 = {}".format(true_sigma0))

ls_np = true_l.reshape(-1,xdim).astype(nptype)
sigmas_np = np.array([ true_sigma ], dtype=nptype)
sigma0s_np = np.array([ true_sigma0 ], dtype=nptype)

seed = 1

print("True GP hyperparameters: l:{} sigma:{} sigma0(noise var):{}".format(true_l, true_sigma, true_sigma0))
print("nhyp: {}".format(nhyp))
print("nrun:{}, nqueries:{}".format(nrun, nquery))
print("____________________________________________")



ls_toload = tf.get_variable(dtype=dtype, shape=(nhyp,xdim), name='ls')
sigmas_toload = tf.get_variable(dtype=dtype, shape=(nhyp,), name='sigmas') 
sigma0s_toload = tf.get_variable(dtype=dtype, shape=(nhyp,), name='sigma0s')

crit_params = {'nhyp': nhyp,
               'xdim': xdim,

               'xmin': xmin,
               'xmax': xmax,

               'nysample': nysample,
               'parallel_iterations': parallel_iterations,

               'opt_crit_top_init_k': 5,
               
               'ntrain': ntrain,
               'n_min_sample': 2}

print("crit_params: {}".format(crit_params))


required_placeholder_keys = {
    'bes': ['X', 'Y', 'invKs'],
    'dare': ['X', 'Y', 'invKs'],
    'straddle': ['X', 'Y', 'invKs']
}


required_placeholders = get_required_placeholders(criterion, crit_params, dtype, is_debug_mode=False)

intermediate_tensors = get_intermediate_tensors(criterion, crit_params,
            required_placeholders,
            ls_toload, sigmas_toload, sigma0s_toload,
            level,
            dtype,
            is_debug_mode=False)

log_losses = np.zeros([nrun, nquery+1])

all_xx = np.zeros([nrun, nquery + n_initial_training_x, xdim])
all_ff = np.zeros([nrun, nquery + n_initial_training_x]) 
all_yy = np.zeros([nrun, nquery + n_initial_training_x])


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

        for nq in range(nquery):

            startime_query = time.time()

            # for randomly drawing different functions
            sess.run(tf.global_variables_initializer())

            print("")
            print("{}:{}.=================".format(nr, nq))
            print("  X: {}".format(Xsamples_np.T))
            print("  Y: {}".format(Ysamples_np.T))

            ls_toload.load(ls_np, sess)
            sigmas_toload.load(sigmas_np, sess)
            sigma0s_toload.load(sigma0s_np, sess)
            print("")

            candidate_xs = {
                'opt_crit': None # candidate_xs_to_optimize_np
            }
            
            while True:
                # repeat if query_x is nan
                placeholder_values = get_placeholder_values(sess,
                            criterion, crit_params,
                            required_placeholders,
                            intermediate_tensors,
                            
                            ls_np, sigmas_np, sigma0s_np,
                            Xsamples_np, Ysamples_np,
                            candidate_xs,

                            # for computing log loss
                            validation_xs, # (m,xdim) np array
                            validation_fs, # (m,1) np array
                            level, # scalar 
                            # # #

                            init_random,
                            init_npoint,
                            init_seed = seed + nr + nq,
                            
                            iteration=nq+1,
                            dtype=dtype,
                            is_debug_mode=False)
                
                print("Logloss: {}".format(placeholder_values['logloss']))
                log_losses[nr,nq] = placeholder_values['logloss']
                print("All logloss: {}".format(log_losses[nr,:nq+1]))

                feed_dict = {
                    required_placeholders['X']: Xsamples_np,
                    required_placeholders['Y']: Ysamples_np,
                    required_placeholders['validation_X']: validation_xs
                }

                for key in required_placeholder_keys[criterion]:
                    if key not in ['X', 'Y']:
                        feed_dict[ required_placeholders[key] ] = placeholder_values[key]
                

                if candidate_xs['opt_crit'] is None:
                    print("Init optimize crit with f_info initializers")
                    opt_crit_candidate_xs_np = functions.get_initializers(
                                xdim, 
                                xmin, xmax, 
                                name = "opt_crit", 
                                random = init_random, 
                                npoint = init_npoint, 
                                seed = seed + nr + nq)
                else:
                    print("Preload the opt_crit_candidate_xs")
                    opt_crit_candidate_xs_np = candidate_xs['opt_crit']

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
            'opt_crit': None
        }

        placeholder_values = get_placeholder_values(sess,
                    criterion, crit_params,
                    required_placeholders,
                    intermediate_tensors,
                    
                    ls_np, sigmas_np, sigma0s_np,
                    Xsamples_np, Ysamples_np,
                    candidate_xs,

                    # for computing log loss
                    validation_xs, # (m,xdim) np array
                    validation_fs, # (m,1) np array
                    level, # scalar 
                    # # #
                    
                    init_random,
                    init_npoint,
                    init_seed = seed + nr + nq,

                    iteration=nquery,
                    dtype=dtype,
                    is_debug_mode=False)

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









