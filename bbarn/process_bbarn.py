
import sys
import numpy as np 
import tensorflow as tf
import gpflow

import matplotlib.pyplot as plt 

sys.path.insert(0, './..')

from functions import * 
import utils



def get_gphyp(xdim, X,Y):
    m = gpflow.models.GPR(X, Y, kern=gpflow.kernels.RBF(xdim, ARD=True))
    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)

    sig_var = m.kern.variance.value
    lengthscales = 1.0 / (m.kern.lengthscales.value * m.kern.lengthscales.value)
    noise_var = m.likelihood.variance.value

    # print(m.as_pandas_table())
    print("    Loglikelihood: {}".format(m.compute_log_likelihood()))
    print("    RBF variance = {}".format(sig_var))
    print("    RBF lengthscale = {}".format(lengthscales))
    print("    Likelihood variance = {}".format(noise_var))

    return m, sig_var, lengthscales, noise_var


xdim = 2
dtype = tf.float32
xplot_plc = tf.placeholder(dtype=dtype, shape=[None,xdim], name='xplot_plc')
X_plc = tf.placeholder(dtype=dtype, shape=[None,xdim], name='X_plc')
Y_plc = tf.placeholder(dtype=dtype, shape=[None,1], name='Y_plc')
lengthscale_plc = tf.placeholder(dtype=dtype, shape=[1,xdim], name='lengthscales')
sigma_plc = tf.placeholder(dtype=dtype, shape=(), name='sigma')
sigma0_plc = tf.placeholder(dtype=dtype, shape=(), name='sigma0')

meanf, varf = utils.compute_mean_var_f(xplot_plc, X_plc, Y_plc, lengthscale_plc, sigma_plc, sigma0_plc, NKInv=None, fullcov=False, dtype=dtype)

Kmm = utils.computeKmm_old(X_plc, lengthscale_plc, sigma_plc, dtype=dtype)
Knm = utils.computeKnm(xplot_plc, X_plc, lengthscale_plc, sigma_plc, dtype=dtype)


attrs = "East,North,potassium,log10K,pH,phosphorus,log10P".split(',')
d = np.genfromtxt('bbarn.csv', delimiter=',')
# x1: d[:,0]: 1->18
# x2: d[:,1]: 1->31

nplot = 50
X0, X1 = np.meshgrid( np.linspace(0., 1., nplot), np.linspace(0., 1., nplot) )
xplot = np.concatenate([X0.reshape(-1,1), X1.reshape(-1,1)], axis=1)

x = {}
y = {}
for i,attr in enumerate(attrs[2:]):
    if attr in ['potassium', 'phosphorus']:
        continue

    x[attr] = d[:,:2]
    y[attr] = d[:,2+i].reshape(-1,1)

    rm_idxs = np.where(y[attr] == -9)[0]
    y[attr] = np.delete(y[attr], rm_idxs, axis=0)
    x[attr] = np.delete(x[attr], rm_idxs, axis=0)

    x[attr] = (x[attr] - 1.0) / np.array([[17., 30.]])
    y[attr] = y[attr] - np.mean(y[attr])

    meany = np.mean(y[attr])
    stdy = np.std(y[attr])

    print("{}: {} rows".format(attr, x[attr].shape[0]))
    print("    y stats: mean {} std {}".format(meany, stdy))
    print("    ymax: {} ymin: {}".format(np.max(y[attr]), np.min(y[attr])))
    print("    xmax: {} xmin: {}".format(np.max(x[attr], axis=0), np.min(x[attr], axis=0)))

    m, sig_var, lengthscales, noise_var = get_gphyp(xdim=2, X=x[attr], Y=y[attr])

    hyperparameters = np.array([sig_var, lengthscales[0], lengthscales[1], noise_var])
    np.savetxt('X_{}.txt'.format(attr), x[attr])
    np.savetxt('Y_{}.txt'.format(attr), y[attr])
    np.savetxt('hyperparameters_{}.txt'.format(attr), hyperparameters)


    meany, vary = m.predict_y(xplot)
    meany_np = utils.compute_mean_f_np(xplot, x[attr], y[attr], lengthscales, sig_var, noise_var)

    Kmm_np0 = utils.computeKmm_np(x[attr], lengthscales, sig_var)
    Knm_np0 = utils.computeKnm_np(xplot, x[attr], lengthscales, sig_var)

    with tf.Session() as sess:
        meanf_np, varf_np, Kmm_np, Knm_np = sess.run([meanf, varf, Kmm, Knm], feed_dict = {
            xplot_plc: xplot,
            X_plc: x[attr],
            Y_plc: y[attr],
            sigma_plc: sig_var,
            sigma0_plc: noise_var,
            lengthscale_plc: lengthscales.reshape(1,xdim)
        })

    print("meany_tf - meany_gf")
    print(np.sum( np.abs(meanf_np.squeeze() - meany.squeeze()) ))
    print("meany_gf - meany_np")
    print(np.sum( np.abs(meany.squeeze() - meany_np.squeeze()) ))


    plt.imshow(meanf_np.reshape(nplot, nplot))
    plt.colorbar()
    plt.show()

