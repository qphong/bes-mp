import numpy as np 
import scipy.stats as scst
import scipy.optimize as scopt

import utils


def maximize_func(xdim, f, xs, xmin, xmax):

    negf = lambda x: -f(x.reshape(-1,xdim))
    fs = negf(xs).reshape(-1,)

    x0 = xs[np.argmax(-fs)].reshape(-1,xdim)

    res = scopt.minimize(fun=negf, 
                x0=x0, 
                method='L-BFGS-B', 
                bounds=[(xmin, xmax)]*xdim)

    print("Maximize result: {}".format(res))
    print("bound = {} {}".format(xmin, xmax))

    maximum = -res.fun 
    maximizer = res.x.squeeze()

    res = scopt.minimize(fun=f, 
                x0=x0, 
                method='L-BFGS-B', 
                bounds=[(xmin, xmax)]*xdim)

    minimum = res.fun
    minimizer = res.x.squeeze()

    print("maximum value: {} at {}".format(maximum, maximizer))
    print("minimum value: {} at {}".format(minimum, minimizer))
    
    return maximizer, maximum, minimizer,minimum



def get_gphyp_gpy(X, Y, noise_var=None, train_noise_var=True, max_iters=500):
    # use gpflow to get the hyperparameters for the function

    try:
        import GPy
    except:
        raise Exception("Requires gpflow!")

    xdim = X.shape[1]

    kernel = GPy.kern.RBF(input_dim=xdim, variance=1., lengthscale=np.ones(xdim), ARD=True)
    meanf = GPy.mappings.Constant(input_dim=xdim, output_dim=1, value=0.0)

    if train_noise_var:
        m = GPy.models.GPRegression(X, Y, kernel=kernel, mean_function=meanf)
    elif noise_var is not None:
        m = GPy.models.GPRegression(X, Y, kernel=kernel, mean_function=meanf, noise_var=noise_var)
        m.Gaussian_noise.variance.fix() # unfix()
    else:
        raise Exception("functions.py get_gphyp_gpy:Require noise variance!")

    try:
        m.optimize(max_iters=max_iters)
    except:
        return None, None, None, None


    gpy_lscale = m.rbf.lengthscale.values
    gpy_signal_var = m.rbf.variance.values
    lscale = 1.0 / (gpy_lscale * gpy_lscale)
    mean_f_const = m.constmap.C.values

    # print("Mean: {}".format(mean_f_const))
    # print("Kernel: sigvar {}, lscale {}".format(gpy_signal_var, lscale))
    # print("Gaussian_noise variance: {}".format(m.Gaussian_noise.variance.values))

    return mean_f_const, gpy_signal_var, lscale, m.Gaussian_noise.variance



def get_meshgrid(xmin, xmax, nx, xdim):
    x1d = np.linspace(xmin, xmax, nx)
    vals = [x1d] * xdim
    xds = np.meshgrid(*vals)

    xs = np.concatenate([xd.reshape(-1,1) for xd in xds], axis=1)
    return xs


def get_initializers(dim, xmin, xmax, name, random=False, randomize_method=None, npoint=20, seed=None):
    """
    if random == True:
        npoint is the total number of random initializers
    else:
        npoint is the number of partition per dimension
        the total number of initializers: npoint^dim
    """

    if random:
        print("Generating {} random initializers for {}".format(npoint, name))
        print("     random seed = {}".format(seed))

        """
        randomize_method['type'] in ['uniform', 'guass']
        if randomize_method['type'] == 'gauss', need to specify:
            randomize_method['mean'] (dim,)
            randomize_method['std'] (dim,)
            truncated within the range [xmin, xmax]
        """
        if seed is not None:
            np.random.seed(seed)

        if randomize_method is None or randomize_method['type'] == 'uniform':
            xs = np.random.rand(npoint, dim) * (xmax - xmin) + xmin
        elif randomize_method['type'] == 'gauss':
            mean_x = randomize_method['mean']
            std_x = randomize_method['std']
            
            xs = np.zeros([npoint,dim])

            for i in range(dim):
                m = mean_x[i]
                s = std_x[i]
                xs[:,i] = scst.truncnorm.rvs(a=(xmin-m)/s, b=(xmax-m)/s, loc=m, scale=s, size=npoint)
        else:
            raise Exception("Unknown randomize method type!")
    else:
        print("Generating a meshgrid for initializers for {}".format(name))
        print("   npoint_per_dim = {}".format(npoint))

        xs = get_meshgrid(xmin, xmax, npoint, dim)

    return xs


def get_info(func_name):
    f_info = globals()[func_name]()

    return f_info['xdim'], f_info['xmin'], f_info['xmax'], f_info['xs'],\
        f_info['noise.variance'], f_info['RBF.variance'], f_info['RBF.lengthscale'], \
        f_info['maximizer'], f_info['maximum']


def call_func(x, func_name, log_noise_std):
    f_info = globals()[func_name]()#func_1d_4modes()
    f = f_info['function']
    xdim = f_info['xdim']

    x = np.array(x).reshape(-1,xdim)
    n = x.shape[0]
    return f(x).squeeze() + np.random.randn(n) * np.exp(log_noise_std)


def func_gp_prior(xdim, l, sigma, seed):
    np.random.seed(seed)

    n_feats = 10000
    l = np.ones([1,xdim]) * l
    W = np.random.randn(n_feats, xdim) * np.tile( np.sqrt(l), (n_feats,1) )
    b = 2. * np.pi * np.random.rand(n_feats,1)
    theta = np.random.randn(n_feats,1)

    def f(x):
        x = np.array(x).reshape(-1,xdim)
        return ( theta.T.dot( np.sqrt(2. * sigma / n_feats) ).dot( np.cos(W.dot(x.T) 
                    + np.tile(b, (1,x.shape[0])) )) ).squeeze()

    return f


def func_1d_4modes():
    xdim = 1
    xmin = 0.
    xmax = 10.
    seed = 1
    l = 1.0
    sigma = 2.0

    xs = np.linspace(xmin, xmax, 400).reshape(-1,1)
    f = func_gp_prior(xdim, l, sigma, seed)

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            # initializer
            'init_random': False,
            'npoint_per_dim': 20,
            ############
            'noise.variance': 0.0001,
            'RBF.variance': sigma, # 3.5855
            'RBF.lengthscale': np.array([l]), # 0.8121
            'maximizer': 2.518537,
            'maximum': 1.95724434}


def func_2d_largels():
    xdim = 2
    xmin = 0.
    xmax = 1.
    seed = 2
    l = 9.0
    sigma = 1.0

    xs = get_meshgrid(xmin, xmax, 50, xdim)

    f = func_gp_prior(xdim, l, sigma, seed)
    
    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            # initializer
            'init_random': False,
            'npoint_per_dim': 20,
            ############
            'noise.variance': 0.0001,
            'RBF.variance': sigma,
            'RBF.lengthscale': np.array([l] * xdim),
            'maximizer': np.array([0.87298294, 0.59436789]),
            'maximum': 2.302762}


def func_2d_smallls():
    xdim = 2
    xmin = 0.
    xmax = 1.
    seed = 2
    l = 64.0
    sigma = 1.0

    xs = get_meshgrid(xmin, xmax, 50, xdim)

    f = func_gp_prior(xdim, l, sigma, seed)

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            # initializer
            'init_random': False,
            'npoint_per_dim': 20,
            ############
            'noise.variance': 0.0001,
            'RBF.variance': sigma,
            'RBF.lengthscale': np.array([l] * xdim),
            'maximizer': np.array([0.23135697, 0.44321898]),
            'maximum': 2.34468903}


def log10P():
    xdim = 2
    xmin = 0.
    xmax = 1.

    X = np.loadtxt('bbarn/X_log10P.txt')
    Y = np.loadtxt('bbarn/Y_log10P.txt')
    hypers = np.loadtxt('bbarn/hyperparameters_log10P.txt')

    sigma = hypers[0]
    lengthscales = hypers[1:3]
    sigma0 = hypers[3]

    xs = get_meshgrid(xmin, xmax, 20, xdim)

    def f(x):
        x = x.reshape(-1,xdim)

        vals = utils.compute_mean_f_np(x, X, Y, lengthscales, sigma, sigma0)
        return vals

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            # initializer
            'init_random': False,
            'npoint_per_dim': 20,
            ############
            'noise.variance': sigma0,
            'RBF.variance': sigma,
            'RBF.lengthscale': lengthscales,
            'maximizer': np.array([0.68527863, 0.06831491]),
            'maximum': 1.09202265}


def negative_hartmann3d():
    # xdim = 3
    # range: (0,1) for all dimensions
    # global maximum: -3.86278 at (0.114614, 0.555649, 0.852547)
    xdim = 3
    xmin = 0.
    xmax = 1.
    maximum = 3.86277979
    minimum = 0.00027354

    xs = get_meshgrid(xmin, xmax, 10, xdim)
    # xs = np.random.rand(2000, xdim) * (xmax - xmin) + xmin

    A = np.array([
            [3., 10., 30.],
            [0.1, 10., 35.],
            [3., 10., 30.],
            [0.1, 10., 35.]
        ])

    alpha = np.array([1., 1.2, 3., 3.2])

    P = 1e-4 * np.array([
            [3689., 1170., 2673.],
            [4699., 4387., 7470.],
            [1091., 8732., 5547.],
            [381., 5743., 8828.]
        ])


    def f(x):
        x = np.tile(x.reshape(-1,1,xdim), reps=(1,4,1))
        val = np.sum(alpha * np.exp(- np.sum(A * (x - P)**2, axis=2)), axis=1)
        val = (val - minimum) / (maximum - minimum) - 0.06778382075008364
        return val

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            # initializer
            'init_random': True,
            'npoint_per_dim': 20,
            ############
            'noise.variance': 0.0001,
            'RBF.variance': 0.031146723800158257,
            'RBF.lengthscale': np.array([ 2.36555227, 10.97620631, 33.26062591]),
            'maximizer': np.array([0.11458923, 0.55564889, 0.85254695]),
            'maximum': 0.93221618}


def negative_Branin():
    xdim = 2
    xmin = 0.
    xmax = 1.
    maximum = 1.04739389
    minimum = -4.87620974

    xs = get_meshgrid(xmin, xmax, 20, xdim)
    # xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    def f(x):
        x = x.reshape(-1,xdim)
        x = 15. * x - np.array([5., 0.])

        val = -1.0 / 51.95 * (
            (x[:,1] - 5.1 * x[:,0]**2 / (4*np.pi**2) + 5.*x[:,0] / np.pi - 6.)**2
            + (10. - 10. / (8.*np.pi)) * np.cos(x[:,0])
            - 44.81
        )

        val = (val - minimum) / (maximum - minimum) + 0.5956139289792839
        return val


    return {'function': f,
                'xdim': xdim,
                'xmin': xmin,
                'xmax': xmax,
                'xs': xs,
                # initializer
                'init_random': False,
                'npoint_per_dim': 20,
                ############
                'noise.variance': 0.0001,
                'RBF.variance': 1.5294688560240726,
                'RBF.lengthscale': np.array([12.14689435,  0.3134626]),
                'maximizer': np.array([0.9616514,  0.16500012]),
                'maximum': 1.6}


def negative_Goldstein():
    xdim = 2
    xmin = 0.
    xmax = 1.
    maximum = 2.18038839
    minimum = -0.33341016

    xs = get_meshgrid(xmin, xmax, 30, xdim)
    # xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    def f(x):
        x = x.reshape(-1,xdim)
        xb = x * 4. - 2. 

        val = - (
            np.log(
                (
                    1 
                    + (xb[:,0] + xb[:,1] + 1.)**2
                    * (19 - 14 * x[:,0] + 3 * x[:,0]**2 - 14 * x[:,1] + 6 * x[:,0] * x[:,1] + 3 * x[:,1]**2)
                )
                * (
                    30 
                    + (2 * x[:,0] - 3 * x[:,1])**2 
                    * (18 - 32 * x[:,0] + 12 * x[:,0]**2 + 48 * x[:,1] - 36 * x[:,0] * x[:,1] + 27 * x[:,1]**2)
                )
                ) - 8.693
            ) / 2.427

        val = (val - minimum) / (maximum - minimum) - 0.39980307675624344
        return val

        
    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            # initializer
            'init_random': False,
            'npoint_per_dim': 30,
            ############
            'noise.variance': 0.0001,
            'RBF.variance': 0.02584212360067521,
            'RBF.lengthscale': np.array([81.1012626,  83.22416009]),
            'maximizer': np.array([0.45, 0.30]),
            'maximum': 0.61}


def negative_michaelwicz():
    # constraint to [0., np.pi]
    # rescaled output to range [-0.5,0.5] BEFORE shifted to mean = 0
    xdim = 2
    xmin = 0.
    xmax = 1
    maximum = 1.82104368
    minimum = 0.0

    xs = get_meshgrid(xmin, xmax, 30, xdim)
    # xs = np.random.rand(1200, xdim) * (xmax - xmin) + xmin
    arr = np.array([[1., 2.]])

    def f(x):
        x = x.reshape(-1,xdim)
        x = x * np.pi

        val = np.sum( 
                np.sin(x) 
                * np.power(np.sin( arr * x * x / np.pi ), 4),
                axis = 1)
        
        return val - 0.27307129841631705

        
    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            # initializer
            'init_random': False,
            'npoint_per_dim': 20,
            ############
            'noise.variance': 0.0001,
            'RBF.variance': 0.13453541615735948,
            'RBF.lengthscale': np.array([39.47015387, 140.49721557]),
            'maximizer': np.array([0.68040595, 0.5]),
            'maximum': 1.54798}

