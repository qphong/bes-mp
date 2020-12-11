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
import evaluate_interval_rmes




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
    
    ymaxs, # (nhyp, nmax)
    invKs, 
    nsamples=100, 
    dtype=tf.float32):
   
   return evaluate_interval_rmes.interval_rmes(x, 
                    ls, sigmas, sigma0s, 
                    X, Y, 

                    xdim, n_hyp, 
                    
                    ymaxs, 
                    invKs, 
                    nsamples, 
                    dtype)
