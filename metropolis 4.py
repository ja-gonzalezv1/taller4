# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:50:27 2023

@author: Juan4
"""

import numpy as np
import matplotlib.pyplot as plt

def Cauchy (x):
    return 1/(np.pi * (1 + x**2))

x_ = np.linspace(-7,7,1000)

plt.plot(x_, Cauchy(x_))

def Metropolis(x0, Posterior, NSteps=int(1e4), delta= 0.4):
    
    x = np.zeros((NSteps,1))
    
    # Prior
    x[0] = x0
    
    for i in range(1,NSteps):
        
        P0 = Posterior(x[i-1])
        
        xf = x[i-1] + delta*2*(np.random.rand()-0.5)
        
        P1 = Posterior(xf)
        
        alpha = np.minimum( 1, P1/P0 )
        g = np.random.rand()
        
        if alpha > g:
            x[i,0] = xf
        else:
            x[i,:] = x[i-1,:]
            
    return x[1000:,:]

initparams = np.array([0])
MCMC = Metropolis(initparams,Cauchy)

plt.hist(MCMC,density=True,bins=50)