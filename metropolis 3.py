# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:16:27 2023

@author: Juan4
"""

import numpy as np
import matplotlib.pyplot as plt

def normal(x, m, s):
    uno = (1 / np.sqrt(2*np.pi*(s**2)))
    dos = np.exp((-1/2)*((x-m)/s)**2)
    return uno*dos

x_ = np.linspace(0,4,1000)
m = 2
r = 2
s = 0.5
n = 0.5

plt.plot(x_, normal(x_,m,s))

def Metropolis(x0, Posterior, NSteps=int(1e4), delta= 0.4):
    
    x = np.zeros((NSteps,1))
    
    # Prior
    x[0] = x0
    
    for i in range(1,NSteps):
        
        P0 = Posterior(x[i-1],r,n)
        
        xf = x[i-1] + delta*2*(np.random.rand()-0.5)
        
        P1 = Posterior(xf,r,n)
        
        alpha = np.minimum( 1, P1/P0 )
        g = np.random.rand()
        
        if alpha > g:
            x[i,0] = xf
        else:
            x[i,:] = x[i-1,:]
            
    return x[1000:,:]

initparams = np.array([0])
MCMC = Metropolis(initparams,normal)

plt.hist(MCMC,density=True,bins=50)
