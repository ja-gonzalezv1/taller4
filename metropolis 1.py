# -*- coding: utf-8 -*-
"""
Created on Mon May 22 07:03:04 2023

@author: Juan4
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate



def Prior(p):
    return np.piecewise( p, [p>= 0 and p <= 1, p<0 and p > 1], [lambda p: 1, lambda p:0])

Prior = np.vectorize(Prior)

def Likelihood(p,r,n):
    return p**r*(1-p)**(n-r)

def Posterior(p,r,n):
    return Likelihood(p,r,n)*Prior(p)

p = np.linspace(0,1,100)
r = 7
n = 10
Pos = Posterior(p,r,n)

I,_ = integrate.quad(Posterior,p[0],p[-1],args=(r,n))


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

initparams = np.array([0.1])
MCMC = Metropolis(initparams,Posterior)

plt.hist(MCMC,density=True,bins=50)
plt.plot(p,Pos/I)

print(max(Pos/I), np.argmax(Pos/I))

p_= 0.7
q= 0.3

var = n*p_*q
std = np.sqrt(var)

print(std)

print(np.percentile(MCMC,68))

# Si esta truncada, ya que el valor devprobabilidad maxima esta en 0.69 y no en 0.5. Habria que hacer mas lanzamientos para estar seguros





