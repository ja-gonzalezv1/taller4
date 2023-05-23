# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:10:23 2023

@author: Juan4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Gaussiano.csv")

x= np.array(data.T)[0]

def Prior(m, s):
    m_ = np.piecewise( m, [m>= 3 and m <= 5 and s>= 0.5 and s <= 3.5, m<3 and m > 5 and s<0.5 and s > 3.5], [lambda m: 1, lambda m:0])
    return m_

Prior = np.vectorize(Prior)

def Like(x, m, s):
    p = 0
    for i in x:
        uno = (1 / np.sqrt(2*np.pi*(s**2)))
        dos = np.exp((-1/2)*((i-m)/s)**2)
        p += (uno*dos)
    return p

def LPosterior (x, m, s):
    return np.log(Prior(m, s) * Like(x, m, s))

N = int(2e4)
#m = np.linspace(3, 4, 100)
#s= np.linspace(0.5, 3.5, 100)
m= 3.5
s= 1.5

Pos = LPosterior(x, m, s)
print(Pos)

def Metropolis(x0, Posterior, NSteps=int(1e4), delta= 0.4):
    
    x = np.zeros((NSteps,1))
    
    # Prior
    x[0] = x0
    
    for i in range(1,NSteps):
        
        P0 = Posterior(x[i-1], m, s)
        
        xf = x[i-1] + delta*2*(np.random.rand()-0.5)
        
        P1 = Posterior(xf, m, s)
        
        alpha = np.minimum( 1, P1/P0 )
        g = np.random.rand()
        
        if alpha > g:
            x[i,0] = xf
        else:
            x[i,:] = x[i-1,:]
            
    return x[1000:,:]

initparams = np.array([0])
MCMC = Metropolis(initparams, LPosterior)
plt.plot(np.linspace(1,10,100), Pos)
plt.hist(MCMC,density=True,bins=50)