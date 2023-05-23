# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:26:46 2023

@author: Juan4
"""

import numpy as np
import matplotlib.pyplot as plt

# Para la probabilidad de que no se tenga el mismo cumpleaños, se utiliza la fórmula -> n! / (n**r)(n-r)!, donde n es 365 y r es el numero de personas
# Para encontrar la probabilidad de que si se tenga el mismo cumpleaños, se resta la anterior a 1
r= np.linspace(0,80, 81)


def p(x):
    return (np.math.factorial(365) / (np.math.factorial(365 - x)))


ps = np.zeros(len(r))

for i in range(len(r)):
    pi= 1 - (p(r[i])/(365**r[i]))
    ps[i] = pi
    

plt.plot(r,ps)

