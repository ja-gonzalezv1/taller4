# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:08:23 2023

@author: Juan4
"""

import numpy as np

def BC(x, y):
    return (np.math.factorial(x)) / ((np.math.factorial(y)) * (np.math.factorial(x-y)))

def f(x,y):
    return (BC(3, x) * BC(2, y)* BC(3, 4-x-y)) / BC(8,4)

def g(x):
    return (BC(3, x) * BC(5, 4-x)) / BC(8,4)

def h(y):
    return (BC(2, y)* BC(6, 4-y)) / BC(8,4)

x = [0, 1, 2, 3]
gx = []
for i in x:
    gx.append(g(i))
    
y = [0, 1, 2]
hy = []
for j in y:
    hy.append(h(j))
    
"""
fxy = []
for i in x:
    for j in y:
        if i + j < 4:
            fxy.append(f(i,j))
        """
        
print(x)
print(gx)
print("------")
print(y)
print(hy)
print("------")

ex= 0 
for i in range(len(x)):
    ex += x[i]*gx[i]
print(ex)

ey= 0 
for i in range(len(y)):
    ey += y[i]*hy[i]
print(ey)

exy = 0
for i in range(len(x)):
    for j in range (len(y)):
        if i+j < 4 and i+j != 0:
            exy += i*j*f(i,j)

cov1 = exy - (ex*ey)
print(cov1)

cov2 = 0
for i in range(len(x)):
    for j in range (len(y)):
        if i+j < 4 and i+j != 0:
            cov2 += (i - ex)*(j -ey)*f((i - ex),(j - ey))
print(cov2)

# Las variables x e y no son independientes, ya que extraer un elemento de la caja cambia la probabilidad de sacar el proximo
