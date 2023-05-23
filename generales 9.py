# -*- coding: utf-8 -*-
"""
Created on Sat May 20 06:47:28 2023

@author: Juan4
"""

import numpy as np
import matplotlib.pyplot as plt

# Hay 6 casos posibles en los que salen dos caras y dos sellos. De estos 6, en 3 la primera modeda es cara, y en los otros 3 es sello.
# La posibilidad de que la primera salga cara es p1, las probabilidades del resto de monedas es 0.5. Asi, p1*0.5*0.5*05= 0.125p1
# De igual manera, para que salga sello es 1-p1. Cuando la primera sale sello, la probabilidad conjunta es (1-p1)*0.5*0.5*0.5 = 0.125(1-p1)
# La probabilidad total de tener 2 caras y 2 sellos seria: 3*0.125*p1 + 3*0.125*(1-p1) = 3/8

# Con monedad 1 y 2 trucadas: p1*p2*0.25 + 2*p1*(1-p2)*0.25 + 2*(1-p1)*p2*0.25 + (1-p1)*(1-p2)*0.25


def pro(x,y):
    return -0.5*x*y + 0.25*x + 0.25*y + 0.25

p1 = np.linspace(0.1,0.9,100)
p2 = np.linspace(0.1, 0.5, 100)
P1, P2 = np.meshgrid(p1,p2)

PT = pro(P1, P2)

fig = plt.figure()
ax= plt.axes(projection= "3d")
ax.plot_surface(P1, P2, PT)
ax.view_init(10, 35)

max_ = 0
min_ = 10000000
for i in p1:
    for j in p2:
        pt = pro(i, j)
        if pt > max_ :
            max_ = pt
            maxs = i,j
        if pt < min_ :
            min_ = pt
            mins = i,j 
            
print(maxs, mins)
            

