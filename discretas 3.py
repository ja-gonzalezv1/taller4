# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:43:50 2023

@author: Juan4
"""

import numpy as np

# Para encontrar la distrivucion de probabilidad para cada caso, hay que hacer un arbol de probabilidad
# En el caso en que no sale ningun chip defectuoso, la probabilidad de que el primero sea bueno es igual a 7/10, y la del segundo es de 6/9. (7/10)*(6/9) = (7/15)
# En el caso en que sale solo uno defectuoso, puede salir el primero bueno y el segundo malo: (7/10)*(3/9)= (7/30), o el primero malo y el sugundo bueno: (3/10)*(7/9) = 7/30. La suma de ambos casos es igual a 7/15
# Finalmente, pueden salir ambos malos: (3/10)*(2/9) = 1/15
# Asi, para los f(x): f(0)= 7/15, f(1)= 7/15, f(2) = 1/15. La suma de los 3 es igual a 1.

def BC(x, y):
    return (np.math.factorial(x)) / ((np.math.factorial(y)) * (np.math.factorial(x-y)))

def Distribucion(x):
    return ((BC(7, 2-x) * BC(3, x)) / (BC(10 , 2)))

f0 = Distribucion(0)
f1 = Distribucion(1)
f2 = Distribucion(2)

x= [0, 1, 2]
dist= [f0, f1, f2]

print(f0, f1, f2, f0+f1+f2)

valor= 0 
for i in range(len(x)):
    valor += x[i]*dist[i]

print(valor)

