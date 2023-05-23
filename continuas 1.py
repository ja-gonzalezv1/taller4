# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:52:09 2023

@author: Juan4
"""

import sympy as sym

x = sym.Symbol("x", real=True)
y = sym.Symbol("y", real=True) 

f = (2/3) * (x + 2*y)

Fx= sym.integrate(f, (x , 0, 1))
Fy = sym.integrate(Fx, (y, 0, 1))

print(Fy)

# La integral da 1, por lo que si es una funcion valida

g = sym.integrate(f, (y, 0, 1))
print(g)
Ex = sym.integrate((x*g), (x, 0, 1))
print(Ex)

h = sym.integrate(f, (x, 0, 1))
print(h)
Ey = sym.integrate(y*h, (y, 0, 1))
print(Ey)

Efx= sym.integrate(x*f, (x , 0, 1))
Efxy = sym.integrate(y*Efx, (y, 0, 1))

cov1= Efxy - (Ex*Ey)
print(cov1)


fc = (2/3) * ((x - Ex) + 2*(y - Ey))

Efcx= sym.integrate(((x - Ex)*fc), (x , 0, 1))
cov2 = sym.integrate(((y - Ey)*Efx), (y, 0, 1))

print(cov2)


f_ = sym.lambdify([x,y], f)
g_ = sym.lambdify(x, g)
h_ = sym.lambdify(y, h)

print(f_(1,1))
print((h_(1))*(g_(1)))

# No son independientes, ya que f(x,y) no es igual a g(x)*h(y) 


