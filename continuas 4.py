# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:49:34 2023

@author: Juan4
"""


import sympy as sym

x = sym.Symbol("x", real= True)

f = sym.exp(-x)
g = sym.exp((2/3)*x)

fg= f*g

E = sym.integrate(fg, (x, 0, sym.oo))
print(E)

