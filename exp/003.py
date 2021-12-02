import sympy as sp
import numpy as np

# import utils.utils as ut

###############################################################
# a, b, l = sp.symbols('a b lambda', real=True)
# f = 5 * a * b - a * sp.cos(l) + a ** 2 + l ** 8 * b
#
# # differentiating function f in respect to a
# print(sp.diff(f, l))
###############################################################

# ut.sigmoid_function()

# tk, ok = sp.symbols('tk ok', real=True)
#
# e = (tk - ok)**2
# print(sp.diff(e, ok))

###############################################################

x, y = sp.symbols('x y', real=True)

# print(np.power(2, x))
# fun = 1/(1 + math.pow(2,x))

fun = 1/(1+sp.exp(-x))

print(sp.diff(fun, x))
