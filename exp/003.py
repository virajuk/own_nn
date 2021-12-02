import sympy as sp
import math

###############################################################
# a, b, l = sp.symbols('a b lambda', real=True)
# f = 5 * a * b - a * sp.cos(l) + a ** 2 + l ** 8 * b
#
# # differentiating function f in respect to a
# print(sp.diff(f, l))
###############################################################


tk, ok = sp.symbols('tk ok', real=True)

e = (tk - ok)**2
print(sp.diff(e, ok))
