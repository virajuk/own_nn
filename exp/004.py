import sympy as sp

# x, y = sp.symbols('x y', real=True)
# u = sp.asin(x/y)
#
# print(sp.diff(u, x))

x, y = sp.symbols('x y', real=True)
func = sp.sin(y)

print(sp.diff(func, x))
