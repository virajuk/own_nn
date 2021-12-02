import sympy as sp

a, b, c = sp.symbols('a b c', real=True)
f = 5*a*b - a*sp.cos(c) + a**2 + c**8*b

# differentiating function f in respect to a
print(sp.diff(f, a))
