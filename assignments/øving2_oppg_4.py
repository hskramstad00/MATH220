# Lotka-Volterra

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# params
a, b, c, d = 1.0, 0.1, 1.5, 0.075

def lotka_volterra(t, z):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

# start values
z0 = [10, 5]
t_span = (0, 100)
t_eval = np.linspace(0, 100, 2000)

sol = solve_ivp(lotka_volterra, t_span, z0, dense_output=True)

solvals = sol.sol(t_eval)

# plot i faseplan
plt.figure(figsize=(8,6))
plt.plot(solvals[0], solvals[1])
plt.xlabel('Byttedyr (x)')
plt.ylabel('Rovdyr (y)')
plt.title('Lotka-voltera')
plt.grid(True)

# sjekk bevairingslov
def H(x, y):
    return d*x - c*np.log(x) + b*y - a*np.log(y)

h_values = H(solvals[0], solvals[1])
plt.figure()
plt.plot(t_eval, h_values, label='H')
plt.title('Bevaringslov')
plt.legend(loc='center right')
plt.show()

H0 = h_values[0]
print("max avvik:", np.max(np.abs(h_values - H0)))
print("range:", np.max(h_values) - np.min(h_values))