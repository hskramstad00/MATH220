'''
Tar utgangspunkt i modellen for alkoholinntak fra kap. 2.3

ds/dt = -k_a * S
dB/dt = k_a * S - V_max*B/K_m+B

Bruk paramterveridene for "mann 75 kg" fra notatene.

Intialinnntak S(0) = 35 gram (2 halvlitere pils). B(0) =0
Absorpsjonsrate k_a = 2.8 time^-1
V_max = 7.35 gram/time
K_m = 4.9 gram
Volum for fordeling V_dist = 49 liter
'''

# Oppgave 7. Implementer modellen ved hjelp av scipy.integrate.solve_ivp
# Simuler over et tidsintervall på 10 timer

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Define constants
V_max = 7.35                    # gram/time
k_a = 2.8                       # time^-1
K_m = 4.9                       # gram
V_dist = 49                     # liter

# Define systems of ODEs
def alchohol_breakdown(t, y):
    S, B = y

    # Rate of change
    dS = -k_a * S
    dB = k_a * S - (V_max * B) / (K_m + B)

    return [dS, dB]

# Set intial conditions
S0 = 35                         # gram
B0 = 0
intial_state = [S0, B0]

# Set time span (hours)
t_span = (0, 10)                # simulate for 10 hours

# specific time points to evulate the soultion at
t_eval = np.linspace(t_span[0], t_span[1], 100)    

solution = solve_ivp(
    alchohol_breakdown,
    t_span,
    intial_state,
    t_eval=t_eval
)

# Plot results
if solution.success:
    t = solution.t
    S = solution.y[0]
    B = solution.y[1]

    plt.figure(figsize=(8,5))
    plt.plot(t, S, 'b-', label='B(t) - alkohol i blodet')

    # Oppgave 8. Plott S(t) og B(t) i samme figur
    plt.plot(t,B, 'g-', label='S(t) - alkohol i mage/tarm')

    plt.xlabel('Tid (timer)')
    plt.ylabel('Mengde alkohol (gram)')
    plt.title('Solution of alchol breakdown')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Oppgave 9. Plott som viser promillen som funksjon av tid

else:
    print(f'Solver terminated with status: {solution.message}')