import sys
import numpy as np
import matplotlib.pyplot as plt

# Augmenter la limite de récursion pour i+k=100
sys.setrecursionlimit(2000)

def get_b_dist(i_target, k_target, eta):
    memo = {}
    eta_bar = 1 - eta
    def get_b(i, k, n):
        if n < 0 or i < 0 or k < 0 or n > (i + k): return 0.0
        if i == 0 and k == 0: return 1.0 if n == 0 else 0.0
        state = (i, k, n)
        if state in memo: return memo[state]
        if i > 0 and k > 0:
            res = (eta * get_b(i-1, k, n-1) + eta * get_b(i, k-1, n) +
                   eta_bar * get_b(i-1, k, n) + eta_bar * get_b(i, k-1, n-1) -
                   get_b(i-1, k-1, n-1))
        elif i > 0: res = eta * get_b(i-1, 0, n-1) + eta_bar * get_b(i-1, 0, n)
        elif k > 0: res = eta * get_b(0, k-1, n) + eta_bar * get_b(0, k-1, n-1)
        else: res = 0.0
        memo[state] = res
        return res
    
    N = i_target + k_target
    dist = [get_b(i_target, k_target, n) for n in range(N + 1)]
    return np.array(dist)

# --- 1. Paramètres (basés sur ton image) ---
I_VAL = 10
K_VAL = 90
ETA = 0.1
N_TOTAL = I_VAL + K_VAL
N_SHOTS = 10000

# --- 2. Simulation de l'expérience ---
# On a besoin de la théorie juste pour simuler le comportement quantique
vraie_distribution = get_b_dist(I_VAL, K_VAL, ETA)
vraie_distribution = np.maximum(vraie_distribution, 0)
vraie_distribution /= np.sum(vraie_distribution)

valeurs_n = np.arange(N_TOTAL + 1)
mesures_labo = np.random.choice(valeurs_n, size=N_SHOTS, p=vraie_distribution)

# --- 3. Traitement des données empiriques ---
# On compte combien de fois on a obtenu chaque valeur de n
frequences_exp = np.zeros(N_TOTAL + 1)
valeurs_uniques, comptes = np.unique(mesures_labo, return_counts=True)

# On transforme ces comptes en fréquences (équivalent empirique des probabilités)
frequences_exp[valeurs_uniques] = comptes / N_SHOTS

# --- 4. Tracé (Style Scatter Plot) ---
plt.figure(figsize=(10, 6))

# On utilise scatter pour faire des points, comme sur ton image
plt.scatter(valeurs_n, frequences_exp, marker='o')

plt.title(f"Fréquences expérimentales pour i={I_VAL}, k={K_VAL} et eta={ETA} ({N_SHOTS} tirs)")
plt.xlabel("n (nombre de photons en sortie)")
plt.ylabel("Fréquence d'apparition")
plt.grid(True)

plt.show()

