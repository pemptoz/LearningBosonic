import sys
import numpy as np

sys.setrecursionlimit(2000)

# --- Fonction théorique (identique) ---
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
    dist_array = np.array(dist)
    return np.maximum(dist_array, 0) / np.sum(np.maximum(dist_array, 0))

# --- Paramètres Cachés ---
I_SECRET = 20
K_SECRET = 40
N_SHOTS = 100 # Seulement 100 tirs !

print("--- DÉBUT DU PROTOCOLE HYBRIDE ---")

# ==========================================
# EXPÉRIENCE 1 : Trouver le total (eta = 0.5)
# ==========================================
ETA_1 = 0.5
dist_1 = get_b_dist(I_SECRET, K_SECRET, ETA_1)
mesures_1 = np.random.choice(np.arange(I_SECRET + K_SECRET + 1), size=N_SHOTS, p=dist_1)

# On estime N uniquement avec la moyenne
moyenne_1 = np.mean(mesures_1)
N_estime = int(round(2 * moyenne_1))

print(f"Étape 1 (eta=0.5) : Moyenne mesurée = {moyenne_1:.2f}  => N estimé = {N_estime}")

# ==========================================
# EXPÉRIENCE 2 : Trouver la répartition (eta = 0.1)
# ==========================================
ETA_2 = 0.1
dist_2 = get_b_dist(I_SECRET, K_SECRET, ETA_2)
# En réalité le détecteur ne mesure qu'au maximum N_estime photons
mesures_2 = np.random.choice(np.arange(I_SECRET + K_SECRET + 1), size=N_SHOTS, p=dist_2)

moyenne_2 = np.mean(mesures_2)

# Mathématiques : mu = i*eta + k*(1-eta). Sachant que k = N - i
# mu = i*eta + (N-i)*(1-eta)
# mu = i*eta + N - N*eta - i + i*eta
# mu - N*(1-eta) = i * (2*eta - 1)
i_calcule = (moyenne_2 - N_estime * (1 - ETA_2)) / (2 * ETA_2 - 1)
i_estime = int(round(i_calcule))
k_estime = N_estime - i_estime

print(f"Étape 2 (eta=0.1) : Moyenne mesurée = {moyenne_2:.2f}")
print(f"\n--- RÉSULTAT FINAL ---")
print(f"Estimation : i = {i_estime}, k = {k_estime}")
print(f"Vérité abs : i = {I_SECRET}, k = {K_SECRET}")