import sys
import numpy as np

sys.setrecursionlimit(5000)

# --- Fonction théorique (inchangée) ---
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

# --- Paramètres pour de grands nombres ---
I_SECRET = 200  # Nombres plus grands !
K_SECRET = 180
N_SHOTS = 10   # Un nombre de tirs raisonnable

print(f"--- PROTOCOLE CROISÉ (Vraies valeurs : i={I_SECRET}, k={K_SECRET}) ---")

# ==========================================
# EXPÉRIENCE A (eta = 0.9)
# ==========================================
ETA_A = 0.9
dist_A = get_b_dist(I_SECRET, K_SECRET, ETA_A)
mesures_A = np.random.choice(np.arange(I_SECRET + K_SECRET + 1), size=N_SHOTS, p=dist_A)
mu_A = np.mean(mesures_A)
print(f"Moyenne mesurée avec eta=0.9 : {mu_A:.2f}")

# ==========================================
# EXPÉRIENCE B (eta = 0.1)
# ==========================================
ETA_B = 0.1
dist_B = get_b_dist(I_SECRET, K_SECRET, ETA_B)
mesures_B = np.random.choice(np.arange(I_SECRET + K_SECRET + 1), size=N_SHOTS, p=dist_B)
mu_B = np.mean(mesures_B)
print(f"Moyenne mesurée avec eta=0.1 : {mu_B:.2f}")

# ==========================================
# RÉSOLUTION DU SYSTÈME CROISÉ
# ==========================================
# 1. On trouve N en additionnant les deux moyennes
N_estime_brut = mu_A + mu_B
N_estime = int(round(N_estime_brut))

# 2. On trouve l'écart (i - k) en soustrayant les moyennes
# mu_A - mu_B = 0.8 * (i - k)
ecart_ik = (mu_A - mu_B) / 0.8

# 3. On en déduit i et k (puisque i = (N + ecart)/2 et k = (N - ecart)/2)
i_estime = int(round((N_estime + ecart_ik) / 2))
k_estime = int(round((N_estime - ecart_ik) / 2))

print(f"\n--- RÉSULTAT FINAL ---")
print(f"Estimation croisée : i = {i_estime}, k = {k_estime} (Total N = {i_estime + k_estime})")