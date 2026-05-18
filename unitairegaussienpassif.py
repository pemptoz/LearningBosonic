import numpy as np
import qutip as qt
from scipy.special import factorial

# =====================================================================
# 1. CONFIGURATION DU LABORATOIRE QUANTIQUE (QUTIP)
# =====================================================================
N_dim = 30      # Dimension de troncature de la base de Fock
i_fock = 8       # État de Fock en entrée du Mode 0 (|i>)
k_fock = 12       # État de Fock en entrée du Mode 1 (|k>)
N_shots = 10000   # Nombre de tirs expérimentaux à générer

# Le canal : angle de mélange symplectique passif (ex: 33 degrés)
theta_phys = np.radians(150) 

print("--- INITIALISATION DE LA SIMULATION QUTIP ---")
print(f"Préparation de l'état initial : |{i_fock}> ⊗ |{k_fock}>")
print(f"Application de l'unitaire symplectique (Angle réel = {np.degrees(theta_phys)}°)")

# Opérateurs de destruction dans l'espace composite à 2 modes
a1 = qt.tensor(qt.destroy(N_dim), qt.qeye(N_dim))
a2 = qt.tensor(qt.qeye(N_dim), qt.destroy(N_dim))

# Hamiltonien d'interaction du Beam Splitter et calcul de l'unitaire U
H_bs = a1.dag() * a2 - a1 * a2.dag()
U_quantum = (theta_phys * H_bs).expm()

# Génération de l'état de sortie
psi_in = qt.tensor(qt.fock(N_dim, i_fock), qt.fock(N_dim, k_fock))
psi_out = U_quantum * psi_in

# Extraction de la matrice des coefficients complexes
coeffs = psi_out.full().reshape(N_dim, N_dim)

# =====================================================================
# 2. ÉCHANTILLONNEUR DE MESURE HÉTÉRODYNE (MCMC AVEC THINNING EN MANIFOLD)
# =====================================================================
# Pré-calcul des factorielles
fock_indices = np.arange(N_dim)
sqrt_fact = np.sqrt(factorial(fock_indices))

def log_Q_husimi(coords):
    """ Calcule la densité de probabilité (log) de la mesure hétérodyne """
    q0, p0, q1, p1 = coords
    alpha0 = (q0 - 1j*p0) / np.sqrt(2)
    alpha1 = (q1 - 1j*p1) / np.sqrt(2)
    
    v0 = (np.conj(alpha0) ** fock_indices) / sqrt_fact
    v1 = (np.conj(alpha1) ** fock_indices) / sqrt_fact
    amplitude = v0 @ coeffs @ v1
    
    prob = np.abs(amplitude)**2 * np.exp(-(q0**2 + p0**2 + q1**2 + p1**2)/2) / (np.pi**2)
    return np.log(prob) if prob > 1e-20 else -np.inf

# --- PARAMÈTRES MCMC OPTIMISÉS ---
thinning = 25  # On ne garde qu'un pas sur 25 pour briser l'autocorrélation
print(f"Génération de {N_shots} tirs hétérodynes indépendants (Thinning = {thinning})...")

X_list = []
current_coords = np.array([0.0, 0.0, 0.0, 0.0])
current_log_p = log_Q_husimi(current_coords)

# 1. Phase de Burn-in (Stabilisation de la chaîne)
for _ in range(2000):
    proposal = current_coords + np.random.normal(0, 0.6, 4)
    proposal_log_p = log_Q_husimi(proposal)
    if np.log(np.random.uniform(0, 1)) < proposal_log_p - current_log_p:
        current_coords = proposal
        current_log_p = proposal_log_p

# 2. Phase de Production avec écremage (Thinning)
for shot in range(N_shots):
    for _ in range(thinning):  # On boucle dans le vide pour sauter les pas corrélés
        proposal = current_coords + np.random.normal(0, 0.6, 4)
        proposal_log_p = log_Q_husimi(proposal)
        if np.log(np.random.uniform(0, 1)) < proposal_log_p - current_log_p:
            current_coords = proposal
            current_log_p = proposal_log_p
    
    # On enregistre uniquement le point isolé
    X_list.append(current_coords.copy())

# Matrice finale sous forme standard (4 lignes x N_shots colonnes)
X_obs = np.array(X_list).T# =====================================================================
# 2. ÉCHANTILLONNEUR DE MESURE HÉTÉRODYNE (MCMC AVEC THINNING EN MANIFOLD)
# =====================================================================

print("\n--- ÉTAPE 1 (QUTIP) COMPLÉTÉE ---")
print(f"Matrice de données X_obs prête. Forme : {X_obs.shape}")
print(f"Premier tir mesuré (q0, p0, q1, p1) :\n{X_obs[:, 0]}")
from scipy.optimize import minimize_scalar
import scipy.special as sep

# =====================================================================
# ÉTAPE 2 & 3 : ICA DIRECTE SANS BLANCHIMENT (PRÉSERVE L'ÉNERGIE)
# =====================================================================
print("\n--- ÉTAPE 3 : OPTIMISATION DE LA BLOCK-ICA DIRECTE ---")

def cost_function_direct(theta):
    """ Calcule la corrélation d'intensité directement sur les données brutes """
    c, s = np.cos(theta), np.sin(theta)
    # Matrice de rotation pure (sans déformation d'échelle)
    W = np.array([
        [c, 0, -s, 0],
        [0, c, 0, -s],
        [s, 0, c, 0],
        [0, s, 0, c]
    ])
    Y = W @ X_obs
    r0_sq = Y[0]**2 + Y[1]**2
    r1_sq = Y[2]**2 + Y[3]**2
    return np.cov(r0_sq, r1_sq)[0, 1]**2

print("Recherche de l'angle de rotation physique optimal...")
res = minimize_scalar(cost_function_direct, bounds=(-np.pi, np.pi), method='bounded')
theta_opt = res.x
print(f"Convergence réussie ! Angle de séparation trouvé : {np.degrees(theta_opt):.2f}°")

# Application de la séparation brute
c_opt, s_opt = np.cos(theta_opt), np.sin(theta_opt)
W_opt = np.array([
    [c_opt, 0, -s_opt, 0],
    [0, c_opt, 0, -s_opt],
    [s_opt, 0, c_opt, 0],
    [0, s_opt, 0, c_opt]
])
Y_final = W_opt @ X_obs

# Gestion automatique de l'indétermination de permutation de l'ICA
# Le mode avec le plus petit rayon moyen sera associé à |5>, l'autre à |10>
r0_sq_check = Y_final[0]**2 + Y_final[1]**2
r1_sq_check = Y_final[2]**2 + Y_final[3]**2

if np.mean(r0_sq_check) < np.mean(r1_sq_check):
    q_A, p_A = Y_final[0], Y_final[1]
    q_B, p_B = Y_final[2], Y_final[3]
else:
    q_A, p_A = Y_final[2], Y_final[3]
    q_B, p_B = Y_final[0], Y_final[1]

# =====================================================================
# ÉTAPE 4 : TOMOGRAPHIE QUANTIQUE ET MESURE D'ÉCART (VERSION CONVERGÉE)
# =====================================================================
print("\n--- ÉTAPE 4 : TOMOGRAPHIE QUANTIQUE ET MESURE D'ÉCART ---")

def tomographie_mle_fock(q, p, target_fock, N_max=18):
    """ 
    Reconstruit la distribution de Fock diagonale par maximum de vraisemblance
    à partir des tirs hétérodynes (échantillons de la fonction Q de Husimi).
    """
    r_sq = q**2 + p**2
    N_shots = len(r_sq)
    
    # Pré-calcul de la matrice de projection Husimi M [N_shots x N_max+1]
    M = np.zeros((N_shots, N_max + 1))
    for n in range(N_max + 1):
        log_Q = n * np.log(r_sq / 2 + 1e-15) - sep.gammaln(n + 1) - r_sq / 2 - np.log(np.pi)
        M[:, n] = np.exp(log_Q)
        
    # Algorithme Expectation-Maximization (EM) poussé à 2000 itérations
    p_estimate = np.ones(N_max + 1) / (N_max + 1) # Initialisation uniforme
    
    # Évolution de l'estimation pour faire s'effondrer les profils chevauchants
    for _ in range(2000): 
        projections = M @ p_estimate
        projections = np.maximum(projections, 1e-15)
        p_estimate = p_estimate * (M.T @ (1.0 / projections)) / N_shots
        p_estimate /= np.sum(p_estimate) # Normalisation Trace = 1
        
    # Calcul des écarts par rapport à la cible pure
    fidelity = p_estimate[target_fock]
    infidelity = 1.0 - fidelity
    
    p_true = np.zeros(N_max + 1)
    p_true[target_fock] = 1.0
    trace_distance = 0.5 * np.sum(np.abs(p_estimate - p_true))
    
    return p_estimate, infidelity, trace_distance

# Évaluation des deux canaux extraits
p_A, infidelite_A, dist_trace_A = tomographie_mle_fock(q_A, p_A, target_fock=i_fock)
p_B, infidelite_B, dist_trace_B = tomographie_mle_fock(q_B, p_B, target_fock=k_fock)

# --- AFFICHAGE DES RÉSULTATS ---
print(f" Statistiques Mode 0 (Attendu: |{i_fock}>) :")
print(f"  -> Infidélité (1 - F) : {infidelite_A:.4f}")
print(f"  -> Distance de trace  : {dist_trace_A:.4f}")
print(f"  -> Top 3 des populations Fock reconstruites :")
for idx in np.argsort(p_A)[::-1][:3]:
    print(f"     |{idx}> : {p_A[idx]*100:.2f}%")

print(f"\n Statistiques Mode 1 (Attendu: |{k_fock}>) :")
print(f"  -> Infidélité (1 - F) : {infidelite_B:.4f}")
print(f"  -> Distance de trace  : {dist_trace_B:.4f}")
print(f"  -> Top 3 des populations Fock reconstruites :")
for idx in np.argsort(p_B)[::-1][:3]:
    print(f"     |{idx}> : {p_B[idx]*100:.2f}%")

# Écart global du système composite
infidelite_totale = 1.0 - ((1.0 - infidelite_A) * (1.0 - infidelite_B))
print(f"\n ÉCART GLOBAL DU SYSTÈME :")
print(f"  -> Infidélité totale de l'état extrait : {infidelite_totale:.4f}")
