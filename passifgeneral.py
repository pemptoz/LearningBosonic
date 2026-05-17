import numpy as np
import qutip as qt
import scipy.special as sep
from scipy.optimize import minimize
from scipy.linalg import expm

# =====================================================================
# 0. FONDATIONS MATHÉMATIQUES : ALGÈBRE DE LIE Sp(4) ∩ O(4)
# =====================================================================
# Les 4 générateurs réels de U(2) dans l'espace des phases
G0 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]) 
G1 = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]) 
G2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]]) 
G3 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]) 

def parametres_to_W(params):
    """ Transforme 4 paramètres libres en une matrice parfaitement symplectique et orthogonale. """
    t0, t1, t2, t3 = params
    A = t0*G0 + t1*G1 + t2*G2 + t3*G3
    return expm(A)

# =====================================================================
# 1. GÉNÉRATION DU CANAL ET SIMULATION DES DONNÉES EXPERIMENTALES
# =====================================================================
print("--- 1. CRÉATION DU CANAL ET MESURE HÉTÉRODYNE ---")

N_dim = 30
i_fock = 5   # Mode A d'origine
k_fock = 10  # Mode B d'origine
N_shots = 200000
thinning = 25

# Création d'un canal caché S_phys (mélange arbitraire)
vraies_phases = np.random.uniform(-1.5, 1.5, 4)
S_phys = parametres_to_W(vraies_phases)

print("Vraie matrice du canal caché (S_phys) :")
print(np.round(S_phys, 3))

# Traduction de S_phys en Hamiltonien quantique pour QuTiP
U_complexe = np.zeros((2, 2), dtype=complex)
U_complexe[0, 0] = S_phys[0, 0] - 1j*S_phys[0, 1]
U_complexe[0, 1] = S_phys[0, 2] - 1j*S_phys[0, 3]
U_complexe[1, 0] = S_phys[2, 0] - 1j*S_phys[2, 1]
U_complexe[1, 1] = S_phys[2, 2] - 1j*S_phys[2, 3]

U_qutip = qt.Qobj(U_complexe)
H_int = (U_qutip.logm() / 1j).full()

a1 = qt.tensor(qt.destroy(N_dim), qt.qeye(N_dim))
a2 = qt.tensor(qt.qeye(N_dim), qt.destroy(N_dim))
H_quantum = (
    H_int[0,0]*a1.dag()*a1 + H_int[0,1]*a1.dag()*a2 + 
    H_int[1,0]*a2.dag()*a1 + H_int[1,1]*a2.dag()*a2
)
U_quantum = (1j * H_quantum).expm()

# État de sortie mélangé
psi_out = U_quantum * qt.tensor(qt.fock(N_dim, i_fock), qt.fock(N_dim, k_fock))
coeffs = psi_out.full().reshape(N_dim, N_dim)

# Échantillonnage MCMC (Mesure hétérodyne simulée)
fock_indices = np.arange(N_dim)
sqrt_fact = np.sqrt(sep.factorial(fock_indices))

def log_Q_husimi(coords):
    q0, p0, q1, p1 = coords
    alpha0, alpha1 = (q0 - 1j*p0)/np.sqrt(2), (q1 - 1j*p1)/np.sqrt(2)
    v0 = (np.conj(alpha0) ** fock_indices) / sqrt_fact
    v1 = (np.conj(alpha1) ** fock_indices) / sqrt_fact
    prob = np.abs(v0 @ coeffs @ v1)**2 * np.exp(-(q0**2 + p0**2 + q1**2 + p1**2)/2) / (np.pi**2)
    return np.log(prob) if prob > 1e-20 else -np.inf

print(f"Génération de {N_shots} tirs MCMC (Patiente un instant)...")
X_list = []
coords = np.zeros(4)
log_p = log_Q_husimi(coords)
for _ in range(N_shots):
    for _ in range(thinning):
        prop = coords + np.random.normal(0, 0.6, 4)
        prop_log_p = log_Q_husimi(prop)
        if np.log(np.random.rand()) < prop_log_p - log_p:
            coords, log_p = prop, prop_log_p
    X_list.append(coords.copy())
X_obs = np.array(X_list).T

# =====================================================================
# 2. BLOCK-ICA : SÉPARATION AVEUGLE GÉOMÉTRIQUE
# =====================================================================
print("\n--- 2. BLOCK-ICA : RECHERCHE DE LA MATRICE DE SÉPARATION ---")

def cout_covariance_rayons(params):
    W = parametres_to_W(params)
    Y = W @ X_obs
    r0_sq, r1_sq = Y[0]**2 + Y[1]**2, Y[2]**2 + Y[3]**2
    return np.cov(r0_sq, r1_sq)[0, 1]**2

# Multi-start pour éviter les faux plats
meilleur_score = np.inf
meilleurs_params = None
points_depart = [np.zeros(4)] + [np.random.uniform(-2, 2, 4) for _ in range(4)]

for start in points_depart:
    res = minimize(cout_covariance_rayons, start, method='Nelder-Mead', 
                   options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-7})
    if res.fun < meilleur_score:
        meilleur_score = res.fun
        meilleurs_params = res.x

W_opt = parametres_to_W(meilleurs_params)

print("Matrice de séparation trouvée (W_opt) :")
print(np.round(W_opt, 3))
print("\nProduit de contrôle W_opt @ S_phys (Devrait être bloc-diagonale) :")
print(np.round(W_opt @ S_phys, 3))

# =====================================================================
# 3. TOMOGRAPHIE (MLE) ET IDENTIFICATION DES ÉTATS
# =====================================================================
print("\n--- 3. RECONSTRUCTION MLE DES SIGNAUX SÉPARÉS ---")

Y_final = W_opt @ X_obs
r0_sq = Y_final[0]**2 + Y_final[1]**2
r1_sq = Y_final[2]**2 + Y_final[3]**2

# Permutation automatique basée sur l'énergie moyenne (5 photons vs 10 photons)
if np.mean(r0_sq) < np.mean(r1_sq):
    rA_sq, rB_sq = r0_sq, r1_sq
else:
    rA_sq, rB_sq = r1_sq, r0_sq

def tomographie_mle(r_sq, N_max=16):
    """ Reconstruction de la matrice densité diagonale depuis les intensités. """
    N_data = len(r_sq)
    M = np.zeros((N_data, N_max + 1))
    for n in range(N_max + 1):
        log_Q = n * np.log(r_sq / 2 + 1e-15) - sep.gammaln(n + 1) - r_sq / 2 - np.log(np.pi)
        M[:, n] = np.exp(log_Q)
        
    p_est = np.ones(N_max + 1) / (N_max + 1)
    for _ in range(3000): # Itérations EM
        proj = np.maximum(M @ p_est, 1e-15)
        p_est = p_est * (M.T @ (1.0 / proj)) / N_data
        p_est /= np.sum(p_est)
    return p_est

print("Tomographie en cours...")
p_A = tomographie_mle(rA_sq, N_max=16)
p_B = tomographie_mle(rB_sq, N_max=16)

fidelite_A = p_A[i_fock]
fidelite_B = p_B[k_fock]

print(f"\nMode 0 (Cible |{i_fock}>) -> Fidélité : {fidelite_A*100:.2f}% (Infidélité: {1-fidelite_A:.4f})")
print(f"Mode 1 (Cible |{k_fock}>) -> Fidélité : {fidelite_B*100:.2f}% (Infidélité: {1-fidelite_B:.4f})")
print(f"Fidélité globale du système : {fidelite_A * fidelite_B * 100:.2f}%")