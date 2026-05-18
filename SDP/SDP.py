import numpy as np
from qutip import basis, tensor, destroy, qeye, ket2dm, expect


# ÉTAPE 1 : Paramètres du système

i = 4  # Nombre de photons injectés en entrée 1
k = 2  # Nombre de photons injectés en entrée 2
N_tot = i + k  # Nombre total de photons dans le système

# Troncature de l'espace de Fock
# On ajoute une petite marge (+2) pour autoriser notre futur algorithme 
# à "explorer" les états avec un peu plus de photons à cause du bruit.
N_max = 15
dim = N_max + 1  # La dimension des matrices (de 0 à N_max photons)

print(f"Simulation pour |{i},{k}> avec troncature à {N_max} photons.")


# ÉTAPE 2 : Création de l'état initial

# basis(dim, i) crée l'état de Fock |i> 
# tensor() colle les deux états ensemble pour faire |i>|k>
psi_in = tensor(basis(dim, i), basis(dim, k))

# On le transforme en matrice densité rho = |i,k><i,k|
rho_in = ket2dm(psi_in)


# ÉTAPE 3 : Création du Beam-Splitter (U_BS)

# Opérateurs d'annihilation (destroy) et Identité (qeye)
# 'a' agit sur le mode 1, 'b' agit sur le mode 2
a = tensor(destroy(dim), qeye(dim))
b = tensor(qeye(dim), destroy(dim))

# Transmittance du Beam-Splitter (50:50 correspond à un angle theta = pi/4)
theta = np.pi / 6

# L'opérateur unitaire d'un Beam-Splitter s'écrit avec une exponentielle
# U_BS = exp( theta * (a^dag * b - a * b^dag) )
U_BS = (theta * (a.dag() * b - a * b.dag())).expm()


# ÉTAPE 4 : Calcul des B_n et E_n
B_mesures = []
E_matrices = []

# On regarde les probabilités d'avoir n photons dans la sortie 1
for n in range(dim):
    
    # Projecteur |n><n| sur le mode 1
    proj_n = ket2dm(basis(dim, n))
    
    # Projecteur global Pi_n = |n><n| (x) Identité
    Pi_n = tensor(proj_n, qeye(dim))
    

    # Construction de notre observable de tomographie E_n
    E_n = U_BS.dag() * Pi_n * U_BS
    E_matrices.append(E_n)

    # Calcul de la probabilité théorique B_n 
    # expect(A, rho) calcule directement la trace(B_n   * rho) dans QuTiP
    B_n_the = expect(E_n, rho_in)
    B_mesures.append(B_n_the)
    


# Affichage des résultats
print("\n--- Résultats de la simulation ---")
for n in range(dim):
    print(f"Probabilité de détecter {n} photons (B_{n}) : {B_mesures[n]}")


# Nombre de fois qu'on répète l'expérience dans notre laboratoire virtuel
num_shots = 1000  

# On demande à numpy de simuler 1000 tirages basés sur nos probabilités parfaites
tirages_bruites = np.random.multinomial(num_shots, B_mesures)

# On recalcule les nouvelles probabilités "mesurées" par notre faux détecteur
B_mesures_bruites = tirages_bruites / num_shots

print("\n--- Mesures réelles avec bruit statistique (1000 shots) ---")
for n, prob in enumerate(B_mesures_bruites):
    print(f"Probabilité mesurée pour {n} photons : {prob:.4f}")

#--------------------------------------------------------------------------------

import cvxpy as cp
import numpy as np
from qutip import num

print("\n==========================================")
print(" DÉMARRAGE DE L'OPTIMISATION SDP (CVXPY)")
print("==========================================\n")


# 1. Préparation des données pour CVXPY

# La taille de notre matrice rho globale (dim_tot x dim_tot)
# Rappel : l'espace est le produit tensoriel des deux modes
dim_tot = dim * dim 

# On convertit nos observables QuTiP en tableaux NumPy standards
E_np = [E.full() for E in E_matrices]

# On convertit nos mesures en un vecteur colonne NumPy
B_np = np.array(B_mesures_bruites).reshape(-1, 1)

# Construction de l'opérateur Nombre de photons global (N_a + N_b)
# On utilise la fonction 'num' de QuTiP, puis on convertit en NumPy
N_a = tensor(num(dim), qeye(dim))
N_b = tensor(qeye(dim), num(dim))
N_tot_op = (N_a + N_b).full()


# Paramètres de l'algorithme IRL1
iterations_irl1 = 5
epsilon = 1e-3
lambda_penalite = 0.00005  # À ajuster selon le niveau de bruit

# Initialisation des poids (tous à 1 au départ)
W = np.ones(dim_tot)

rho = cp.Variable((dim_tot, dim_tot), hermitian=True)
predictions = cp.vstack([cp.real(cp.trace(E_np[n] @ rho)) for n in range(len(B_mesures_bruites))])
erreur_L2 = cp.sum_squares(predictions - B_np)

contraintes = [
    rho >> 0,
    cp.trace(rho) == 1
]

print("Début de l'optimisation itérative IRL1...")

for m in range(iterations_irl1):
    # La pénalité est le produit scalaire entre les poids W et la diagonale de rho
    penalite_sparse = lambda_penalite * cp.sum(cp.multiply(W, cp.real(cp.diag(rho))))
    
    objectif = cp.Minimize(erreur_L2 + penalite_sparse)
    probleme = cp.Problem(objectif, contraintes)
    
    # Résolution de l'itération m
    probleme.solve(solver=cp.SCS)
    
    # Extraction de la diagonale trouvée
    p_courant = np.real(np.diag(rho.value))
    
    # Nettoyage des valeurs négatives dues aux erreurs numériques avant la division
    p_courant = np.clip(p_courant, 0, None)
    
    # Mise à jour des poids pour la prochaine itération
    W = 1.0 / (p_courant + epsilon)
    
    # Diagnostic rapide de l'itération
    index_max = np.argmax(p_courant)
    print(f"Itération {m+1} | Index max : {index_max} | Prob : {p_courant[index_max]:.4f}")

# Le rho.value final contient l'état le plus sparse possible
rho_reconstruit = rho.value

# 4. Analyse du résultat trouvé

# On va chercher la plus grande probabilité dans la diagonale de l'état reconstruit
diagonale = np.real(np.diag(rho_reconstruit))
index_max = np.argmax(diagonale)
prob_max = diagonale[index_max]

print("\n--- Diagnostic de l'état reconstruit ---")
print(f"La plus grande population se trouve à l'index {index_max} avec une probabilité de {prob_max*100:.2f}%")

# On traduit l'index 1D en coordonnées 2D (i, k)
i_trouve = index_max // dim
k_trouve = index_max % dim

print("\n==========================================")
print(" CONCLUSION DE LA TOMOGRAPHIE SDP")
print("==========================================")
print(f"L'algorithme SDP est formel (à {prob_max*100:.2f}%) :")
print(f"-> Nombre de photons injectés en entrée 1 (i) : {i_trouve}")
print(f"-> Nombre de photons injectés en entrée 2 (k) : {k_trouve}")
print(f"L'état initial était donc : |{i_trouve}, {k_trouve}>")  