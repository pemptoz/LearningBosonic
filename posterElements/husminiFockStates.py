import numpy as np
import matplotlib.pyplot as plt
from qutip import fock, qfunc

# --- Paramètres de la simulation ---
N_dim = 15       # Dimension de l'espace de Hilbert tronqué (doit être > n)
n_photons = 6    # Le nombre de photons de notre état de Fock (ex: n=3)

# 1. Création de l'état de Fock |n>
# On définit l'état pur de Fock n=3 dans un espace de dimension N_dim
psi = fock(N_dim, n_photons)

# 2. Définition de la grille de l'espace des phases (alpha = x + iy)
# On choisit une plage allant de -4 à 4 pour bien voir l'anneau
xvec = np.linspace(-5, 5, 200)
yvec = np.linspace(-5, 5, 200)

# 3. Calcul de la fonction Q de Husimi
# qutip.qfunc prend l'état et les vecteurs x et y de la grille
Q = qfunc(psi, xvec, yvec)

# --- Paramètres d'affichage (Plotting) ---
fig, ax = plt.subplots(figsize=(6, 5))

# Affichage avec pcolor ou contourf
# cmap 'viridis', 'plasma' ou 'magma' rendent très bien pour les densités
c = ax.contourf(xvec, yvec, Q, 100, cmap='viridis')
fig.colorbar(c, ax=ax, label='Probabilité Q(α)')

# Personnalisation des axes
ax.set_title(f'Fonction de Husimi Q pour l\'état de Fock $|{n_photons}\\rangle$')
ax.set_xlabel(r'Re($\alpha$)')
ax.set_ylabel(r'Im($\alpha$)')
ax.set_aspect('equal') # Pour que le donut soit bien rond

# Sauvegarde de l'image (parfaite pour votre poster)
plt.savefig('husimi_donut.pdf', bbox_inches='tight')
plt.show()