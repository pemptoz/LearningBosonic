import numpy as np
import matplotlib.pyplot as plt
from qutip import fock, qfunc

# --- Paramètres de la simulation ---
N_dim = 15       # Dimension de l'espace tronqué
n_photons = 5    # Nombre de photons (n=3 pour bien voir le donut)

# 1. Création de l'état de Fock |n>
psi = fock(N_dim, n_photons)

# 2. Définition de la grille de l'espace des phases
xvec = np.linspace(-6, 6, 100) # Un peu moins de points (100) pour fluidifier la 3D
yvec = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(xvec, yvec) # Création de la grille 2D pour la surface 3D

# 3. Calcul de la fonction Q de Husimi
Q = qfunc(psi, xvec, yvec)

# --- Paramètres d'affichage 3D ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Tracé de la surface
# 'cmap' gère les couleurs, 'alpha' gère la transparence (optionnel)
surf = ax.plot_surface(X, Y, Q, cmap='viridis', edgecolor='none', antialiased=True)

# Ajout de la barre de couleur
fig.colorbar(surf, ax=ax, label='Probability Q(α)', shrink=0.5, aspect=10)

# Personnalisation des axes
ax.set_title(f"Husimi Q Representation (3D) for the Fock State $|{n_photons}\\rangle$")
ax.set_xlabel(r'Re($\alpha$)')
ax.set_ylabel(r'Im($\alpha$)')
ax.set_zlabel(' Q(α)')

# Ajustement de l'angle de vue (élévation, azimut)
# Tu peux modifier ces valeurs pour tourner la figure (ex: 30, 45)
ax.view_init(elev=35, azim=45)

# Sauvegarde pour le poster
plt.savefig('husimi_donut_3d.pdf', bbox_inches='tight')
plt.show()