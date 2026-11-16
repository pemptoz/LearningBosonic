import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# --- Paramètres ---
# Définition de l'axe des abscisses (Rayon au carré R)
R = np.linspace(0, 15, 500)

# Valeurs de n (nombre de photons des états de Fock à tracer)
n_values = [1, 3, 5]

# --- Création de la figure ---
plt.figure(figsize=(8, 5))

# Couleurs pour un beau rendu sur le poster
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Tracé des distributions pour chaque état de Fock
for n, color in zip(n_values, colors):
    # La distribution théorique est Gamma(n+1, 1)
    # Dans scipy.stats.gamma, le paramètre de forme (shape) 'a' correspond à n+1
    # L'échelle (scale) est à 1.
    P_R = gamma.pdf(R, a=n+1, scale=1)
    
    # Remplissage sous la courbe (optionnel mais très esthétique pour un poster)
    plt.fill_between(R, P_R, alpha=0.1, color=color)
    
    # Tracé de la ligne principale
    plt.plot(R, P_R, label=f'Fock State $|{n}\\rangle$', color=color, linewidth=2.5)

# --- Personnalisation ---
plt.title("Distribution of the square radius $R = |\\alpha|^2$ (Heterodyne Measure)", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Square radius $R$", fontsize=12)
plt.ylabel("Probability Density $P(R)$", fontsize=12)

# Ajout de la légende et d'une grille
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Limites pour bien cadrer l'information
plt.xlim(0, 12)
plt.ylim(0, 0.4)

plt.tight_layout()

# --- Sauvegarde ---
# Sauvegarde en format vectoriel PDF, idéal pour ne pas pixeliser sur un poster imprimé en A0
plt.savefig('gamma_distribution_fock.pdf', bbox_inches='tight')
plt.show()