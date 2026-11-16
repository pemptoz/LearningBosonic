import numpy as np
import pickle
import matplotlib.pyplot as plt

try:
    with open('banque_fock_big.pkl', 'rb') as fichier:
        banque_theorique = pickle.load(fichier)
    print("Banque de données chargée avec succès.")
except FileNotFoundError:
    print("Erreur : Le fichier 'banque_fock.pkl' est introuvable. Lancez le Code 1 d'abord.")
    exit()

I_SECRET = 50
K_SECRET = 60
N_SHOTS = 100  # Nombre de tirs expérimentaux

vraie_distribution = banque_theorique[(I_SECRET, K_SECRET)]
mesures_labo = np.random.choice(np.arange(len(vraie_distribution)), size=N_SHOTS, p=vraie_distribution)

n_max_observe = np.max(mesures_labo)
frequences_exp = np.zeros(n_max_observe + 1)
valeurs_uniques, comptes = np.unique(mesures_labo, return_counts=True)
frequences_exp[valeurs_uniques] = comptes / N_SHOTS

print(n_max_observe)


meilleur_i = -1
meilleur_k = -1
min_erreur = float('inf')
meilleure_distribution_theorique = None

for (i, k), dist_theo in banque_theorique.items():
    
    if (i + k) < n_max_observe:
        continue
    taille_max = max(len(frequences_exp), len(dist_theo))
    exp_alignee = np.pad(frequences_exp, (0, taille_max - len(frequences_exp)))
    theo_alignee = np.pad(dist_theo, (0, taille_max - len(dist_theo)))
    
    erreur = np.sum((exp_alignee - theo_alignee)**2)
    
    if erreur < min_erreur:
        min_erreur = erreur
        meilleur_i = i
        meilleur_k = k
        meilleure_distribution_theorique = dist_theo

print(f"L'algorithme a trouvé : i = {meilleur_i}, k = {meilleur_k} (Erreur MSE = {min_erreur:.5f})")
print(f"Les VRAIES valeurs étaient : i = {I_SECRET}, k = {K_SECRET}")

plt.figure(figsize=(10, 6))

plt.scatter(np.arange(len(frequences_exp)), frequences_exp, color='royalblue', 
            marker='o', s=60, label=f'Expérimental ({N_SHOTS} tirs)', zorder=3)

plt.scatter(np.arange(len(meilleure_distribution_theorique)), meilleure_distribution_theorique, 
            color='crimson', marker='X', s=100, label=f'Théorie gagnante (i={meilleur_i}, k={meilleur_k})', zorder=2)

plt.xlabel("n (nombre de photons en sortie)")
plt.ylabel("Probabilité / Fréquence")
plt.legend()
plt.grid(True, alpha=0.4)

plt.show()