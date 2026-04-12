import numpy as np
import pickle
import time

def generer_banque_ultra_rapide(N_MAX, eta):
    print(f"Création de la Matrice 3D pour N_MAX = {N_MAX}...")
    
    # 1. CRÉATION DE LA MATRICE GLOBALE
    # On crée un cube géant rempli de zéros. Dimensions : (i, k, n)
    # Pour N=200, ce cube pèse environ 65 Mo en RAM. C'est très léger.
    B = np.zeros((N_MAX + 1, N_MAX + 1, N_MAX + 1))
    
    eta_bar = 1 - eta
    
    # 2. LA BASE (Condition initiale)
    # Si 0 photon rentre, on a 100% de chance d'avoir 0 photon en sortie
    B[0, 0, 0] = 1.0
    
    # 3. LE REMPLISSAGE ITÉRATIF (Sans aucune récursivité !)
    # On remplit le cube couche par couche
    for i in range(N_MAX + 1):
        for k in range(N_MAX + 1 - i):
            if i == 0 and k == 0:
                continue # Déjà géré
            
            N_total = i + k
            for n in range(N_total + 1):
                # On applique la formule de récurrence du papier de recherche
                # En vérifiant simplement de ne pas lire des index négatifs
                
                term1 = eta * B[i-1, k, n-1] if (i > 0 and n > 0) else 0.0
                term2 = eta * B[i, k-1, n] if (k > 0) else 0.0
                term3 = eta_bar * B[i-1, k, n] if (i > 0) else 0.0
                term4 = eta_bar * B[i, k-1, n-1] if (k > 0 and n > 0) else 0.0
                term5 = B[i-1, k-1, n-1] if (i > 0 and k > 0 and n > 0) else 0.0
                
                B[i, k, n] = term1 + term2 + term3 + term4 - term5

    # 4. CRÉATION DU DICTIONNAIRE (Pour la compatibilité avec ton Code 2)
    print("Formatage de la banque de données...")
    banque_theorique = {}
    
    for i in range(N_MAX + 1):
        for k in range(N_MAX + 1 - i):
            # On extrait juste la tranche utile (jusqu'à n = i+k)
            dist = B[i, k, :i+k+1].copy()
            
            # Nettoyage des erreurs de flottants (comme vu précédemment)
            dist = np.maximum(dist, 0)
            somme = np.sum(dist)
            if somme > 0:
                dist /= somme
                
            banque_theorique[(i, k)] = dist
            
    return banque_theorique

# --- LANCEMENT ---
N_MAX = 300 # Tu peux mettre 300 ici !
ETA = 0.3

temps_debut = time.time()

# Exécution de l'algorithme optimisé
banque = generer_banque_ultra_rapide(N_MAX, ETA)

temps_fin = time.time()
print(f"Génération terminée en {temps_fin - temps_debut:.2f} secondes !")
print(f"Le dictionnaire contient {len(banque)} combinaisons (i, k).")

# Sauvegarde
with open('banque_fock.pkl', 'wb') as fichier:
    pickle.dump(banque, fichier)
    
print("Sauvegardé avec succès dans 'banque_fock.pkl'.")