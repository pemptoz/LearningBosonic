import numpy as np
import pickle
import time

def generer_banque_big_data(N_MAX, eta):
    print(f"Génération Big Data pour N_MAX = {N_MAX} (eta={eta})")
    eta_bar = 1.0 - eta
    
    # On ne garde en mémoire que les DEUX dernières couches pour économiser la RAM
    # couche_S1 = Couche (i+k - 1)
    # couche_S2 = Couche (i+k - 2)
    
    # Initialisation pour N=0 (0 photon)
    couche_S2 = {} 
    couche_S1 = {0: np.array([1.0])} # i=0 -> [1.0]
    
    banque_theorique = {(0, 0): np.array([1.0])}
    
    # On avance couche par couche (S = nombre total de photons injectés)
    for S in range(1, N_MAX + 1):
        if S % 100 == 0:
            print(f"Calcul de la couche S={S}...")
            
        couche_actuelle = {}
        
        # Pour cette couche S, on calcule tous les i possibles (k = S - i)
        # Comme ils sont indépendants, c'est ICI qu'on pourrait utiliser le Multiprocessing !
        for i in range(S + 1):
            k = S - i
            
            # On prépare le tableau des probabilités (qui aura S+1 cases)
            dist_n = np.zeros(S + 1)
            
            # Astuce de vectorisation : on décale les tableaux avec du padding (des zéros)
            # au lieu de faire une boucle 'for n' lente.
            
            # Terme 1: eta * B(i-1, k, n-1) -> On décale vers la droite
            if i > 0:
                b_prev = couche_S1[i-1]
                dist_n[1:] += eta * b_prev
                
            # Terme 2: eta * B(i, k-1, n) -> Reste à gauche
            if k > 0:
                b_prev = couche_S1[i]
                dist_n[:-1] += eta * b_prev
                
            # Terme 3: eta_bar * B(i-1, k, n) -> Reste à gauche
            if i > 0:
                b_prev = couche_S1[i-1]
                dist_n[:-1] += eta_bar * b_prev
                
            # Terme 4: eta_bar * B(i, k-1, n-1) -> On décale vers la droite
            if k > 0:
                b_prev = couche_S1[i]
                dist_n[1:] += eta_bar * b_prev
                
            # Terme 5 (Interférence quantique): - B(i-1, k-1, n-1)
            if i > 0 and k > 0:
                b_prev_prev = couche_S2[i-1]
                dist_n[1:-1] -= b_prev_prev
                
            # Nettoyage et Normalisation immédiate
            dist_n = np.maximum(dist_n, 0)
            somme = np.sum(dist_n)
            if somme > 0:
                dist_n /= somme
                
            couche_actuelle[i] = dist_n
            banque_theorique[(i, k)] = dist_n
            
        # Rotation des mémoires (On avance d'un pas)
        couche_S2 = couche_S1
        couche_S1 = couche_actuelle

    return banque_theorique

# --- LANCEMENT (Exemple avec N=500 pour tester) ---
N_MAX = 1000 
ETA = 0.3

t0 = time.time()
banque = generer_banque_big_data(N_MAX, ETA)
t1 = time.time()

print(f"\nGénération terminée en {t1 - t0:.2f} secondes !")
print(f"Nombre total de combinaisons générées : {len(banque)}")

# Sauvegarde
with open('banque_fock_big.pkl', 'wb') as fichier:
    pickle.dump(banque, fichier)