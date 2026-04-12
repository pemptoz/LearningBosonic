import sys
import numpy as np
import matplotlib.pyplot as plt
def calculate_bn_ik(n_target, i_target, k_target, eta):
    """
    Calcule le coefficient B_n^{(i,k)} en utilisant la relation de récurrence :
    B(i,k,n) = eta*B(i-1,k,n-1) + eta*B(i,k-1,n) + (1-eta)*B(i-1,k,n) 
               + (1-eta)*B(i,k-1,n-1) - B(i-1,k-1,n-1)
    """
    memo = {}
    eta_bar = 1 - eta

    def get_b(i, k, n):
        # --- Cas de base et conditions aux limites ---
        if n < 0 or i < 0 or k < 0 or n > (i + k):
            return 0.0
        
        # État vide (0 photon en entrée, 0 en sortie)
        if i == 0 and k == 0:
            return 1.0 if n == 0 else 0.0
        
        # Vérification si déjà calculé
        state = (i, k, n)
        if state in memo:
            return memo[state]

        # --- Application de la relation de récurrence ---
        if i > 0 and k > 0:
            res = (eta * get_b(i-1, k, n-1) + 
                   eta * get_b(i, k-1, n) + 
                   eta_bar * get_b(i-1, k, n) + 
                   eta_bar * get_b(i, k-1, n-1) - 
                   get_b(i-1, k-1, n-1))
        elif i > 0: # k est nul
            res = eta * get_b(i-1, 0, n-1) + eta_bar * get_b(i-1, 0, n)
        elif k > 0: # i est nul
            res = eta * get_b(0, k-1, n) + eta_bar * get_b(0, k-1, n-1)
        else:
            res = 0.0

        memo[state] = res
        return res

    return get_b(i_target, k_target, n_target)

# --- Exemple d'utilisation ---

i, k =7, 86
eta = 0.5

liste_bn_ik = []
for n in range(i + k + 1):
    resultat = calculate_bn_ik(n, i, k, eta)
    liste_bn_ik.append(resultat)


plt.figure(figsize=(10, 6))
plt.scatter(range(i + k + 1), liste_bn_ik, marker='o')
plt.title(f"Coefficients B_n^(i={i}, k={k}) pour eta={eta}")
plt.xlabel("n (nombre de photons en sortie)")
plt.ylabel("B_n^(i,k)")
plt.grid()
plt.show()

