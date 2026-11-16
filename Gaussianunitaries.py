import numpy as np
import matplotlib.pyplot as plt
from qutip import fock, tensor, displace, ptrace, destroy, expect, wigner

def simuler_wigner_deplacement(n, m, alpha1, alpha2, cutoff=80):
    # 1. État initial
    etat_initial = tensor(fock(cutoff, n), fock(cutoff, m))
    
    # 2. Déplacement
    D_total = tensor(displace(cutoff, alpha1), displace(cutoff, alpha2))
    etat_final = D_total * etat_initial
    
    # 3. Trace Partielle (Mode 0)
    rho1 = ptrace(etat_final, 0)

    vec = np.linspace(-10, 10, 200)
    W = wigner(rho1, vec, vec) 
    
    X, P = np.meshgrid(vec, vec)
    plt.figure(figsize=(8, 6))

    cp = plt.contourf(X, P, W, levels=50, cmap='RdBu_r') 
    plt.colorbar(cp)
    
    plt.plot([alpha1.real*np.sqrt(2)], [alpha1.imag*np.sqrt(2)], 'ro', label="Cible théorique")
    
    plt.title(f"Wigner de la sortie 1 (Mode 0)")
    plt.xlabel('X')
    plt.ylabel('P')
    plt.legend()
    plt.show()

    a = destroy(cutoff)
    x_moy = expect((a + a.dag()), rho1) / np.sqrt(2)
    p_moy = expect(1j * (a.dag() - a), rho1) / np.sqrt(2)

    print(f"Position X : {x_moy:.3f} (Attendu : {alpha1.real})")
    print(f"Impulsion P : {p_moy:.3f} (Attendu : {alpha1.imag})")

simuler_wigner_deplacement(n=0, m=0, alpha1=2.0 + 1j, alpha2=0)