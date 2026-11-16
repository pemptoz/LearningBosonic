import numpy as np
import scipy.special as sep
from scipy.optimize import minimize
from scipy.linalg import expm
import time


G0 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]) 
G1 = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]) 
G2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]]) 
G3 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]) 

def parametres_to_W(params):
    t0, t1, t2, t3 = params
    A = t0*G0 + t1*G1 + t2*G2 + t3*G3
    return expm(A)


i_fock = 15   
k_fock = 22  
N_shots = 30000

vraies_phases = np.random.uniform(-1.5, 1.5, 4)
S_phys = parametres_to_W(vraies_phases)

print(np.round(S_phys, 3))

t_start = time.time()

R_A = np.random.gamma(i_fock + 1, 1.0, N_shots)
theta_A = np.random.uniform(0, 2*np.pi, N_shots)

R_B = np.random.gamma(k_fock + 1, 1.0, N_shots)
theta_B = np.random.uniform(0, 2*np.pi, N_shots)


q0 = np.sqrt(2 * R_A) * np.cos(theta_A)
p0 = np.sqrt(2 * R_A) * np.sin(theta_A)

# Mode B
q1 = np.sqrt(2 * R_B) * np.cos(theta_B)
p1 = np.sqrt(2 * R_B) * np.sin(theta_B)

# État d'entrée pur
X_in = np.array([q0, p0, q1, p1])


X_obs = S_phys @ X_in


"""def cout_covariance_rayons(params):
    W = parametres_to_W(params)
    Y = W @ X_obs
    r0_sq, r1_sq = Y[0]**2 + Y[1]**2, Y[2]**2 + Y[3]**2
    return np.cov(r0_sq, r1_sq)[0, 1]**2"""
def cout_covariance_rayons(params):
    W = parametres_to_W(params)
    Y = W @ X_obs
    r0_sq, r1_sq = Y[0] ** 2 + Y[1] ** 2, Y[2] ** 2 + Y[3] ** 2

    corr_matrix = np.corrcoef(r0_sq, r1_sq)
    return corr_matrix[0, 1] ** 2
meilleur_score = np.inf
meilleurs_params = None
points_depart = [np.zeros(4)] + [np.random.uniform(-2, 2, 4) for _ in range(4)]

for start in points_depart:
    res = minimize(cout_covariance_rayons, start, method='BFGS', 
                   options={'maxiter': 15000, 'gtol': 1e-8})
    if res.fun < meilleur_score:
        meilleur_score = res.fun
        meilleurs_params = res.x

W_opt = parametres_to_W(meilleurs_params)

print(np.round(W_opt, 3))
print(np.round(W_opt @ S_phys, 3))


Y_final = W_opt @ X_obs
r0_sq = Y_final[0]**2 + Y_final[1]**2
r1_sq = Y_final[2]**2 + Y_final[3]**2

if np.mean(r0_sq) < np.mean(r1_sq):
    rA_sq, rB_sq = r0_sq, r1_sq
else:
    rA_sq, rB_sq = r1_sq, r0_sq

def tomographie_mle(r_sq, N_max=40):
    N_data = len(r_sq)
    M = np.zeros((N_data, N_max + 1))
    
    for n in range(N_max + 1):
        log_Q = n * np.log(r_sq / 2 + 1e-15) - sep.gammaln(n + 1) - r_sq / 2 - np.log(np.pi)
        M[:, n] = np.exp(log_Q)
        
    p_est = np.ones(N_max + 1) / (N_max + 1)
    
    for k in range(15000): 
        proj = M @ p_est
        proj = np.maximum(proj, 1e-15)
        
        p_est = p_est * (M.T @ (1.0 / proj)) / N_data
        p_est /= np.sum(p_est)
        
        if k % 1000 == 0 and k > 0:
            p_est[p_est < 1e-4] = 0 
            p_est /= np.sum(p_est)
            
    return p_est
N_max_reco = i_fock+k_fock + 10 
p_A = tomographie_mle(rA_sq, N_max=N_max_reco)
p_B = tomographie_mle(rB_sq, N_max=N_max_reco)

fidelite_A = p_A[i_fock]
fidelite_B = p_B[k_fock]

print(f"\nMode 0 (Cible |{i_fock}>) -> Fidélité : {fidelite_A*100:.2f}% (Infidélité: {1-fidelite_A:.4f})")
print(f"Mode 1 (Cible |{k_fock}>) -> Fidélité : {fidelite_B*100:.2f}% (Infidélité: {1-fidelite_B:.4f})")
print(f"Fidélité globale théorique : {fidelite_A * fidelite_B * 100:.2f}%")