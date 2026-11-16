import numpy as np
import qutip as qt
from scipy.special import factorial
from scipy.optimize import minimize_scalar
import scipy.special as sep


N_dim = 30      
i_fock = 8       # État de Fock en entrée du Mode 0 
k_fock = 12       # État de Fock en entrée du Mode 1 
N_shots = 10000   

theta_phys = np.radians(150) 


print(f"Vrai angle = {np.degrees(theta_phys)}°)")

a1 = qt.tensor(qt.destroy(N_dim), qt.qeye(N_dim))
a2 = qt.tensor(qt.qeye(N_dim), qt.destroy(N_dim))

H_bs = a1.dag() * a2 - a1 * a2.dag()
U_quantum = (theta_phys * H_bs).expm()


psi_in = qt.tensor(qt.fock(N_dim, i_fock), qt.fock(N_dim, k_fock))
psi_out = U_quantum * psi_in


coeffs = psi_out.full().reshape(N_dim, N_dim)


fock_indices = np.arange(N_dim)
sqrt_fact = np.sqrt(factorial(fock_indices))

def log_Q_husimi(coords):
    q0, p0, q1, p1 = coords
    alpha0 = (q0 - 1j*p0) / np.sqrt(2)
    alpha1 = (q1 - 1j*p1) / np.sqrt(2)
    
    v0 = (np.conj(alpha0) ** fock_indices) / sqrt_fact
    v1 = (np.conj(alpha1) ** fock_indices) / sqrt_fact
    amplitude = v0 @ coeffs @ v1
    
    prob = np.abs(amplitude)**2 * np.exp(-(q0**2 + p0**2 + q1**2 + p1**2)/2) / (np.pi**2)
    return np.log(prob) if prob > 1e-20 else -np.inf

thinning = 25  

X_list = []
current_coords = np.array([0.0, 0.0, 0.0, 0.0])
current_log_p = log_Q_husimi(current_coords)

for _ in range(2000):
    proposal = current_coords + np.random.normal(0, 0.6, 4)
    proposal_log_p = log_Q_husimi(proposal)
    if np.log(np.random.uniform(0, 1)) < proposal_log_p - current_log_p:
        current_coords = proposal
        current_log_p = proposal_log_p


for shot in range(N_shots):
    for _ in range(thinning):  
        proposal = current_coords + np.random.normal(0, 0.6, 4)
        proposal_log_p = log_Q_husimi(proposal)
        if np.log(np.random.uniform(0, 1)) < proposal_log_p - current_log_p:
            current_coords = proposal
            current_log_p = proposal_log_p
    
    X_list.append(current_coords.copy())

X_obs = np.array(X_list).T

print(X_obs.shape)


def cost_function_direct(theta):
    """Calcul Corrélation pour ICA """
    c, s = np.cos(theta), np.sin(theta)
    W = np.array([
        [c, 0, -s, 0],
        [0, c, 0, -s],
        [s, 0, c, 0],
        [0, s, 0, c]
    ])
    Y = W @ X_obs
    r0_sq = Y[0]**2 + Y[1]**2
    r1_sq = Y[2]**2 + Y[3]**2
    return np.cov(r0_sq, r1_sq)[0, 1]**2

res = minimize_scalar(cost_function_direct, bounds=(-np.pi, np.pi), method='bounded')
theta_opt = res.x
print( np.degrees(theta_opt))


c_opt, s_opt = np.cos(theta_opt), np.sin(theta_opt)
W_opt = np.array([
    [c_opt, 0, -s_opt, 0],
    [0, c_opt, 0, -s_opt],
    [s_opt, 0, c_opt, 0],
    [0, s_opt, 0, c_opt]
])
Y_final = W_opt @ X_obs

r0_sq_check = Y_final[0]**2 + Y_final[1]**2
r1_sq_check = Y_final[2]**2 + Y_final[3]**2

if np.mean(r0_sq_check) < np.mean(r1_sq_check):
    q_A, p_A = Y_final[0], Y_final[1]
    q_B, p_B = Y_final[2], Y_final[3]
else:
    q_A, p_A = Y_final[2], Y_final[3]
    q_B, p_B = Y_final[0], Y_final[1]


def tomographie_mle_fock(q, p, target_fock, N_max=18):
    r_sq = q**2 + p**2
    N_shots = len(r_sq)
    
    M = np.zeros((N_shots, N_max + 1))
    for n in range(N_max + 1):
        log_Q = n * np.log(r_sq / 2 + 1e-15) - sep.gammaln(n + 1) - r_sq / 2 - np.log(np.pi)
        M[:, n] = np.exp(log_Q)
        
    p_estimate = np.ones(N_max + 1) / (N_max + 1) 
    
    for _ in range(2000): 
        projections = M @ p_estimate
        projections = np.maximum(projections, 1e-15)
        p_estimate = p_estimate * (M.T @ (1.0 / projections)) / N_shots
        p_estimate /= np.sum(p_estimate) 
        
    fidelity = p_estimate[target_fock]
    infidelity = 1.0 - fidelity
    
    p_true = np.zeros(N_max + 1)
    p_true[target_fock] = 1.0
    trace_distance = 0.5 * np.sum(np.abs(p_estimate - p_true))
    
    return p_estimate, infidelity, trace_distance

p_A, infidelite_A, dist_trace_A = tomographie_mle_fock(q_A, p_A, target_fock=i_fock)
p_B, infidelite_B, dist_trace_B = tomographie_mle_fock(q_B, p_B, target_fock=k_fock)

print(f" Statistiques Mode 0 (Attendu: |{i_fock}>) :")
print(f" Infidélité {infidelite_A:.4f}")
print(f"Distance trace  :{dist_trace_A:.4f}")
print(f" Top 3 des populations Fock reconstruites :")
for idx in np.argsort(p_A)[::-1][:3]:
    print(f"     |{idx}> : {p_A[idx]*100:.2f}%")

print(f"\n Statistiques Mode 1 (Attendu: |{k_fock}>) :")
print(f" Infidélité  {infidelite_B:.4f}")
print(f" Distance trace  : {dist_trace_B:.4f}")
print(f" Top 3 des populations Fock reconstruites :")
for idx in np.argsort(p_B)[::-1][:3]:
    print(f"     |{idx}> : {p_B[idx]*100:.2f}%")

infidelite_totale = 1.0 - ((1.0 - infidelite_A) * (1.0 - infidelite_B))
print(f"Infidélité totale  {infidelite_totale:.4f}")
