import sys
import numpy as np
import pickle

# Augmenter la limite de récursion
sys.setrecursionlimit(2000)

def get_b_dist(i_target, k_target, eta):
    memo = {}
    eta_bar = 1 - eta
    def get_b(i, k, n):
        if n < 0 or i < 0 or k < 0 or n > (i + k): return 0.0
        if i == 0 and k == 0: return 1.0 if n == 0 else 0.0
        state = (i, k, n)
        if state in memo: return memo[state]
        if i > 0 and k > 0:
            res = (eta * get_b(i-1, k, n-1) + eta * get_b(i, k-1, n) +
                   eta_bar * get_b(i-1, k, n) + eta_bar * get_b(i, k-1, n-1) -
                   get_b(i-1, k-1, n-1))
        elif i > 0: res = eta * get_b(i-1, 0, n-1) + eta_bar * get_b(i-1, 0, n)
        elif k > 0: res = eta * get_b(0, k-1, n) + eta_bar * get_b(0, k-1, n-1)
        else: res = 0.0
        memo[state] = res
        return res
    
    N = i_target + k_target
    dist = [get_b(i_target, k_target, n) for n in range(N + 1)]
    return np.array(dist)

N_MAX = 60
ETA = 0.3
banque_theorique = {}


for N in range(N_MAX + 1):
    for i in range(N + 1):
        k = N - i
    
        dist = get_b_dist(i, k, ETA)
        dist = np.maximum(dist, 0)
        dist /= np.sum(dist)
        
        banque_theorique[(i, k)] = dist

with open('banque_fock.pkl', 'wb') as fichier:
    pickle.dump(banque_theorique, fichier)
