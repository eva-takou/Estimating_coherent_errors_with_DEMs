import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import stim
import itertools
from pymatching import Matching
import scipy.linalg
import numpy as np
from scipy.linalg import eigh
import matplotlib
import math
from typing import Any, Sequence

matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif" 


def inv_sqrt_2x2(H):
    a, b = H[0,0], H[0,1]
    c = H[1,1]

    tr  = a + c
    det = a*c - b*np.conj(b)

    s = np.sqrt((tr/2)**2 - det)
    lam1 = tr/2 + s
    lam2 = tr/2 - s

    I = np.eye(2)

    H_inv_sqrt = ((H - lam2*I)/(lam1-lam2))/np.sqrt(lam1) \
               + ((H - lam1*I)/(lam2-lam1))/np.sqrt(lam2)

    return H_inv_sqrt

def kron_n(operators: list):
    
    out = operators[0]
    for op in operators[1:]:
        out = np.kron(out, op)
    return out


def random_cptp_channel(seed=None):
    if seed is not None:
        np.random.seed(seed)    

    G = [np.random.randn(2,2) + 1j*np.random.randn(2,2) for _ in range(4)]
    H = sum(g.conj().T @ g for g in G)
    H_inv_sqrt = inv_sqrt_2x2(H)
    return [g @ H_inv_sqrt for g in G]

def repetition_code_pcm(d):

    H = np.zeros((d-1,d),dtype=np.uint8)
    
    cnt=0
    for k in range(d-1):

        H[k,cnt] = 1
        H[k,cnt+1]=1
        cnt+=1

    return H


def generic_small_noise_channel(epsilon,seed=None):
    
    K_rand = random_cptp_channel(seed)

    K  = [np.sqrt(1-epsilon) * np.eye(2)]
    K += [np.sqrt(epsilon) * k for k in K_rand]

    # print("len Kraus:",len(K))
    
    # U = haar_random_unitary(1)
    # X = np.array([[0,1],[1,0]],dtype=complex)
    # U = linalg.expm(-1j*epsilon*np.pi*X)

    # K += [ np.sqrt(epsilon)* U]
    # K = [U]

    # K0 = np.array([[1,0],[0,np.sqrt(1-epsilon)]])
    # K1 = np.array([[0,np.sqrt(epsilon)],[0,0]])
    # K = [K0,K1]


    # tp_check = sum(k.conj().T @ k for k in K)
    # print("Trace-preserving sum (should be identity):\n", tp_check)

    #Let's switch this to depolarizing noise..
    # I = np.array([[1,0],[0,1]],dtype=complex)
    # X = np.array([[0,1],[1,0]],dtype=complex)
    # Y = np.array([[0,-1j],[1j,0]],dtype=complex)
    # Z = np.array([[1,0],[0,-1]],dtype=complex)


    # K = [np.sqrt(1-epsilon) * np.eye(2)]
    # K +=[np.sqrt(epsilon/3)*X]
    # K +=[np.sqrt(epsilon/3)*Y]
    # K +=[np.sqrt(epsilon/3)*Z]

    # K = [np.sqrt(1-epsilon) * np.eye(2)]
    # K +=[np.sqrt(epsilon)*X]

    # tp_check = sum(k.conj().T @ k for k in K)
    # print("Trace-preserving sum (should be identity):\n", tp_check)
    # print("U:",U)

    return K



def monte_carlo_apply(rho, kraus_list_per_qubit):
    '''Select Kraus A0 ... A_m for each qubit (tensor identities for the rest), based on the probability
    p_k = Tr[A_k \rho A_k^\dagger]
    Renormalize state \rho_after = (A_k \rho A_k^\dagger)/ Tr[A_k \rho A_k^\dagger] and sample the next Kraus for the next qubit.

    I believe in the long limit shot, this should reproduce taking all the combinations of Kraus {A_k}^{x n_Q} in the full channel E(\rho) = \sum_{k_vec} A_{k_vec} \rho A_{k_vec}^\dagger
    '''

    dim = np.shape(rho)[0]
    n   = int(np.log2(dim))
    q   = 0

    for kraus_options in kraus_list_per_qubit:

        oper = [np.eye(2)] * n 
        probs = []
        opers_tot_stored = []

        for m in range(len(kraus_options)):
            oper[q] = kraus_options[m] 
            oper_tot = kron_n(oper)
            probs.append( np.trace(oper_tot @ rho @ oper_tot.conj().T).real)
            opers_tot_stored.append(oper_tot)

        probs=np.array(probs)

        probs /= probs.sum()

        idx = np.random.choice(len(kraus_options), p=probs)

        K = opers_tot_stored[idx]

        rho = K @ rho @ K.conj().T
        rho /= np.trace(rho)

        q+=1
        

    return rho


def sample_data_qubits(rho):
    """
    Sample a computational basis outcome (0...0 to 1...1) from rho.
    Returns a length-d array of 0/1.
    """
    probs = np.real(np.diag(rho))
    outcome_index = np.random.choice(len(probs), p=probs)
    d = int(np.log2(len(probs)))
    outcome = np.array(list(np.binary_repr(outcome_index, width=d)), dtype=int)
    return outcome


def data_to_detection_events(data_qubits):
    """
    Convert Z-basis measurement of data qubits to stabilizer detection events.
    0 = +1, 1 = -1 for Z_i Z_{i+1}
    """
    return (data_qubits[:-1] ^ data_qubits[1:]).astype(np.uint8)

def sample_detection_and_logical_flip(kraus_list):
    """
    Sample both detection events and logical operator flip for a repetition code.
    
    Returns
    -------
    det_events : ndarray of shape (d-1,)
        Stabilizer outcomes 0=+1, 1=-1
    obs_flip : int
        0 = logical operator unchanged, 1 = logical operator flipped
    """
    # initial state
    d = len(kraus_list)
    psi = np.zeros(2**d)
    psi[0] = 1
    rho = np.outer(psi, psi)
    
    # apply noise
    rho_noisy = monte_carlo_apply(rho, kraus_list)

    
    # sample data qubits in Z basis
    data_outcome = sample_data_qubits(rho_noisy)
    
    
    # detection events
    det_events = data_to_detection_events(data_outcome)
    
    # logical operator flip (Z_L)
    obs_flip = np.sum(data_outcome) % 2  # parity of all qubits
    return det_events, obs_flip


def simulate_one_de_pair_fixed_seed(d, epsilon, num_shots,seed=None):
    """Simulate one (d,e) pair: sample detection events, build DEM, decode, return LER."""
    all_det_events = []
    all_obs_flips  = []

    #This is to have a fixed Kraus channel per shot
    kraus_list = []
    for _ in range(d):

        kraus_ops = generic_small_noise_channel(epsilon,seed=None)
        kraus_list.append(kraus_ops)

    for _ in range(num_shots):
        
        det_events, obs_flips = sample_detection_and_logical_flip(kraus_list)
        all_det_events.append(det_events)
        all_obs_flips.append(obs_flips)

    all_det_events = np.array(all_det_events, dtype=np.uint8)
    all_obs_flips = np.array(all_obs_flips, dtype=np.uint8)

    # Estimate vi_mean
    vi_mean = np.sum(all_det_events, axis=0)/num_shots

    # Estimate bulk probabilities
    pij_bulk = {}
    for k in range(d-2):
        vivj = np.sum(all_det_events[:, k] & all_det_events[:, k+1])/num_shots
        numer = vivj - vi_mean[k]*vi_mean[k+1]
        denom = 1 - 2*(vi_mean[k]+vi_mean[k+1]) + 4*vivj
        p = 1/2 - np.sqrt(1/4 - numer/denom)
        pij_bulk[("D"+str(k), "D"+str(k+1))] = p

    # Boundary probabilities
    p_bd = {}
    for indx in [0, d-2]:
        denom = 1
        for key, val in pij_bulk.items():
            if "D"+str(indx) in key:
                denom *= 1 - 2*val
        p_bd[("D"+str(indx))] = 1/2 + (vi_mean[indx]-1/2)/denom
    
    print("(d,e):",(d,epsilon))
    print("bulk:",pij_bulk)
    print("bd:",p_bd)
    print("------")
    # Build stim DEM
    DEM = stim.DetectorErrorModel()
    for key, val in pij_bulk.items():
        targets = [stim.target_relative_detector_id(int(det[1:])) for det in key]
        targets.append(stim.target_logical_observable_id(0))
        DEM.append("error", val, targets)

    for key, val in p_bd.items():
        t = int(key[1:])
        DEM.append("error", val, [stim.target_relative_detector_id(t), stim.target_logical_observable_id(0)])

    # Decode
    pred = Matching.from_detector_error_model(DEM).decode_batch(all_det_events)
    pred = np.array(pred, dtype=np.uint8)

    # Logical error rate
    ler = np.sum(pred.flatten() != all_obs_flips)/num_shots

    return (d, epsilon, ler)


def simulate_one_de_pair_equal_weights_fixed_seed(d, epsilon, num_shots,seed):
    """Simulate one (d,e) pair: sample detection events, build DEM, decode, return LER."""
    all_det_events = []
    all_obs_flips  = []

    #Try this: fix the channel
    kraus_list = []
    for _ in range(d):

        kraus_ops = generic_small_noise_channel(epsilon,seed=None)
        kraus_list.append(kraus_ops)
    
    for _ in range(num_shots):

        det_events, obs_flips = sample_detection_and_logical_flip(kraus_list)
        
        all_det_events.append(det_events)
        all_obs_flips.append(obs_flips)

    all_det_events = np.array(all_det_events, dtype=np.uint8)
    all_obs_flips = np.array(all_obs_flips, dtype=np.uint8)
    
    H = repetition_code_pcm(d)

    # Decode
    logicals = np.ones(d,dtype=np.uint8)
    pred = Matching.from_check_matrix(H,faults_matrix=logicals).decode_batch(all_det_events)
    pred = np.array(pred, dtype=np.uint8)
    

    # Logical error rate
    ler = np.sum(pred.flatten() != all_obs_flips)/num_shots

    return (d, epsilon, ler)



def estimate_and_decode_parallel_fixed_seed(num_shots,seed):
    e_s = [0.2, 0.3, 0.4, 0.5, ]#   #0.6, 0.8, 
    ds  = [3, 5, 7]

    # Prepare all (d,e) pairs
    de_pairs = [(d, e) for d in ds for e in e_s]

    def get_det_events_and_obs_flips(d,epsilon,num_shots,seed):

        all_det_events = []
        all_obs_flips  = []

        #We will fix the Kraus channel per shot but can vary per qubit.
        kraus_list = []

        seeds = [seed + cnt for cnt in range(d)] #Just a choice of how to create different channels per qubit
        for i in range(d):

            kraus_ops = generic_small_noise_channel(epsilon,seeds[i])
            kraus_list.append(kraus_ops)

        for _ in range(num_shots):
            
            det_events, obs_flips = sample_detection_and_logical_flip(kraus_list)
            all_det_events.append(det_events)
            all_obs_flips.append(obs_flips)

        all_det_events = np.array(all_det_events, dtype=np.uint8)
        all_obs_flips = np.array(all_obs_flips, dtype=np.uint8)


        #decode with equal weights:
        H = repetition_code_pcm(d)

        # Decode
        logicals = np.ones(d,dtype=np.uint8)
        pred = Matching.from_check_matrix(H,faults_matrix=logicals).decode_batch(all_det_events)
        pred = np.array(pred, dtype=np.uint8)
        ler_stim = np.sum(pred.flatten() != all_obs_flips)/num_shots

        #---------------------------------- Do the estimation ------------------------------
        vi_mean = np.sum(all_det_events, axis=0)/num_shots

        # Estimate bulk probabilities
        pij_bulk = {}
        for k in range(d-2):
            vivj = np.sum(all_det_events[:, k] & all_det_events[:, k+1])/num_shots
            numer = vivj - vi_mean[k]*vi_mean[k+1]
            denom = 1 - 2*(vi_mean[k]+vi_mean[k+1]) + 4*vivj
            p = 1/2 - np.sqrt(1/4 - numer/denom)
            pij_bulk[("D"+str(k), "D"+str(k+1))] = p

        # Boundary probabilities
        p_bd = {}
        for indx in [0, d-2]:
            denom = 1
            for key, val in pij_bulk.items():
                if "D"+str(indx) in key:
                    denom *= 1 - 2*val
            p_bd[("D"+str(indx))] = 1/2 + (vi_mean[indx]-1/2)/denom
        
        print("(d,e):",(d,epsilon))
        print("bulk:",pij_bulk)
        print("bd:",p_bd)
        print("------")
        # Build stim DEM
        DEM = stim.DetectorErrorModel()
        for key, val in pij_bulk.items():
            targets = [stim.target_relative_detector_id(int(det[1:])) for det in key]
            targets.append(stim.target_logical_observable_id(0))
            DEM.append("error", val, targets)

        for key, val in p_bd.items():
            t = int(key[1:])
            DEM.append("error", val, [stim.target_relative_detector_id(t), stim.target_logical_observable_id(0)])

        # Decode
        pred = Matching.from_detector_error_model(DEM).decode_batch(all_det_events)
        pred = np.array(pred, dtype=np.uint8)
        ler_est = np.sum(pred.flatten() != all_obs_flips)/num_shots

        return d, epsilon, ler_stim, ler_est

    results = Parallel(n_jobs=-1,verbose=10)(
        delayed(get_det_events_and_obs_flips)(d, e, num_shots, seed) for d, e in de_pairs
    )    

    _, _, all_ler_stim, all_ler_est = zip(*results)
    all_ler_stim = np.array(all_ler_stim)
    all_ler_est  = np.array(all_ler_est)

    all_ler_stim = all_ler_stim.reshape(len(ds), len(e_s))
    all_ler_est  = all_ler_est.reshape(len(ds), len(e_s))


    #-------------  Now plot the results -----------------
    
    fig, ax = plt.subplots()

    colors = ["tab:blue","tab:orange","tab:green","tab:red"]
    CNT=0
    for i,d in enumerate(ds):

        ler_per_d = all_ler_stim[i]
        plt.errorbar(np.array(e_s), ler_per_d,
                     yerr=np.sqrt(ler_per_d*(1-ler_per_d)/num_shots),
                     label=f'd={d}, uni. DEM', marker='s',linestyle='--',linewidth=4,markersize=8,color=colors[CNT])
        
        ler_per_d = all_ler_est[i]
        plt.errorbar(np.array(e_s), ler_per_d,
                     yerr=np.sqrt(ler_per_d*(1-ler_per_d)/num_shots),
                     label=f'd={d}, est. DEM', marker='o',markeredgecolor='k',markeredgewidth=2,linestyle='-.',color=colors[CNT])
        
        CNT+=1

    plt.xlabel('$\epsilon$')
    plt.ylabel('$P_L$')
    plt.yscale('log')
    plt.legend(frameon=False,fontsize=13)

    # fig.savefig(f"Arbitrary_CPTP_map_rep_code_est_vs_uniform_seed_{seed}_{num_shots}.pdf",bbox_inches='tight')

    plt.show()

    return 



num_shots    = 20_000
seed         = 4 #seed 3 and 4 for the paper
estimate_and_decode_parallel_fixed_seed(num_shots,seed)


