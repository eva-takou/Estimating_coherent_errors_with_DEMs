import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import stim
from pymatching import Matching
import numpy as np
import matplotlib
import math 
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif" 


X = np.array([[0,1],[1,0]],dtype=complex)
Z = np.array([[1,0],[0,-1]],dtype=complex)
s00 = np.array([[1,0],[0,0]],dtype=complex)
s11 = np.array([[0,0],[0,1]],dtype=complex)
Id = np.eye(2)

Px_p = 1/2*(Id + X)
Px_m = 1/2*(Id - X)
Pz_p = 1/2*(Id + Z)
Pz_m = 1/2*(Id - Z)

#Simulate amplitude damping for phenomenological noise on repetition code

def kron_n(operators):
    '''
    Kron multiple operators in the order A0 \kron A1 kron A2 etc

    Input:
        operators: list of operators to do the kronecker product
    
    Output:
        out: the kronecker product of the operators
    '''

    out = operators[0]
    for op in operators[1:]:
        out = np.kron(out, op)
        
    return out

def repetition_code_pcm(d: int):
    '''
    Parity check matrix of distance d repetition code.

    Input:
        d: distance of the code

    Output:
        H: parity check matrix

    '''

    H = np.zeros((d-1,d),dtype=np.uint8)
    
    cnt=0
    for k in range(d-1):

        H[k,cnt] = 1
        H[k,cnt+1]=1
        cnt+=1

    return H

def amplitude_damping_noise_channel(epsilon: float):
    '''
    Kraus operators for the amplitude damping channel.

    Input:
        epsilon: amplitude damping strength \in[0,1]
    Output:
        K: the two Kraus operators
    '''

    #Amplitude damping
    # H = 1/np.sqrt(2) * np.array([
    #     [1, 1],
    #     [1,-1]
    # ], dtype=complex)    

    K0 = np.array([[1,0],[0,np.sqrt(1-epsilon)]])
    K1 = np.array([[0,np.sqrt(epsilon)],[0,0]])

    # K0 = H @ K0 @ H
    # K1 = H @ K1 @ H
    K = [K0,K1]

    return K


def monte_carlo_apply(psi, kraus_list_per_qubit):
    '''
    Sample a Kraus operator A0 ... Am for each qubit (local channels), based on the probability.
    Apply the Kraus, project the state, and then move on to the next qubit and so on. (quantum trajectory sim)

    Input:
        psi: state vector
        kraus_list_per_qubit: a list of length nQ, where each element is another list with the Kraus operators to sample per qubit

    Output:
        psi: the updated normalized state after sampling a Kraus per qubit
    '''

    dim = psi.shape[0]
    n   = int(np.log2(dim))
    q   = 0

    
    for kraus_options in kraus_list_per_qubit:

        oper = [np.eye(2)] * n 
       
        probs = []
        opers_tot_stored = []

        for m in range(len(kraus_options)):
            oper[q] = kraus_options[m] 
            oper_tot = kron_n(oper)
            
            v = oper_tot @ psi
            probs.append(np.vdot(v, v).real)            
            
            opers_tot_stored.append(oper_tot)

        probs=np.array(probs)

        probs /= probs.sum()

        idx = np.random.choice(len(kraus_options), p=probs)

        K = opers_tot_stored[idx]

        psi = K @ psi 
        psi /= np.linalg.norm(psi)
        
        q+=1
        

    return psi


def build_projectors(nQ: int):
    '''
    Create the projectors which will be used for measurement. 
    
    Input:
        nQ: number of total qubits
    Output:
        P_cache: dictionary with key (basis,qubit_index,0/1), where basis = 'X' or 'Z' , qubit_index is the qubit that is measured and 0/1 is the outcome.
        '''

    P_cache = {}

    bases = {
        'X': (Px_p, Px_m),
        'Z': (Pz_p, Pz_m)
    }

    for basis in ['X', 'Z']:
        Pp, Pm = bases[basis]

        for q in range(nQ):
            ops_p = [Id]*nQ
            ops_m = [Id]*nQ

            ops_p[q] = Pp
            ops_m[q] = Pm

            
            P_cache[(basis, q, 0)] = kron_n(ops_p)
            P_cache[(basis, q, 1)] = kron_n(ops_m)


    return P_cache


def measure_qubit_and_project(psi, nQ, qubit_index: int, basis: str, P_cache: dict):
    '''
    Measure a qubit in Z/X basis and update the state psi.

    Inputs:
        psi: the input state vector
        nQ: the # of qubits
        qubit_index: the qubit to measure
        basis: 'Z' or 'X'
        P_cache: precomputed projectors to use (see build_projectors output)
    Outputs:
        psi: the normalized state vector after projecting the measured qubit based on the measurement outcome
        outcome: the measurement outcome of the qubit
    '''
    Pp = P_cache[(basis, qubit_index, 0)]
    Pm = P_cache[(basis, qubit_index, 1)]

    v     = Pp @ psi
    prob0 = np.vdot(v,v).real
    
    if np.random.random() < prob0:
        outcome = 0
        psi = Pp @ psi
    else:
        outcome = 1
        psi = Pm @ psi
    
    psi /=np.linalg.norm(psi)

    return psi, outcome            


def sample_data_qubits_from_full_psi(psi, d: int, P_cache: dict):
    '''
    Measure the data qubits of repetition code and collect the outcomes.

    Inputs:
        psi: the state vector
        d: distance of the code:
        P_cache: dictionary which stores precomputed projectors 

    Output:
        outcomes: array of outcomes per data qubit
    
    '''
    '''Let's just measure one by one by collapsing the state.'''

    nQ   = d+d-1
    outcomes = []
    for k in range(d):
    
        psi,outcome = measure_qubit_and_project(psi,nQ,qubit_index=k,basis='Z',P_cache=P_cache)
        outcomes.append(outcome)
    
    return np.array(outcomes)


def data_to_detection_events(data_outcomes):
    """
    Convert Z-basis measurement of data qubits to the stabilizer values Z_{i}Z_{i+1} (or X_{i}X_{i+1}) by xoring consecutive measurement outcomes
    
    Input:
        data_outcomes: the outcomes of data qubits
    Output:
        the reconstructed stabilizer values
    """

    return (data_outcomes[:-1] ^ data_outcomes[1:]).astype(np.uint8)


def sample_detection_and_logical_flip(d: int, basis: str, initial_state: str, kraus_list, P_cache, Had):
    """
    Sample both detection events and logical operator flip for a repetition code. This constitutes one shot of the Monte Carlo simulation.

    Inputs:
        d: distance of the repetition code
        basis: 'X' or 'Z'
        initial_state: '0' or '1' (for 'Z') or '+' for 'X'
        kraus_list: the kraus operators to sample from per qubit (list)
        P_cache: precomputed measurement projectors
        Had: Hadamard gate for all qubits (\tensor_{i=1}^n H)
        

    Outputs:
        det_events: detection events (length # of detectors)
        obs_flips: 0/1 whether or not the logical observable was flipped

    """

    # Create the initial state

    nQ      = d       #total number of qubits for repetition code
    psi     = np.zeros(2**nQ)

    if initial_state=='0': # The all |0> state
        psi[0] = 1 
    elif initial_state=='1': # The |1> state for all qubits
        psi[-1] = 1
    elif initial_state=='+': # The all |+> state for both ancilla and data
        psi = np.ones(2**nQ, dtype=complex) / np.sqrt(2**nQ)


    psi = monte_carlo_apply(psi, kraus_list) #Apply noise

    # Return data to Z basis before measurement
    if basis=='X':
        psi = Had @ psi

    # Measure data in Z basis
    data_outcome = sample_data_qubits_from_full_psi(psi,d,P_cache) 

    # Convert data outcomes to effective stabilizer
    det_events = data_to_detection_events(data_outcome) 

    det_events     = np.array(det_events)

    obs_flip = np.sum(data_outcome) % 2  #logical operator flip
    
    return det_events, obs_flip


def estimate_DEM(d: int, detection_events):
    '''
    Estimate repetition code detector error model from detection events.

    Input:
        d: distance of code
        detection_events: the detection events (shape: # of shots x # of detectors)
    
    Output:
        DEM: the estimated DEM
        pij_space: dictionary of estimated space-bulk probabilities
        pi: dictionary of estimated space-boundary probabilities
    '''

    #start with data qubit bulk edges
    pij_space={}
    pi={}
    n_anc = d-1 
    
    num_shots = np.shape(detection_events)[0]
    vi_mean   = np.sum(detection_events, axis=0)/num_shots

    print("vi_mean:",vi_mean)


    #bulk space edges

    for anc in range(n_anc-1):

        indx1 = anc 
        indx2 = (anc+1) 
        det_names = tuple(["D"+str(indx1),"D"+str(indx2)])
        
        vi = vi_mean[indx1]
        vj = vi_mean[indx2]
        vivj = np.sum(detection_events[:, indx1] & detection_events[:, indx2])/num_shots
        print("vi,vj,vivj:",(vi,vj,vivj), "for (indx1,indx2):",(indx1,indx2))
        numer = vivj - vi*vj 
        denom = 1-2*(vi+vj)+4*vivj
        p = 1/2 - np.sqrt(1/4-numer/denom)
        if p<0 or math.isnan(p) or p>1:
            p=0
        pij_space[det_names] = p


    #bd edges
    idxs_bd = [0,d-2] #d=3, we have ancilla=0, and ancilla=1, d=5, we have ancilla=0 and ancilla=3

    for anc in idxs_bd:

        INDX = anc 
        vi = vi_mean[INDX]

        det_name = "D"+str(INDX)
        denom = 1 

        for key,val in pij_space.items():
            if det_name in key:
                denom *= 1-2*val 



        p=1/2 + (vi-1/2)/denom
        if p<0 or math.isnan(p) or p>1:
            p=0            
        
        pi[tuple([det_name])] = p

    #Now build the DEM so that we pass this directly to pymatching
    DEM = stim.DetectorErrorModel()

    for key, val in pij_space.items():
        targets = [stim.target_relative_detector_id(int(det[1:])) for det in key]
        targets.append(stim.target_logical_observable_id(0))
        try:
            if math.isnan(val):
                raise ValueError("nan value in space prob")
            DEM.append("error", val, targets)
        except ValueError:
            DEM.append("error", 0*val, targets)
            # raise Exception("something wrong in pij_space:",(key,val))


    for key, val in pi.items():
        t = int(key[0][1:])
        try:
            if math.isnan(val):
                raise ValueError("nan value in bd prob")            

            DEM.append("error", val, [stim.target_relative_detector_id(t), stim.target_logical_observable_id(0)])
        except ValueError:
            DEM.append("error", 0*val, [stim.target_relative_detector_id(t), stim.target_logical_observable_id(0)])
            # raise Exception("something wrong in pi_bd:",(key,val))


    return DEM, pij_space,pi


def fixed_seed_epsilons(n: int, epsilon: float, sigma: float):
    '''
    Sample n fixed seed epsilon values around a mean value epsilon, with standard deviation sigma.

    Input:
        n: how many epsilon values to sample
        epsilon: the mean of the normal distribution
        sigma: the standard deviation
    
    Output:
        epsilon_sample: the sampled values (length n)
    '''

    def sample_epsilons(n, eps, sigma, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=eps, scale=sigma, size=n)

    epsilon_sample = sample_epsilons(n,epsilon,sigma,seed=1)
    
    
    return epsilon_sample


def estimate_and_decode_parallel_fixed_seed(num_shots: int, sigma=1e-3):
    '''
    Decode the estimated DEM of repetition code, as well as the uniform weight DEM with pymatching and plot the LERs.
    The noise model is phenomenological amplitude damping. Syndrome extraction is repeated for d rounds per distance.

    Input:
        basis: 'Z' or 'X'
        initial_state: '1' or '0' for 'Z' basis, and '+' for 'X' basis
        num_shots: number of Monte Carlo shots 
        sigma: standard deviation (if we want the \epsilon values per qubit to differ)

    '''

    e_s = [0.1, 0.2, 0.3, 0.4,  0.6 ]  
    ds  = [3,5,7]

    # Prepare all (d,e) pairs for the parallel simulation
    de_pairs = [(d, e) for d in ds for e in e_s]

    def get_det_events_and_obs_flips(d,epsilon,basis,initial_state,num_shots,sigma):
        
        nQ          = d 
        P_cache     = build_projectors(nQ)
        
        Had  = 1/np.sqrt(2)*np.array([[1,1],[1,-1]],dtype=complex)
        Had = [Had]*nQ 
        Had = kron_n(Had)        


        #We will fix the Kraus channel per shot but can vary per qubit.
        kraus_list = []

        epsilons=fixed_seed_epsilons(nQ,epsilon,sigma)
        
        for i in range(nQ):

            kraus_ops = amplitude_damping_noise_channel(epsilons[i])
            
            kraus_list.append(kraus_ops)
        

        all_det_events = []
        all_obs_flips  = []

        for _ in range(num_shots):
            
            det_events, obs_flips = sample_detection_and_logical_flip(d, basis, initial_state, kraus_list,P_cache,Had)
            all_det_events.append(det_events)
            all_obs_flips.append(obs_flips)

        all_det_events = np.array(all_det_events, dtype=np.uint8)
        all_obs_flips  = np.array(all_obs_flips, dtype=np.uint8)


        # Decode
        H = repetition_code_pcm(d)
        logicals = np.ones(d,dtype=np.uint8)
        pred = Matching.from_check_matrix(H,faults_matrix=logicals).decode_batch(all_det_events) 
        pred = np.array(pred, dtype=np.uint8)

        if initial_state=='1':
            all_obs_flips = all_obs_flips^1

        ler_stim = np.sum(pred.flatten() != all_obs_flips)/num_shots

        #-- Estimate ------------
        DEM, pij_space, pi = estimate_DEM(d,all_det_events)

        print("DEM:",DEM)        
        print("(d,e):",(d,epsilon))
        print("bulk space:",pij_space)
        print("bd:",pi)
        print("------")
 
        # Decode
        pred = Matching.from_detector_error_model(DEM).decode_batch(all_det_events)
        pred = np.array(pred, dtype=np.uint8)
        ler_est = np.sum(pred.flatten() != all_obs_flips)/num_shots

        return d, epsilon, ler_stim, ler_est


    basis         = 'Z'
    initial_state = '1'
    results = Parallel(n_jobs=-1,verbose=10)(delayed(get_det_events_and_obs_flips)(d, e, basis, initial_state, num_shots, sigma) for d, e in de_pairs)    

    _, _, all_ler_stim, all_ler_est = zip(*results)
    all_ler_stim = np.array(all_ler_stim)
    all_ler_est  = np.array(all_ler_est)

    all_ler_stim = all_ler_stim.reshape(len(ds), len(e_s))
    all_ler_est  = all_ler_est.reshape(len(ds), len(e_s))


    #-------------  Now plot the results -----------------
    
    fig, ax = plt.subplots()

    colors = ["tab:blue","tab:orange","tab:green"] 
    CNT    = 0

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


    basis         = 'X'
    initial_state = '+'
    results = Parallel(n_jobs=-1,verbose=10)(delayed(get_det_events_and_obs_flips)(d, e, basis, initial_state, num_shots, sigma) for d, e in de_pairs)    

    
    _, _, all_ler_stim, all_ler_est = zip(*results)
    all_ler_stim = np.array(all_ler_stim)
    all_ler_est  = np.array(all_ler_est)

    all_ler_stim = all_ler_stim.reshape(len(ds), len(e_s))
    all_ler_est  = all_ler_est.reshape(len(ds), len(e_s))


    colors = ["#1f4e79","#cc6600" ,"#1b5e20" ]#,"darkorange"

    CNT    = 0

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


    plt.xlabel('$\epsilon_{AD}$')
    plt.ylabel('$P_L$')
    plt.yscale('log')
    plt.legend(frameon=False,fontsize=13,loc='best',ncols=2,bbox_to_anchor=(0.9, 1.17))
    plt.grid(True)



    plt.show()

    return 




estimate_and_decode_parallel_fixed_seed(num_shots=30_000,sigma=0)



