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

def CNOT_ij(nQ: int, control: int, target: int):
    '''
    CNOT gate promoted in arbitrary qubit dimensions. CNOT = s00_{control} \otimes 1_{target} + s11_{control} \otimes X_{target}

    Input:
        nQ: total number of qubits
        control: control qubit
        target: target qubit

    Output:
        CNOT: the CNOT matrix

    '''

    if control>(nQ-1) or target>(nQ-1):
        raise ValueError("Control or target qubit index out of bounds for input nQ qubits.")
    
    Ids          = [np.eye(2)]*nQ
    Ids[control] = s00

    CNOT = kron_n(Ids) 
    
    Ids[control] = s11
    Ids[target]  = X

    CNOT += kron_n(Ids) 

    return CNOT

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


def measure_ancilla_qubits(d: int, basis: str, psi, P_cache: dict):
    '''
    Measure the ancilla qubits one by one and collapse the state.
    
    Inputs:
        d: distance of repetition code
        basis: 'X' or 'Z'
        psi: the input state vector
        P_cache: dictionary storing precomputed measurement projectors
    Outputs:
        ancilla_det_events: measurement outcomes of ancilla
        psi: the updated normalized state
    '''
    nQ = d + d-1

    ancilla_det_events = [] #to store all ancilla det events
    for k in range(d-1):

        psi,outcome = measure_qubit_and_project(psi,nQ,qubit_index=k+d,basis=basis,P_cache=P_cache)
        ancilla_det_events.append(outcome)
    
    #if basis = 'X' reset to |+> state
    #if basis = 'Z' reset to |1> state (this is valid if initial state of ancilla is |1>, if we switch to initial state of ancilla to |0>, then we want to reprepare to |0>)

    if basis=='Z': 
        Xs = [np.eye(2)]*nQ 

        for k, outcome in enumerate(ancilla_det_events):
            if outcome == 0: #flip to |1>
                Xs[k + d] = X

        Xs = kron_n(Xs) 
        psi = Xs @ psi

    elif basis=='X': #if it was |->, apply Z to bring it to |+> 

        Zs = [np.eye(2)]*nQ 

        for k, outcome in enumerate(ancilla_det_events):
            if outcome == 1:
                Zs[k + d] = Z 

        Zs  = kron_n(Zs) 
        psi = Zs @ psi

    return ancilla_det_events,psi


def sample_detection_and_logical_flip(d: int, rds: int, basis: str, initial_state: str, kraus_list, P_cache,Had,CNOTs):
    """
    Sample both detection events and logical operator flip for a repetition code. This constitutes one shot of the Monte Carlo simulation.

    Inputs:
        d: distance of the repetition code
        rds: # of syndrome extraction rounds
        basis: 'X' or 'Z'
        initial_state: '0' or '1' (for 'Z') or '+' for 'X'
        kraus_list: the kraus operators to sample from per qubit (list)
        P_cache: precomputed measurement projectors
        Had: Hadamard gate for all qubits (\tensor_{i=1}^n H)
        CNOTS: the CNOT schedule

    Outputs:
        all_det_events: detection events (length # of detectors)
        obs_flips: 0/1 whether or not the logical observable was flipped

    """

    # Create the initial state

    nQ      = d + (d-1)       #total number of qubits for repetition code
    psi     = np.zeros(2**nQ)

    if initial_state=='0': # The all |0> state
        psi[0] = 1 
    elif initial_state=='1': # The |1> state for all qubits
        psi[-1] = 1
    elif initial_state=='+': # The all |+> state for both ancilla and data
        psi = np.ones(2**nQ, dtype=complex) / np.sqrt(2**nQ)


    #Do syndrome extraction for r rounds: 
    all_det_events = [] 
    for r in range(rds):
        
        psi = monte_carlo_apply(psi, kraus_list) #Apply noise
        psi = CNOTs @ psi #Apply CNOTs (they are perfect so we can apply them in 1 step)

        ancilla_det_events, psi = measure_ancilla_qubits(d,basis,psi, P_cache) #Measure the ancilla only (Z or X basis), and reset
        
        all_det_events.append(ancilla_det_events)

    # Return data to Z basis before measurement
    if basis=='X':
        psi = Had @ psi

    # Measure data in Z basis
    data_outcome = sample_data_qubits_from_full_psi(psi,d,P_cache) 

    # Convert data outcomes to effective stabilizer
    det_events = data_to_detection_events(data_outcome) 

    det_events     = np.array(det_events)
    all_det_events = np.array(all_det_events)

    if initial_state=='1': #ancilla start from 1, so we should xor all entries to redefine according to the error-free case
        all_det_events ^= 1   

    all_det_events = np.vstack([all_det_events, det_events])

    all_det_events[1:] ^= all_det_events[:-1] #d_{r} := m_{r} \oplus m_{r-1}


    all_det_events= all_det_events.flatten() #now flatten this as [ancillas_rd=0, ancillas_rd=1,...] 

    obs_flip = np.sum(data_outcome) % 2  #logical operator flip
    
    return all_det_events, obs_flip


def estimate_DEM(d: int, rds: int, detection_events):
    '''
    Estimate repetition code detector error model from detection events.

    Input:
        d: distance of code
        rds: rounds of syndrome extraction
        detection_events: the detection events (shape: # of shots x # of detectors)
    
    Output:
        DEM: the estimated DEM
        pij_space: dictionary of estimated space-bulk probabilities
        pij_time: dictionary of estimated time probabilities
        pi: dictionary of estimated space-boundary probabilities
    '''

    #start with data qubit bulk edges
    pij_time={}
    pij_space={}
    pi={}
    n_anc = d-1 
    
    num_shots = np.shape(detection_events)[0]
    vi_mean   = np.sum(detection_events, axis=0)/num_shots

    print("vi_mean:",vi_mean)

    print("shape of dete events:",np.shape(detection_events),"for rds:",rds)

    #bulk space edges
    for rd in range(rds+1): #effective rds (+1 for last data qubit measurements)

        for anc in range(n_anc-1):

            indx1 = anc + n_anc * rd
            indx2 = (anc+1) + n_anc * rd
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
            

    #bulk time edges
    for rd in range(rds): 

        for anc in range(n_anc):

            indx1 = anc + n_anc * rd
            indx2 = anc + n_anc * (rd+1)
            det_names = tuple(["D"+str(indx1),"D"+str(indx2)])
            
            vi = vi_mean[indx1]
            vj = vi_mean[indx2]
            vivj = np.sum(detection_events[:, indx1] & detection_events[:, indx2])/num_shots
            print("vi,vj,vivj:",(vi,vj,vivj), "for (indx1,indx2):",(indx1,indx2))
            numer = vivj - vi*vj 
            denom = 1-2*(vi+vj)+4*vivj
            p=1/2 - np.sqrt(1/4-numer/denom)
            if p<0 or math.isnan(p) or p>1:
                p=0            
            pij_time[det_names] = p

    #bd edges
    idxs_bd = [0,d-2] #d=3, we have ancilla=0, and ancilla=1, d=5, we have ancilla=0 and ancilla=3

    for rd in range(rds+1):

        for anc in idxs_bd:

            INDX = anc + n_anc * rd 
            vi = vi_mean[INDX]

            det_name = "D"+str(INDX)
            denom = 1 

            for key,val in pij_space.items():
                if det_name in key:
                    denom *= 1-2*val 

            for key,val in pij_time.items():
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

    for key, val in pij_time.items():
        targets = [stim.target_relative_detector_id(int(det[1:])) for det in key]
        try:
            if math.isnan(val):
                raise ValueError("nan value in time prob")            
            DEM.append("error", val, targets)
        except ValueError:
            DEM.append("error", 0*val, targets)
            # raise Exception("something wrong in pij_time:",(key,val))

    for key, val in pi.items():
        t = int(key[0][1:])
        try:
            if math.isnan(val):
                raise ValueError("nan value in bd prob")            

            DEM.append("error", val, [stim.target_relative_detector_id(t), stim.target_logical_observable_id(0)])
        except ValueError:
            DEM.append("error", 0*val, [stim.target_relative_detector_id(t), stim.target_logical_observable_id(0)])
            # raise Exception("something wrong in pi_bd:",(key,val))


    return DEM, pij_space, pij_time,pi

def get_DEM_ideal(d: int, rds: int, epsilons):
    '''
    Get the ground-truth DEM with rate \epsilon_j for space-like and time-like edges of repetition code (besides last reconstructed round space edges whose rate is 0).
    
    Input:
        d: distance of code
        rds: rounds of syndrome extraction
        epsilons: list of amplitude damping strengths (first d are for data qubits, next d+1 for ancilla qubits)
    
    Output:
        DEM: the uniform DEM
        pij_space: dictionary of space-bulk probabilities
        pij_time: dictionary of time probabilities
        pi: dictionary of space-boundary probabilities
    '''

    
    pij_time={}
    pij_space={}
    pi={}
    n_anc  = d-1 
    n_data = d

    epsilons_bulk_space = epsilons[1:n_data-1]             #ignore the boundaries
    epsilons_bd_space   = [epsilons[0],epsilons[n_data-1]] #keep only the boundaries
    epsilons_time       = epsilons[n_data:]                #keep only of ancilla qubits

    #bulk space edges
    for rd in range(rds+1): #effective rds (+1 for last data qubit measurements)

        for anc in range(n_anc-1):

            indx1 = anc + n_anc * rd
            indx2 = (anc+1) + n_anc * rd
            det_names = tuple(["D"+str(indx1),"D"+str(indx2)])
            
            if rd==rds:
                pij_space[det_names] = 0 #last data qubit measurements round (edges are 0)
            else:
                pij_space[det_names] = epsilons_bulk_space[anc]
                

    #bulk time edges
    for rd in range(rds): 

        for anc in range(n_anc):

            indx1 = anc + n_anc * rd
            indx2 = anc + n_anc * (rd+1)
            det_names = tuple(["D"+str(indx1),"D"+str(indx2)])
            
            pij_time[det_names] = epsilons_time[anc]

    #bd edges
    idxs_bd = [0,d-2] 

    for rd in range(rds+1):

        for anc in idxs_bd:

            INDX      = anc + n_anc * rd 
            det_name = "D"+str(INDX)

            if rd==rds:
                pi[tuple([det_name])] = 0
            else:
                pi[tuple([det_name])] = epsilons_bd_space[anc]

    
    DEM = stim.DetectorErrorModel()

    for key, val in pij_space.items():
        targets = [stim.target_relative_detector_id(int(det[1:])) for det in key]
        targets.append(stim.target_logical_observable_id(0))
        DEM.append("error", val, targets)

    for key, val in pij_time.items():
        targets = [stim.target_relative_detector_id(int(det[1:])) for det in key]
        DEM.append("error", val, targets)

    for key, val in pi.items():
        t = int(key[0][1:])
        DEM.append("error", val, [stim.target_relative_detector_id(t), stim.target_logical_observable_id(0)])


    return DEM, pij_space, pij_time,pi


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


def estimate_and_decode_parallel_fixed_seed(basis: str, initial_state: str, num_shots: int, sigma=1e-3):
    '''
    Decode the estimated DEM of repetition code, as well as the uniform weight DEM with pymatching and plot the LERs.
    The noise model is phenomenological amplitude damping. Syndrome extraction is repeated for d rounds per distance.

    Input:
        basis: 'Z' or 'X'
        initial_state: '1' or '0' for 'Z' basis, and '+' for 'X' basis
        num_shots: number of Monte Carlo shots 
        sigma: standard deviation (if we want the \epsilon values per qubit to differ)

    '''

    e_s = [0.08, 0.1, 0.12  ] #amplitude damping strengths
    ds  = [3,5]              #distances

    # Prepare all (d,e) pairs for the parallel simulation
    de_pairs = [(d, e) for d in ds for e in e_s]

    def get_det_events_and_obs_flips(d,epsilon,num_shots,sigma):
        
        nQ          = d + d-1
        rds         = d
        P_cache     = build_projectors(nQ)
        
        Had  = 1/np.sqrt(2)*np.array([[1,1],[1,-1]],dtype=complex)
        Had = [Had]*nQ 
        Had = kron_n(Had)        

        #Store the CNOTs gates based on the basis

        CNOTs = np.eye(2**nQ, dtype=complex)

        if basis=='X': #control are the ancilla qubits
            for k in range(d-1): #number of ancilla        
                control = k + d  #d=3 -> 0,1,2 are data qubits, for k=0, 3 is the first ancilla
                CNOTs = CNOT_ij(nQ,control=control,target=k)  @ CNOTs
                CNOTs = CNOT_ij(nQ,control=control,target=k+1) @ CNOTs


        elif basis=='Z': #control are the data qubits
            for k in range(d-1):
                target = k + d 
                CNOTs = CNOT_ij(nQ,control=k,target=target) @ CNOTs
                CNOTs = CNOT_ij(nQ,control=k+1,target=target) @ CNOTs


        all_det_events = []
        all_obs_flips  = []

        #We will fix the Kraus channel per shot but can vary per qubit.
        kraus_list = []

        epsilons=fixed_seed_epsilons(nQ,epsilon,sigma)
        
        for i in range(nQ):

            kraus_ops = amplitude_damping_noise_channel(epsilons[i])
            
            kraus_list.append(kraus_ops)
        
        for _ in range(num_shots):
            
            det_events, obs_flips = sample_detection_and_logical_flip(d, rds, basis, initial_state, kraus_list,P_cache,Had,CNOTs) # det_events, obs_flips = sample_detection_and_logical_flip(kraus_list)
            all_det_events.append(det_events)
            all_obs_flips.append(obs_flips)

        all_det_events = np.array(all_det_events, dtype=np.uint8)
        all_obs_flips  = np.array(all_obs_flips, dtype=np.uint8)

        DEM_ideal,_,_,_=get_DEM_ideal(d,rds,epsilons) #ground-truth DEM (each rate is epsilon_j):

        # Decode
        # H = repetition_code_pcm(d)
        # logicals = np.ones(d,dtype=np.uint8)
        # pred = Matching.from_check_matrix(H,repetitions=rds+1,faults_matrix=logicals).decode_batch(all_det_events) #rds+1 because we also measure final data qubits
        pred = Matching.from_detector_error_model(DEM_ideal).decode_batch(all_det_events)
        pred = np.array(pred, dtype=np.uint8)

        if initial_state=='1':
            all_obs_flips = all_obs_flips^1

        ler_stim = np.sum(pred.flatten() != all_obs_flips)/num_shots

        #-- Estimate ------------
        DEM, pij_space, pij_time,pi = estimate_DEM(d,rds,all_det_events)

        # print("DEM:",DEM)        
        # print("(d,e):",(d,epsilon))
        # print("bulk space:",pij_space)
        # print("bulk time:",pij_time)
        # print("bd:",pi)
        # print("------")
 
        # Decode
        pred = Matching.from_detector_error_model(DEM).decode_batch(all_det_events)
        pred = np.array(pred, dtype=np.uint8)
        ler_est = np.sum(pred.flatten() != all_obs_flips)/num_shots

        return d, epsilon, ler_stim, ler_est

    results = Parallel(n_jobs=-1,verbose=10)(delayed(get_det_events_and_obs_flips)(d, e, num_shots,sigma) for d, e in de_pairs)    

    _, _, all_ler_stim, all_ler_est = zip(*results)
    all_ler_stim = np.array(all_ler_stim)
    all_ler_est  = np.array(all_ler_est)

    all_ler_stim = all_ler_stim.reshape(len(ds), len(e_s))
    all_ler_est  = all_ler_est.reshape(len(ds), len(e_s))


    #-------------  Now plot the results -----------------
    
    fig, ax = plt.subplots()

    colors = ["tab:blue","tab:orange","tab:green","tab:red"]
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
    plt.legend(frameon=False,fontsize=13)
    plt.grid(True)


    plt.show()

    return 


basis         = 'Z'
initial_state = '1'

estimate_and_decode_parallel_fixed_seed(basis,initial_state,num_shots=50_000,sigma=0*2e-3)



