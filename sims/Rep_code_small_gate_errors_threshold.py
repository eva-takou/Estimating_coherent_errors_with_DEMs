import sys

#Provide the path to the build folder
sys.path.insert(0, "/Users/evatakou/test_c/Estimating_coherent_errors_with_DEMs/build")  # path to the .so file

import sample_repetition_code
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif" 

def process_single_run_estimated(d,theta,ITERS,theta_G):
    '''
    Get LER for a noise model with e^{-i\theta Z} errors on data and ancilla qubits per QEC round,
    and small gate errors theta_G after every CNOT. Note internally the code does e^{i\theta_G ZZ}, so we need to also provide the sign.
    The decoding graph is assumed to be the estimated one.    

    Input:
    d: distance of repetition code (X-memory)
    theta: error angle for data and ancilla qubits (e^{-i\theta Z} errors)
    ITERS: the number of shots for estimation + decoding 
    theta_G: error angle for gate errors after perfect CNOTS (e^{i\theta_G ZZ} errors)

    Output:
    logical error rate
    '''
    rds        = d
    theta_data = theta
    theta_anc  = theta 
    Reset_ancilla = 1
    q_readout   = 0
    include_higher_order = 0 #Do not include higher order corrections
    print_higher_order = 0
    
    
    LER = sample_repetition_code.get_LER_from_estimated_DEM(d,rds,ITERS,theta_data,theta_anc,
                                                            theta_G,q_readout,Reset_ancilla,include_higher_order,print_higher_order)
    return LER



def plot_LER_small_gate_errors(ITERS, theta_G):

    ds      = [3,5,7]
    
    #Range of thetas for data+ancilla qubits needs to be adjusted.
    if theta_G==-0.01*np.pi:
        thetas  = np.array([0.075*np.pi, 0.08*np.pi, 0.085*np.pi,  0.09*np.pi, 0.095*np.pi, ])   #This is for theta=-0.01*pi
    elif theta_G==-0.018*np.pi:
        thetas = np.array([0.07*np.pi, 0.075*np.pi, 0.08*np.pi, 0.085*np.pi, 0.09*np.pi, ])   #This is for theta=-0.025*pi
    

    thetas  = np.array([ 0.07*np.pi, 0.08*np.pi, 0.09*np.pi, 0.1*np.pi ])  

    param_grid = [(d, theta) for d in ds for theta in thetas]


    flat_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_run_estimated)(d, theta, ITERS, theta_G)
        for d, theta in param_grid
    )    


    PL_per_d = []
    idx      = 0
    for _ in ds:
        PL_per_d.append(flat_results[idx:idx + len(thetas)])

        idx += len(thetas)


    sin_sq = [np.sin(th)**2*100 for th in thetas]

    fig, ax = plt.subplots()

    for k in range(len(ds)):
        
        LER     = np.array(PL_per_d[k])
        std_err = (LER*(1-LER)/ITERS)**0.5
        plt.errorbar(sin_sq, PL_per_d[k], yerr=std_err,marker='o' )

    plt.legend(["$d=3$","$d=5$", "$d=7$", "$d=9$", "$d=11$"],frameon=False)
    plt.yscale("log")
    plt.ylabel("$P_L$")
    plt.xlabel("Physical error rate (%)")

    plt.show()

    return 


theta_G              = -0.018*np.pi #Notice this implements e^{-i\theta_G ZZ} and we need to put this sign.
ITERS                = 5*10**4
plot_LER_small_gate_errors(ITERS,theta_G)

