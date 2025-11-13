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



def process_single_run_code_capacity(d,theta,ITERS):
    '''
    Get the logical infidelity P_L = \sum_s P(s)sin^2(\theta_s), where P(s) is the probability to get the syndrome s, and
    \theta_s the logical rotation angle.
    This works only in code capacity, or when we have coherent data errors + stochastic readout errors.
    Due to the DEM structure under these models, we expect the same threshold as for stochastic noise.
    '''
    
    theta_data    = theta
    q_readout     = 0
    rds           = 1
    Reset_ancilla = 1
    
    
    LER = sample_repetition_code.get_logical_infidelity(d,  rds,  ITERS,  theta_data,   q_readout,  Reset_ancilla)
    
    return LER


def plot_logical_infidelity_code_capacity(ITERS):

    ds        = [3,5,7]

    thetas  = np.array([ 0.1*np.pi, 0.15*np.pi, 0.2*np.pi, 0.25*np.pi, 0.3*np.pi])  

    param_grid = [(d, theta) for d in ds for theta in thetas]


    flat_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_run_code_capacity)(d,theta,ITERS)
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


def process_single_run_code_capacity_and_readout_errors(d,theta,ITERS):
 
    
    theta_data    = theta
    q_readout     = np.sin(theta)**2
    rds           = d
    Reset_ancilla = 1
    
    
    LER = sample_repetition_code.get_logical_infidelity(d,  rds,  ITERS,  theta_data,   q_readout,  Reset_ancilla)
    
    return LER

def plot_logical_infidelity_cc_and_readout_errors(ITERS):

    ds        = [3,5,7]

    
    thetas  = np.array([ 0.08*np.pi, 0.09*np.pi,0.1*np.pi, 0.11*np.pi])  

    param_grid = [(d, theta) for d in ds for theta in thetas]


    flat_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_run_code_capacity_and_readout_errors)(d,theta,ITERS)
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


ITERS                = 10**5
# plot_logical_infidelity_code_capacity(ITERS) #for e^{-i\theta Z} errors on data qubits
plot_logical_infidelity_cc_and_readout_errors(ITERS)  #for e^{-i\theta Z} errors on data qubits + classical readout errors

