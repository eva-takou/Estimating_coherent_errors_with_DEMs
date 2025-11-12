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
    Get the logical error rate if all coherent error angles are theta for data, and ancilla. The gate error
    angle should be sufficiently smaller for the estimation to work well. Here we want to see what happens if we ignore the hyperedges.
    The decoding graph is assumed to be the estimated one.
    '''
    rds        = d
    theta_data = theta
    theta_anc  = theta 
    Reset_ancilla = 1
    q_readout   = 0
    include_higher_order = 0
    print_higher_order = 0
    
    
    LER = sample_repetition_code.get_LER_from_estimated_DEM(d,rds,ITERS,theta_data,theta_anc,
                                                            theta_G,q_readout,Reset_ancilla,include_higher_order,print_higher_order)
    return LER



def plot_LER_phenom_level(ITERS, theta_G):

    ds      = [3,5,7]
    
    #This has to be adjusted for different values of \theta_G
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


theta_G              = 0.01*np.pi
ITERS                = 5*10**5
plot_LER_phenom_level(ITERS,theta_G)

