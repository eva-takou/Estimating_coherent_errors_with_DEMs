#Logical error rate for fully coherent circuit-level model for a repetition code memory

import sys
from pathlib import Path
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

# path to compiled .so
sys.path.insert(0, str(proj_root / "build"))



import sample_repetition_code
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from joblib import Parallel, delayed




matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif" 


def process_single_run_uniform(d,theta,ITERS):
    '''
    Get the logical error rate if all coherent error angles are theta for data, ancilla and gate errors.
    The decoding graph is assumed to be uniform.
    '''
    rds        = d
    theta_data = theta
    theta_anc  = theta 
    theta_G    = -theta
    Reset_ancilla = 1
    q_readout   = 0

    
    LER = sample_repetition_code.get_LER_from_uniform_DEM_circuit_level(d,rds,ITERS,theta_data,theta_anc,
                                  theta_G,q_readout,Reset_ancilla)
    return LER


def process_single_run_estimated(d,theta,ITERS):
    '''
    Get the logical error rate if all coherent error angles are theta for data, ancilla and gate errors.
    The decoding graph is assumed to be the estimated one.
    '''
    rds        = d
    theta_data = theta
    theta_anc  = theta 
    theta_G    = -theta
    Reset_ancilla = 1
    q_readout   = 0
    include_higher_order = 1
    print_higher_order = 0
    
    
    LER = sample_repetition_code.get_LER_from_estimated_DEM(d,rds,ITERS,theta_data,theta_anc,
                                                            theta_G,q_readout,Reset_ancilla,include_higher_order,print_higher_order)
    return LER



def plot_LER_circ_level(ITERS, DEM_choice):

    
    if DEM_choice=="uniform":
        thetas = np.array([0.04*np.pi, 0.043*np.pi, 0.047*np.pi, 0.05*np.pi, 0.052*np.pi,0.054*np.pi  ]) 
        ds      = [3,5,7,9]
    elif DEM_choice=="estimated":
        thetas = np.array([0.043*np.pi, 0.047*np.pi, 0.05*np.pi, 0.052*np.pi, 0.054*np.pi  ]) 
        ds      = [3,5,7,9,]

    def iters_for_d(d):

        if d == 11:
            return 10**5
        
        return 10**6 #Do 10**6  
    
    if DEM_choice=="estimated":
        param_grid = [(d, theta, iters_for_d(d)) for d in ds for theta in thetas]
    elif DEM_choice=="uniform":
        param_grid = [(d, theta) for d in ds for theta in thetas]


    if DEM_choice=="estimated":

        flat_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_single_run_estimated)(d, theta, iters)
            for d, theta, iters in param_grid
        )    

    elif DEM_choice=="uniform":

        flat_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_single_run_uniform)(d, theta, ITERS)
            for d, theta in param_grid
        )    

    PL_per_d = []
    idx      = 0
    for _ in ds:
        PL_per_d.append(flat_results[idx:idx + len(thetas)])

        idx += len(thetas)


    sin_sq = [np.sin(th)**2*100 for th in thetas]

    fig, ax = plt.subplots()

    all_std = []
    all_LER = []
    for k in range(len(ds)):
        
        LER     = np.array(PL_per_d[k])

        if DEM_choice == "estimated":
            std_err = (LER*(1-LER)/iters_for_d(ds[k]))**0.5
            all_std.append(std_err)
        elif DEM_choice == "uniform":
            std_err = (LER*(1-LER)/ITERS)**0.5
            all_std.append(std_err)

        all_LER.append(LER)

        plt.errorbar(sin_sq, PL_per_d[k], yerr=std_err,marker='o' )

    plt.legend(["$d=3$","$d=5$", "$d=7$", "$d=9$", "$d=11$"],frameon=False)
    plt.yscale("log")
    plt.ylabel("$P_L$")
    plt.xlabel("Physical error rate (%)")


    if DEM_choice=="estimated":
        fig.savefig(f"circ_level_threshold_{DEM_choice}_iters_{iters_for_d(5)}_less_than_d_11_and_iters_{iters_for_d(11)}_for_d_11.pdf",bbox_inches='tight')
    else:
        fig.savefig(f"circ_level_threshold_{DEM_choice}_iters_{ITERS}.pdf",bbox_inches='tight')

    plt.show()

    if DEM_choice=="estimated":
        with open(f'LER_for_estimated_for_iters_d_less_11_{iters_for_d(5)}_and_d_11_{iters_for_d(11)}.txt', 'w') as file:
            file.write(str(LER))    
        with open(f'std_for_estimated_for_iters_d_less_11_{iters_for_d(5)}_and_d_11_{iters_for_d(11)}.txt', 'w') as file:
            file.write(str(all_std))                

    return 


DEM_choice           = "estimated"
ITERS                = 10**5
plot_LER_circ_level(ITERS,DEM_choice)

