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
from python_src.estimation_functions import *
import stim 
from pymatching import Matching


matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif" 


def process_single_run_uniform(d,theta,prob_depol1,prob_depol2,ITERS):
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

    
    det_events,obs_flips = sample_repetition_code.sample_circ_level_mixed_coh_stoc_rep_code(d, \
            rds,  ITERS,  theta_data,  theta_anc,  theta_G,  q_readout,\
            prob_depol1,  prob_depol2,   Reset_ancilla)
    
    return det_events,obs_flips


def estimate_stoch_DEM(detection_events,d,rds):

    
    vi_mean = get_vi_mean(detection_events)
    n_stabs = d-1
    rds     = rds+1
    
    pijkl = {}
    pijk  = {}

    #Time edges
    p_time = {}

    for rd1 in range(rds-1):

        rd2 = rd1+1

        for anc2 in range(n_stabs):

            anc1 = anc2 
            indx1 = anc1 + n_stabs * rd1
            indx2 = anc2 + n_stabs * rd2

            p = get_pij_corr(detection_events,vi_mean,indx1,indx2,pijk,pijkl)

            p_time[("D"+str(indx1),"D"+str(indx2))] = p 



    #Space bulk edges 
    p_space = {}
    for rd1 in range(rds):
        rd2=rd1 
        for anc1 in range(n_stabs-1):
            
            anc2 = anc1+1 
            indx1 = anc1 + n_stabs * rd1
            indx2 = anc2 + n_stabs * rd2

            p = get_pij_corr(detection_events,vi_mean,indx1,indx2,pijk,pijkl)

            p_space[("D"+str(indx1),"D"+str(indx2))] = p


    #Diag bulk edges:  @ (t,anc) - (t+1,anc-1)
    p_diag = {}
    for rd1 in range(rds-1):
        rd2 = rd1+1 
        for anc1 in range(1,n_stabs):

            anc2 = anc1-1
            indx1 = anc1 + n_stabs * rd1
            indx2 = anc2 + n_stabs * rd2

            p = get_pij_corr(detection_events,vi_mean,indx1,indx2,pijk,pijkl)
            p_diag[("D"+str(indx1),"D"+str(indx2))] = p

    
    #Bd edges 
    p_bd = {}
    anc1 = 0
    for rd1 in range(rds):
        
        indx1  = anc1 + n_stabs * rd1
        DENOM  = 1
        vi     = vi_mean[indx1]

        #Get nearest space edge 
        indx2 = (anc1+1) + n_stabs * rd1
        DENOM *= 1-2*p_space[("D"+str(indx1),"D"+str(indx2))]

        #Get nearest time edges
        try:
            indx2  = anc1 + n_stabs * (rd1+1)
            DENOM *= 1-2*p_time[("D"+str(indx1),"D"+str(indx2))]
        except KeyError:
            1
        
        try:
            indx2  = anc1 + n_stabs * (rd1-1)
            DENOM *= 1-2*p_time[("D"+str(indx2),"D"+str(indx1))]
        except KeyError:
            1

        #Get nearest diagonal in the past 
        try:
            indx2 = anc1+1 + n_stabs * (rd1-1)
            DENOM *= 1-2*p_diag[("D"+str(indx2),"D"+str(indx1))]
        except KeyError:
            1
        p_bd[("D"+str(indx1))] = 1/2 + (vi-1/2)/DENOM 



    anc1 = n_stabs-1 
    for rd1 in range(rds):
        
        indx1  = anc1 + n_stabs * rd1
        DENOM  = 1
        vi     = vi_mean[indx1]

        #Get nearest space edge 
        indx2 = (anc1-1) + n_stabs * rd1
        DENOM *= 1-2*p_space[("D"+str(indx2),"D"+str(indx1))]
        #Get nearest time edges
        try:
            indx2  = anc1 + n_stabs * (rd1+1)
            DENOM *= 1-2*p_time[("D"+str(indx1),"D"+str(indx2))]
        except KeyError:
            1
        
        try:
            indx2  = anc1 + n_stabs * (rd1-1)
            DENOM *= 1-2*p_time[("D"+str(indx2),"D"+str(indx1))]
        except KeyError:
            1

        #Get nearest diagonal in the future (if it exists) 
        try:
            indx2 = anc1-1 + n_stabs * (rd1+1)
            DENOM *= 1-2*p_diag[("D"+str(indx1),"D"+str(indx2))]
            
        except KeyError:
            1
        
        p_bd[("D"+str(indx1))] = 1/2 + (vi-1/2)/DENOM 



    for key,val in p_bd.items():
        if p_bd[key]<0:
            p_bd[key]=0
        
    for key,val in p_diag.items():
        if p_diag[key]<0:
            p_diag[key]=0

    for key,val in p_space.items():
        if p_space[key]<0:
            p_space[key]=0

    for key,val in p_time.items():
        if p_time[key]<0:
            p_time[key]=0

    DEM = stim.DetectorErrorModel()

    

    for key,val in p_time.items():
            targets = [int(d[1:]) for d in key]
            targets = [stim.target_relative_detector_id(t) for t in targets] 
            
            if val>0:
                DEM.append("error",val,targets)


    for key,val in p_space.items():
            targets = [int(d[1:]) for d in key]
            targets = [stim.target_relative_detector_id(t) for t in targets] 
            targets.append(stim.target_logical_observable_id(0))
            if val>0:
                DEM.append("error",val,targets)

    for key,val in p_diag.items():
            targets = [int(d[1:]) for d in key]
            targets = [stim.target_relative_detector_id(t) for t in targets] 
            targets.append(stim.target_logical_observable_id(0))
            if val>0:
                DEM.append("error",val,targets)

    for key,val in p_bd.items():
            
            targets = [int(key[1:])]
            targets = [stim.target_relative_detector_id(t) for t in targets] 
            targets.append(stim.target_logical_observable_id(0))
            if val>0:
                DEM.append("error",val,targets)


    all_dets = n_stabs * rds 
    

    for k in range(all_dets):
        DEM.append("DETECTOR",[],targets=[stim.target_relative_detector_id(k)])

    return DEM 

def verify_stoch_model(ITERS):

    theta_data = 0.0
    theta_anc  = 0.0
    theta_G    = 0.0 
    q_readout  = 0.0
    Reset_ancilla = 1

    ds = [3,5,7]
    ps = np.linspace(5e-2,1.7e-1,5)

    for d in ds:

        rds=d

        LER_per_d = []
        err_bar_per_d = []

        for p in ps:
            prob_depol1 = p 
            prob_depol2 = 0*p

            detection_events,obs_flips = sample_repetition_code.sample_circ_level_mixed_coh_stoc_rep_code(d, \
                    rds,  ITERS,  theta_data,  theta_anc,  theta_G,  q_readout,\
                    prob_depol1,  prob_depol2,   Reset_ancilla)

            detection_events = np.array(detection_events)
            
            DEM = estimate_stoch_DEM(detection_events,d,rds)

            matching = Matching.from_detector_error_model(DEM)
            predictions = matching.decode_batch(detection_events)

            predictions = np.squeeze(predictions)
            predictions = np.array(predictions)
            obs_flips   = np.array(obs_flips)

            num_errors = np.sum(predictions!=obs_flips)/ITERS 

            
            LER_per_d.append(num_errors)
            err_bar_per_d.append(np.sqrt(num_errors*(1-num_errors)/ITERS))



        plt.errorbar(ps,LER_per_d,yerr=err_bar_per_d,label=f'd={d}')

    plt.ylabel('P_L')
    plt.xlabel("Physical error rate")
    plt.yscale('log')
    plt.show()

    return 



def verify_stoch_model_parallel(ITERS, n_jobs=-1):

    theta_data = 0.0
    theta_anc  = 0.0
    theta_G    = 0.0 
    q_readout  = 0.0
    Reset_ancilla = 1

    ds = [3, 5, 7]
    ps = np.linspace(5e-2, 1.7e-1, 5)

    def compute_LER_for_pair(d, p):
        rds = d
        prob_depol1 = p
        prob_depol2 = 0*p

        detection_events, obs_flips = sample_repetition_code.sample_circ_level_mixed_coh_stoc_rep_code(
            d, rds, ITERS, theta_data, theta_anc, theta_G, q_readout,
            prob_depol1, prob_depol2, Reset_ancilla
        )

        detection_events = np.array(detection_events)

        DEM = estimate_stoch_DEM(detection_events, d, rds)

        matching = Matching.from_detector_error_model(DEM)
        predictions = matching.decode_batch(detection_events)

        predictions = np.squeeze(np.array(predictions))
        obs_flips = np.array(obs_flips)

        num_errors = np.sum(predictions != obs_flips) / ITERS
        err_bar = np.sqrt(num_errors * (1 - num_errors) / ITERS)

        return d, p, num_errors, err_bar

    
    pairs = [(d, p) for d in ds for p in ps]

    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_LER_for_pair)(d, p) for d, p in pairs
    )

    
    for d in ds:
        filtered = [(p, ler, err) for dd, p, ler, err in results if dd == d]
        filtered.sort(key=lambda x: x[0])  
        ps_sorted, LER_per_d, err_bar_per_d = zip(*filtered)
        plt.errorbar(ps_sorted, LER_per_d, yerr=err_bar_per_d, label=f'd={d}')

    plt.ylabel('P_L')
    plt.xlabel("Physical error rate")
    plt.yscale('log')
    plt.legend()
    plt.show()

    return 

verify_stoch_model_parallel(ITERS=2*10**4)

