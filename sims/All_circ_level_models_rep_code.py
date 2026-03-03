#Compare the performance of coherent, stochastic and mixed coherent-stochastic noise models
#on the circuit level.
#The DEM is estimated in the same way (w/o making assumptions about the noise) for each input model.

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


def is_subset(lower_order_key,higher_order_key):

    ints_lower_order  = [int(d[1:]) for d in lower_order_key]
    ints_higher_order = [int(d[1:]) for d in higher_order_key]

    for indx in ints_lower_order:
        if indx not in ints_higher_order:
            return False 

    return True

def redefine_lowest_order_prob(lower_order_key, lower_order_prob,higher_order_dict):

    updated_prob = lower_order_prob

    for higher_order_key,val in higher_order_dict.items():

        if is_subset(lower_order_key,higher_order_key):

            updated_prob = (updated_prob - val) / (1-2*val)

            if updated_prob<0:
                updated_prob = 0

            return updated_prob

    return updated_prob

def multiply_exclusion_factor(lower_order_key,term_to_correct,pijkl):
    '''
    To correct the 3rd order prob with the 4th order prob
    '''
    updated_term = term_to_correct 

    for higher_order_key,val in pijkl.items():
        if is_subset(lower_order_key,higher_order_key):
            updated_term *= 1/ (1-2*val)


    return updated_term

def get_all_four_point_probs(detection_events,vi_mean,n_stabs,rds):

    pijkl = {}

    for rd in range(rds-1):

        rd1 = rd
        rd2 = rd1 
        rd3 = rd1+1
        rd4 = rd3 

        for anc in range(n_stabs-1):

            anc1 = anc 
            anc2 = anc1+1 
            anc3 = anc1 
            anc4 = anc2 

            indx1 = anc1 + n_stabs * rd1 
            indx2 = anc2 + n_stabs * rd2
            indx3 = anc3 + n_stabs * rd3 
            indx4 = anc4 + n_stabs * rd4 

            p = get_4pnt_prob(detection_events,vi_mean,indx1,indx2,indx3,indx4)

            pijkl[("D"+str(indx1),"D"+str(indx2),"D"+str(indx3),"D"+str(indx4))] = p 



    return pijkl

def get_all_three_point_probs(detection_events,vi_mean,n_stabs,rds,pijkl):


    pijk = {}

    for rd in range(rds-1):

        rd1 = rd 
        rd2 = rd 
        rd3 = rd+1 

        for anc in range(n_stabs-1):

            anc1 = anc 
            anc2 = anc1+1 
            anc3 = anc 

            indx1 = anc1 + n_stabs * rd1 
            indx2 = anc2 + n_stabs * rd2 
            indx3 = anc3 + n_stabs * rd3 

            p = get_3pnt_prob(detection_events,vi_mean,indx1,indx2,indx3,pijkl)

            pijk[("D"+str(indx1),"D"+str(indx2),"D"+str(indx3))] = p
            

    return pijk

def estimate_time_edges(detection_events,vi_mean,n_stabs,rds,pijkl,pijk):

    p_time = {}

    for rd1 in range(rds-1):

        rd2 = rd1+1

        for anc2 in range(n_stabs):

            anc1 = anc2 
            indx1 = anc1 + n_stabs * rd1
            indx2 = anc2 + n_stabs * rd2

            p = get_pij_corr(detection_events,vi_mean,indx1,indx2,pijk={},pijkl={})

            lower_order_key = ("D"+str(indx1),"D"+str(indx2))

            p = redefine_lowest_order_prob(lower_order_key,p,pijk)
            p = redefine_lowest_order_prob(lower_order_key,p,pijkl)

            p_time[("D"+str(indx1),"D"+str(indx2))] = p 


    return p_time

def estimate_space_edges(detection_events,vi_mean,n_stabs,rds,pijkl,pijk):

    #Space bulk edges 
    p_space = {}
    for rd1 in range(rds):
        rd2=rd1 
        for anc1 in range(n_stabs-1):
            
            anc2 = anc1+1 
            indx1 = anc1 + n_stabs * rd1
            indx2 = anc2 + n_stabs * rd2

            p = get_pij_corr(detection_events,vi_mean,indx1,indx2,pijk={},pijkl={})

            lower_order_key = ("D"+str(indx1),"D"+str(indx2))

            p = redefine_lowest_order_prob(lower_order_key,p,pijk)
            p = redefine_lowest_order_prob(lower_order_key,p,pijkl)


            p_space[("D"+str(indx1),"D"+str(indx2))] = p


    return p_space

def estimate_diag_edges(detection_events,vi_mean,n_stabs,rds,pijkl,pijk):

    #Diag bulk edges:  @ (t,anc) - (t+1,anc-1)
    p_diag = {}
    for rd1 in range(rds-1):
        rd2 = rd1+1 
        for anc1 in range(1,n_stabs):

            anc2 = anc1-1
            indx1 = anc1 + n_stabs * rd1
            indx2 = anc2 + n_stabs * rd2

            p = get_pij_corr(detection_events,vi_mean,indx1,indx2,pijk={},pijkl={})

            lower_order_key = ("D"+str(indx1),"D"+str(indx2))

            p = redefine_lowest_order_prob(lower_order_key,p,pijk)
            p = redefine_lowest_order_prob(lower_order_key,p,pijkl)


            p_diag[("D"+str(indx1),"D"+str(indx2))] = p

    return p_diag

def estimate_bd_edges(detection_events,vi_mean,n_stabs,rds,p_space,p_time,p_diag,pijkl,pijk):

    #BD edges as defined below do not exclude higher orders besides the p_{ij} order...
    
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


    return p_bd

def estimate_DEM(detection_events,d,rds,include_higher_order):

    
    vi_mean = get_vi_mean(detection_events)
    n_stabs = d-1
    rds     = rds+1

    if include_higher_order==True:

        pijkl = get_all_four_point_probs(detection_events,vi_mean,n_stabs,rds)
        pijk  = get_all_three_point_probs(detection_events,vi_mean,n_stabs,rds,pijkl)

        print("4 pnt events")
        for key,val in pijkl.items():
            print("key:",key,":",val)
        print("3 pnt events")
        for key,val in pijk.items():
            print("key:",key,":",val)            
    elif include_higher_order==False:
        pijkl = {}
        pijk  = {}

    else:
        raise Exception("can only be True or False")

    #Time edges
    p_time = estimate_time_edges(detection_events,vi_mean,n_stabs,rds,pijkl,pijk)

    #Space bulk edges
    p_space = estimate_space_edges(detection_events,vi_mean,n_stabs,rds,pijkl,pijk)

    #Diag space-time edges
    p_diag = estimate_diag_edges(detection_events,vi_mean,n_stabs,rds,pijkl,pijk)

    #Boundary edges (we should also exclude 3rd and 4th order probably in these expressions)
    p_bd = estimate_bd_edges(detection_events,vi_mean,n_stabs,rds,p_space,p_time,p_diag,pijkl,pijk)

    print("time:")
    for key,val in p_time.items():
        print("key:",key,":",val)
    
    print("space:")
    for key,val in p_space.items():
        print("key:",key,":",val)

    print("diag:")
    for key,val in p_diag.items():
        print("key:",key,":",val)
    
    print("pd:")
    for key,val in p_bd.items():
        print("key:",key,":",val)


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


def run_one_case(d, theta, ITERS, include_higher_order, q_readout, Reset_ancilla,prob_depol1,prob_depol2):

    rds = d

    theta_data = theta
    theta_anc  = theta
    theta_G    = -theta

    detection_events, obs_flips = (
        sample_repetition_code.sample_circ_level_mixed_coh_stoc_rep_code(
            d, rds, ITERS,
            theta_data, theta_anc, theta_G,
            q_readout,
            prob_depol1, prob_depol2,
            Reset_ancilla
        )
    )

    detection_events = np.array(detection_events)
    obs_flips = np.array(obs_flips)

    DEM = estimate_DEM(detection_events, d, rds, include_higher_order)

    matching = Matching.from_detector_error_model(DEM)
    predictions = matching.decode_batch(detection_events)

    predictions = np.squeeze(np.array(predictions))

    num_errors = np.sum(predictions != obs_flips) / ITERS
    err_bar = np.sqrt(num_errors * (1 - num_errors) / ITERS)

    return d, theta, num_errors, err_bar

def get_LER_circuit_level_models(ITERS, include_higher_order, n_jobs=-1):
    
    q_readout     = 0.0
    Reset_ancilla = 1

    ds = [3,5,7,9]

    thetas2 = np.array([0.03*np.pi,0.04*np.pi,0.05*np.pi,0.06*np.pi])    
    thetas1 = thetas2

    ps1 = [100*np.sin(th)**2 for th in thetas1]
    ps2 = [100*np.sin(th)**2 for th in thetas2]

    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    
    #------------------ Mixed stochastic - coherent ---------------------
    tasks = [
        (d, theta)
        for d in ds
        for theta in thetas2
    ]


    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_one_case)(
            d, theta, ITERS, include_higher_order,
             q_readout, Reset_ancilla,np.sin(theta)**2,np.sin(theta)**2
        )
        for d, theta in tasks
    )


    # Organize results
    results_dict = {d: {"LER": [], "ERR": []} for d in ds}

    for d, theta, num_errors, err_bar in results:
        results_dict[d]["LER"].append(num_errors)
        results_dict[d]["ERR"].append(err_bar)

    

    cnt=0
    for d in ds:
        plt.errorbar(
            ps2,
            results_dict[d]["LER"],
            yerr=results_dict[d]["ERR"],
            label=f'd={d}',marker='o',color=colors[cnt]
        )
        cnt+=1

    #-------------Now compare vs the purely stochastic
    tasks = [
        (d, theta)
        for d in ds
        for theta in thetas1
    ]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_one_case)(
            d, 0*theta, ITERS, include_higher_order,
             q_readout, Reset_ancilla,np.sin(theta)**2,np.sin(theta)**2
        )
        for d, theta in tasks
    )        

    results_dict = {d: {"LER": [], "ERR": []} for d in ds}

    for d, theta, num_errors, err_bar in results:
        results_dict[d]["LER"].append(num_errors)
        results_dict[d]["ERR"].append(err_bar)
    
    cnt=0
    for d in ds:
        plt.errorbar(
            ps1,
            results_dict[d]["LER"],
            yerr=results_dict[d]["ERR"],
            marker='s',color=colors[cnt],linestyle='-.'
        )
        cnt+=1

    #-------------Compare with the fully coherent model
    tasks = [
        (d, theta)
        for d in ds
        for theta in thetas2
    ]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_one_case)(
            d, theta, ITERS, include_higher_order,
             q_readout, Reset_ancilla,0*np.sin(theta)**2,0*np.sin(theta)**2
        )
        for d, theta in tasks
    )        

    results_dict = {d: {"LER": [], "ERR": []} for d in ds}

    for d, theta, num_errors, err_bar in results:
        results_dict[d]["LER"].append(num_errors)
        results_dict[d]["ERR"].append(err_bar)
    
    cnt=0
    for d in ds:
        plt.errorbar(
            ps2,
            results_dict[d]["LER"],
            yerr=results_dict[d]["ERR"],
            marker='d',color=colors[cnt],linestyle='--'
        )
        cnt+=1



    plt.ylabel('$P_L$')
    plt.xlabel("Physical error rate (%)")
    plt.yscale('log')
    plt.legend(fontsize=13,frameon=False)
    plt.show()

    return results_dict

print("---------------------")

include_higher_order = True
get_LER_circuit_level_models(ITERS=10**6,include_higher_order=include_higher_order)
