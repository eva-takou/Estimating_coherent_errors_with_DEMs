#Simulate mixed stochastc-coherent noise on d=3 rotated surface code in code-capacity setup


import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import scipy.linalg
import matplotlib
import random 
import sys
from pathlib import Path
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))
from scipy import linalg 

from python_src.estimation_functions import *

matplotlib.rcParams.update({'font.size': 24})
plt.rcParams["font.family"] = "Microsoft Sans Serif" 



def kron_n(operators: list):
    '''Calculate A_1 \otimes A_2 ... \otimes A_n kronecker product'''
    out = operators[0]
    for op in operators[1:]:
        out = np.kron(out, op)
    return out

def make_stabs():
    '''Create the stabilizers of the d=3 rotated surface code.
    Qubit numbering is columnwise. '''
    X = np.array([[0,1],[1,0]],dtype=complex)
    Z = np.array([[1,0],[0,-1]],dtype=complex)
    I = np.eye(2,dtype=complex)

    opers = [X,I,I,X,I,I,I,I,I]
    SX_0 = kron_n(opers) #X0*X3

    opers = [I,X,X,I,X,X,I,I,I] #X1*X2*X4*X5
    SX_1 = kron_n(opers)

    opers = [I,I,I,X,X,I,X,X,I] #X3*X4*X6*X7
    SX_2 = kron_n(opers)
    
    opers = [I,I,I,I,I,X,I,I,X] #X5*X8
    SX_3 = kron_n(opers)
    
    opers = [I,Z,Z,I,I,I,I,I,I] #Z1*Z2
    SZ_0 = kron_n(opers)

    opers = [Z,Z,I,Z,Z,I,I,I,I] #Z0*Z1*Z3*Z4 
    SZ_1 = kron_n(opers)
    
    opers = [I,I,I,I,Z,Z,I,Z,Z] #Z4*Z5*Z7*Z8 
    SZ_2 = kron_n(opers)
    
    opers = [I,I,I,I,I,I,Z,Z,I] #Z6*Z7
    SZ_3 = kron_n(opers)

    S = [SX_0,SX_1,SX_2,SX_3,SZ_0,SZ_1,SZ_2,SZ_3]

    return S

def prepare_logical_0_state():
    '''Prepare onto the |0>_L state of the d=3 rotated surface code by measuring
    the +1 e-val of all stabilizers'''

    d    = 3
    I = np.eye(2,dtype=complex)
    S = make_stabs()
    Id_all = [I]*(d**2)
    Id_all = kron_n(Id_all)
    
    psi0 = np.zeros((2**(d*d),1),dtype=complex)
    psi0[0,0]=1

    #Now apply the projectors 
    for G in S:

        Proj = (Id_all + G)/2
        psi0 = Proj@psi0

    norm_ = np.linalg.norm(psi0)
    psi0 = psi0/ norm_
    
    return psi0


def apply_stochastic_Z_errors(p,psi,opers_for_Z_stoch):
    '''Apply stochastic Z error on each qubit with probability p.
    
    p: probability for phase flip
    psi: state vector
    opers_for_Z_stoch: list of operators I....Z_k...I
    '''
    
    d  = 3
    nQ = d**2

    for k in range(nQ):
        r = random.random()

        if r<=p:

            K   = opers_for_Z_stoch[k]
            psi = K@psi
    
    return psi 

def apply_stochastic_X_errors(p,psi,opers_for_X_stoch):
    '''Apply stochastic X error on each qubit with probability p.
    
    p: probability for phase flip
    psi: state vector
    opers_for_X_stoch: list of operators I....X_k...I
    '''

    d  = 3
    nQ = d**2
    
    for k in range(nQ):
        r = random.random()

        if r<=p:

            K   = opers_for_X_stoch[k]
            psi = K@ psi
    

    return psi 

def apply_stochastic_Y_errors(p,psi,opers_for_Y_stoch):
    '''Apply stochastic Y error on each qubit with probability p.
    
    p: probability for phase flip
    psi: state vector
    opers_for_Y_stoch: list of operators I....Y_k...I
    '''    
    d  = 3
    nQ = d**2

    for k in range(nQ):
        r = random.random()

        if r<=p:

            K   = opers_for_Y_stoch[k]
            psi = K@ psi

    return psi     


def apply_coherent_X_errors(theta,psi):
    '''Apply coherent X error e^{-itheta X} on each qubit
    
    theta: error angle
    psi: state vector
    '''    

    d=3
    nQ = d*d 
    X = np.array([[0,1],[1,0]],dtype=complex)
    UX = linalg.expm(-1j*theta*X)

    Us = [UX]*(nQ)
    Us = kron_n(Us) 

    return Us @ psi 

def apply_coherent_Y_errors(theta,psi):
    '''Apply coherent Y error e^{-itheta Y} on each qubit
    
    theta: error angle
    psi: state vector
    '''        
    d=3
    nQ = d*d 
    Y = np.array([[0,-1j],[1j,0]],dtype=complex)
    UY = linalg.expm(-1j*theta*Y)

    Us = [UY]*(nQ)
    Us = kron_n(Us) 

    return Us @ psi 

def apply_coherent_Z_errors(theta,psi):
    '''Apply coherent Z error e^{-itheta Z} on each qubit
    
    theta: error angle
    psi: state vector
    '''       
    d=3
    nQ = d*d 
    Z = np.array([[1,0],[0,-1]],dtype=complex)
    UZ = linalg.expm(-1j*theta*Z)

    Us = [UZ]*(nQ)
    Us = kron_n(Us) 

    return Us @ psi 


def measure_probs(psi,Pminus,Pplus):
    '''Measure each stabilizer and collapse the state. 
    Output:
    detection outcomes 0/1 of stabilizer measurements

    psi: state vector
    Pminus: list of (1+S_j)/2 projectors for each S_j stabilizer
    Pplus: list of (1-S_j)/2 projectors for each S_j stabilizer
    '''  

    outcomes = []
    for k in range(len(Pminus)):
        
        p_minus1 = psi.conj().T @  Pminus[k] @ psi 
        p_minus1 = np.real(p_minus1[0,0])
        

        if np.random.rand() <= p_minus1:
            outcome=1

            psi = Pminus[k] @ psi 

        else:
            outcome=0
            psi = Pplus[k] @ psi 

        psi = psi/np.linalg.norm(psi)

        outcomes.append(outcome)

    
    return outcomes 


def run_single_repeat(ITERS, psi, p, opers_for_Z_stoch, opers_for_X_stoch, opers_for_Y_stoch, Pminus, Pplus):
    '''
    Run Monte Carlo simulation for a number of shots equal to ITERS

    ITERS: number of Monte Carlo shots
    psi: state vector
    p: probability of each stochastic error
    opers_for_Z_stoch: list of operators I...Z_k...I
    opers_for_X_stoch: list of operators I...X_k...I
    opers_for_Y_stoch: list of operators I...Y_k...I

    Output:
    all_outcomes: ITERS x num_stabs array of detection outcomes
    vi_mean: average # of times each detector fires
    '''
    all_outcomes = []

    for _ in range(ITERS):
        PSI = psi.copy()
        PSI = apply_stochastic_Z_errors(p, PSI, opers_for_Z_stoch)
        PSI = apply_stochastic_X_errors(p, PSI, opers_for_X_stoch)
        # PSI = apply_stochastic_Y_errors(p, PSI, opers_for_Y_stoch)
        outcomes = measure_probs(PSI, Pminus, Pplus)
        all_outcomes.append(outcomes)

    all_outcomes = np.array(all_outcomes)
    vi_mean = np.mean(all_outcomes, axis=0)

    return all_outcomes, vi_mean


def sample_Monte_Carlo_parallel(ITERS,REPEAT,p,theta):
    '''
    Run the Monte Carlo simulation and repeat for a number of times REPEAT
    to collect error bars

    ITERS: Monte Carlo shots per REPEAT iteration
    p: probability of stochastic error
    theta: error angle
    '''

    d  = 3
    nQ = d**2
    
    I = np.eye(2,dtype=complex)
    X = np.array([[0,1],[1,0]],dtype=complex)
    Y = np.array([[0,-1j],[1j,0]],dtype=complex)
    Z = np.array([[1,0],[0,-1]],dtype=complex)
    Id_all = np.eye(2**(nQ),dtype=complex)


    psi0  = prepare_logical_0_state()
    Stabs = make_stabs()
    
    Pminus   = [(Id_all - S)/2 for S in Stabs]
    Pplus    = [(Id_all + S)/2 for S in Stabs]

    opers_for_X_stoch = [] #Kronecker product of I.. X_k ... I
    opers_for_Z_stoch = [] #Kronecker product of I.. Z_k ... I
    opers_for_Y_stoch = [] #Kronecker product of I.. Y_k ... I

    for k in range(nQ):
        opers    = [I]*nQ 
        opers[k] = X
        opers_for_X_stoch.append(kron_n(opers))

    for k in range(nQ):
        opers    = [I]*nQ 
        opers[k] = Y
        opers_for_Y_stoch.append(kron_n(opers))        

    for k in range(nQ):
        opers    = [I]*nQ 
        opers[k] = Z
        opers_for_Z_stoch.append(kron_n(opers))


    all_outcomes = []

    #------------- Mixed Coherent-stochastic  ------------------
    #Coherent error needs to be applied only once 
    #Chose which coherent error to apply
    psi      = apply_coherent_Z_errors(theta,psi0)
    psi      = apply_coherent_X_errors(theta,psi)
    # psi = apply_coherent_Y_errors(theta,psi0)
 
    pij_bulk_X_mean = {}
    pij_bulk_Z_mean = {}
    pij_bd_Z_mean = {}
    pij_bd_X_mean = {}
    pijk_mean = {}
    pijkl_mean={}

    #If we want to apply single-qubit depolarizing, then we can apply independently
    #each X,Z,Y stochastic errors with the effective probability:   
    #p_eff = 1/2 - 1/2*np.sqrt(1-4/3*p)

    results = Parallel(n_jobs=-1,verbose=10)( delayed(run_single_repeat)(ITERS, psi, p, opers_for_Z_stoch, opers_for_X_stoch, opers_for_Y_stoch,Pminus, Pplus)
    for _ in range(REPEAT) )


    Hyperedge_3 = [(0,2,5),(2,6,7),(1,4,5),(1,3,6),]  #3 pnt hyperedges
    Hyperedge_4 = [(1,2,5,6)] #4 pnt hyperedge
    
    for all_outcomes, vi_mean in results:

        all_outcomes = np.array(all_outcomes)
        print("vi_mean:",vi_mean)

        pijkl={}
        
        for k in range(len(Hyperedge_4)):
            inds = Hyperedge_4[k]
            val = get_4pnt_prob(all_outcomes,vi_mean,*inds)
            if val<0:
                val=0
            key = ["D"+str(l) for l in inds]
            pijkl[tuple(key)] = val
            try: 
                pijkl_mean[tuple(key)]  = pijkl_mean[tuple(key)]  + [val]
            except KeyError:
                pijkl_mean[tuple(key)]  = [val]
        
        pijk={}
        
        for k in range(len(Hyperedge_3)):    
            inds = Hyperedge_3[k]
            val= get_3pnt_prob(all_outcomes,vi_mean,*inds,pijk)
            if val<0:
                val=0
            key = ["D"+str(l) for l in inds]
            pijk[tuple(key)] = val

            try:
                pijk_mean[tuple(key)]  = pijk_mean[tuple(key)]  + [val]
            except KeyError:
                pijk_mean[tuple(key)]  = [val]


        print("4pnt rate:",pijkl)
        print("3pnt rates:",pijk)

        #Estimate edges
        bulk_edge_pairs = [(0,2),(1,2),(1,3),  (4,5), (5,6),(6,7)] #first 3 are bulk X type edges, last 3 are Z bulk type edges
        #0,1,2,3 are X
        #4,5,6,7 are Z

        pij_bulk_X = {}
        for k in range(3):
            indx0,indx1 = bulk_edge_pairs[k]
            vi    = vi_mean[indx0]
            vj    = vi_mean[indx1]
        
            vij   = np.sum(all_outcomes[:,indx0] & all_outcomes[:,indx1])/ITERS
            numer = vij-vi*vj
            denom = 1-2*(vi+vj)+4*vij

            pij_bulk_X[("D"+str(indx0),"D"+str(indx1))] =  1/2 - np.sqrt(1/4 - numer/denom)

            try:
                pij_bulk_X_mean[("D"+str(indx0),"D"+str(indx1))] = pij_bulk_X_mean[("D"+str(indx0),"D"+str(indx1))] + [1/2 - np.sqrt(1/4 - numer/denom)]
            except KeyError:
                pij_bulk_X_mean[("D"+str(indx0),"D"+str(indx1))] = [1/2 - np.sqrt(1/4 - numer/denom)]


        pij_bulk_Z = {}
        for k in range(3,6):
            indx0,indx1 = bulk_edge_pairs[k]
            vi    = vi_mean[indx0]
            vj    = vi_mean[indx1]
        
            vij   = np.sum(all_outcomes[:,indx0] & all_outcomes[:,indx1])/ITERS
            numer = vij-vi*vj
            denom = 1-2*(vi+vj)+4*vij

            pij_bulk_Z[("D"+str(indx0),"D"+str(indx1))] =  1/2 - np.sqrt(1/4 - numer/denom)        

            try:
                pij_bulk_Z_mean[("D"+str(indx0),"D"+str(indx1))] = pij_bulk_Z_mean[("D"+str(indx0),"D"+str(indx1))] + [1/2 - np.sqrt(1/4 - numer/denom)]
            except KeyError:
                pij_bulk_Z_mean[("D"+str(indx0),"D"+str(indx1))] = [1/2 - np.sqrt(1/4 - numer/denom)]
        
        pij_bd_X={}
        bd_nodes = [0,1,2,3] 
        for node in bd_nodes:
            
            numer = vi_mean[node] - 1/2

            DENOM = 1
            for key,val in pij_bulk_X.items(): 
                if "D"+str(node) in key:
                    DENOM *= 1-2*val

            pij_bd_X[("D"+str(node))] = 1/2 + numer/DENOM 

            try:
                pij_bd_X_mean[("D"+str(node))] = pij_bd_X_mean[("D"+str(node))]  + [1/2 + numer/DENOM]
            except KeyError:
                pij_bd_X_mean[("D"+str(node))] = [1/2 + numer/DENOM] 

        pij_bd_Z={}
        bd_nodes = [4,5,6,7] 
        for node in bd_nodes:
            
            numer = vi_mean[node] - 1/2

            DENOM = 1
            for key,val in pij_bulk_Z.items(): 
                if "D"+str(node) in key:
                    DENOM *= 1-2*val

            pij_bd_Z[("D"+str(node))] = 1/2 + numer/DENOM 

            try:
                pij_bd_Z_mean[("D"+str(node))] = pij_bd_Z_mean[("D"+str(node))]  + [1/2 + numer/DENOM]
            except KeyError:
                pij_bd_Z_mean[("D"+str(node))] = [1/2 + numer/DENOM ]


    std_X_bulk_mean = []
    std_Z_bulk_mean = []
    std_X_bd_mean = []
    std_Z_bd_mean = []

    std_pijkl = []
    std_pijk  = []


    for key,val in pij_bd_X_mean.items():
        pij_bd_X_mean[key] = np.mean(val)
        std_X_bd_mean.append(np.std(val))
    
    for key,val in pij_bd_Z_mean.items():
        pij_bd_Z_mean[key] = np.mean(val)
        std_Z_bd_mean.append(np.std(val))

    for key,val in pij_bulk_Z_mean.items():
        pij_bulk_Z_mean[key] = np.mean(val)
        std_Z_bulk_mean.append(np.std(val))

    for key,val in pij_bulk_X_mean.items():
        pij_bulk_X_mean[key] = np.mean(val)
        std_X_bulk_mean.append(np.std(val))

    for key,val in pijkl_mean.items():
        pijkl_mean[key] = np.mean(val)
        std_pijkl.append(np.std(val))

    for key,val in pijk_mean.items():
        pijk_mean[key] = np.mean(val)
        std_pijk.append(np.std(val))


    pX =   pij_bulk_X_mean | pij_bd_X_mean
    pZ =   pij_bulk_Z_mean | pij_bd_Z_mean 
    xvals = np.arange(len(pX))
    zvals = np.array(np.arange(len(pZ)))+max(xvals)+1

    fig, ax = plt.subplots()

    plt.bar(xvals,pX.values(),yerr=std_X_bulk_mean+std_X_bd_mean,label='X edges',color='tab:blue',edgecolor='black',capsize=6) 

    plt.bar(zvals,pZ.values(),yerr=std_Z_bulk_mean+std_Z_bd_mean,label='Z edges',color='tab:red',edgecolor='black',   capsize=6)

    xvals = max(zvals)+1 + np.array(np.arange(len(pijkl_mean)))
    plt.bar(xvals,pijkl_mean.values(),yerr=std_pijkl,label='4-point hyperedges',color='tab:purple',edgecolor='black',   capsize=6)

    xvals = max(xvals)+1 + np.array(np.arange(len(pijk_mean)))
    plt.bar(xvals,pijk_mean.values(),yerr=std_pijk,label='3-point hyperedges',color='tab:green',edgecolor='black',    capsize=6)    


    #This is to plot horizontal lines
    # plt.axhline(np.sin(theta)**2 + p,
    #         linestyle='-.',
    #         linewidth=1,
    #         color='k',
    #         alpha=0.6)        


    plt.xlabel('Edge index')
    plt.ylabel('Probability')

    #To show the legend
    # plt.legend(frameon=False,fontsize=13,
    #            loc="upper left",bbox_to_anchor=(0.001, 1.3),
    #            ncols=2) #loc='center left'
    
    ax.set_xticklabels([int(t) for t in ax.get_xticks()])
    
    #To save the figure    
    # fig.savefig(f"TEST_ITERS_{ITERS}_REPEAT_{REPEAT}_theta_{theta}_p_{p}.pdf",bbox_inches='tight')

    plt.show()


    return 



ITERS        = 100_000
REPEAT       = 1
p            = 0.05
theta        = 0.05*np.pi 

sample_Monte_Carlo_parallel(ITERS,REPEAT,p,theta)


