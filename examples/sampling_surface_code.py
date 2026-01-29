import sys

#Provide the path to the build folder
path += "/Estimating_coherent_errors_with_DEMs/build"
sys.path.insert(0, path)  # path to the .so file

import sample_surface_code
import numpy as np
import json


def sample(theta_data,theta_anc, theta_G, rds,ITERS):
    '''
    Sample detection events and observable flips for d=3 rotated surface code.
    The error model is e^{-i\theta *Z} on data + ancilla and e^{i\theta ZZ} after each CNOT.

    Input:
        theta_data: the rotation angle for data qubits
        theta_anc: the rotation angle for ancilla qubits
        theta_G: the gate rotation angle
        rds: the # of QEC rounds
        ITERS: number of times to repeat the Monte Carlo sim
    '''    
    
    q_readout  = 0
    
    det_events,obs_flips = sample_surface_code.sample_detection_events( rds,  ITERS,  theta_data,  theta_anc,  theta_G,  q_readout)
            
    return det_events,obs_flips

ITERS      = 10_000
theta_data = 0.07*np.pi
theta_anc  = 0.07*np.pi 
theta_G    = 0.07*np.pi 
rds        = 2


det_events,obs_flips = sample(theta_data,theta_anc, theta_G, rds,ITERS)

#Write the det events

path = f"det_events_rds_{rds}_shots_{ITERS}"
path += f"_theta_d_{theta_data/np.pi}_pi"
path += f"_theta_a_{theta_anc/np.pi}_pi"
path += f"_theta_G_{theta_G/np.pi}_pi.txt"


with open(path, "w") as file:
    file.write(str(det_events))

#Write the obs flips
path = f"obs_flips_rds_{rds}_shots_{ITERS}"
path += f"_theta_d_{theta_data/np.pi}_pi"
path += f"_theta_a_{theta_anc/np.pi}_pi"
path += f"_theta_G_{theta_G/np.pi}_pi.txt"

with open(path, "w") as file:
    file.write(str(obs_flips))



print("DONE!")


#To load data: 
#with open(path, "r") as f:
#   test = json.load(f)
