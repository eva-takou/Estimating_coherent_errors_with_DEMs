#pragma once
#include "PrecisionOfTypes.h"
#include <vector>
#include <tuple>




Real get_logical_infidelity(int d, int rds, int ITERS, Real theta_data,  Real q_readout, bool Reset_ancilla);                                                                    


Real get_LER_from_estimated_DEM(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real theta_G, Real q_readout, 
                                bool Reset_ancilla,  bool include_higher_order, bool print_higher_order);

Real get_LER_from_uniform_DEM_circuit_level(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real theta_G, Real q_readout,  bool Reset_ancilla);                                


Real get_LER_from_uniform_DEM_phenom_level(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real q_readout,  bool Reset_ancilla);
