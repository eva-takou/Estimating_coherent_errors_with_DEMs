#pragma once

#include <vector>
#include <tuple>

#include "PrecisionOfTypes.h"


std::tuple<std::vector<std::vector<int>>, 
           std::vector<std::vector<int>>, 
           std::vector<Real>,std::vector<Real>,
           std::vector<Real>,std::vector<Real>> sample_outcomes_MC_surface_code(int d, int rds, int Nsamples, int Nsamples_data, int nsims,
                                                                             Real theta_data, Real theta_anc, Real theta_G, Real q_readout,
                                                                             int Reset_ancilla, 
                                                                             int include_stab_reconstruction);


Real get_LER_from_uniform_DEM_code_capacity_level(int d, int rds, int ITERS, Real theta_data, Real q_readout, Real pz,  bool Reset_ancilla);                                                                             