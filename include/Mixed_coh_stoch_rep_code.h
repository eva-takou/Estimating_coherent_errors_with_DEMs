#pragma once
#include "PrecisionOfTypes.h"
#include <vector>
#include <tuple>



std::tuple<std::vector<std::vector<uint8_t>>,std::vector<uint8_t>> sample_circ_level_mixed_coh_stoc_rep_code(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real theta_G, Real q_readout, Real prob_depol1, Real prob_depol2,  bool Reset_ancilla);
