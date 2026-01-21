#pragma once
#include <vector>
#include <tuple>
#include "utils_sc.h"




std::tuple<ProbDictXZ,ProbDictXZ,ProbDictXZ,ProbDictXZ> estimate_edges_surf_code(const std::vector<std::vector<uint8_t>>& batch, int d, int n_anc, int rds, 
                                                                                 bool include_higher_order, bool print_higher_order);