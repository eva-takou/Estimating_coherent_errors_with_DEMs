#pragma once
#include <vector>
#include <complex>
#include "PrecisionOfTypes.h"

std::vector<std::vector<int>> decode_batch_with_pymatching(
    const std::vector<std::vector<int>>& H,
    const std::vector<std::vector<uint8_t>>& batch,
    int repetitions);


std::vector<std::vector<int>> decode_with_pymatching_create_graph(const std::vector<std::vector<int>>& H, 
                                                                  const std::vector<Real>& space_prob, 
                                                                  const std::vector<Real>& time_prob,
                                                                  const std::vector<Real>& diag_prob,
                                                                  const std::vector<std::vector<uint8_t>>& batch,int rds, 
                                                                  int include_stab_reconstruction);
       

                                                             