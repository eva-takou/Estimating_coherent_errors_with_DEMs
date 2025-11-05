#pragma once
#include <vector>
#include <tuple>
#include "PrecisionOfTypes.h"


std::tuple<std::vector<Real>,std::vector<Real>,std::vector<Real>> estimate_edges_rep_code(const std::vector<std::vector<uint8_t>>& batch, int d, int n_stabs, int rds,
                                                                                          uint8_t include_higher_order, uint8_t print_higher_order);