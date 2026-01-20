
#pragma once
#include <vector>
#include <Eigen/Dense>

inline void apply_stochastic_Z_on_qubits(VectorXc& psi, const std::vector<int>& qubits, const std::vector<Real>& prob_Z);