#pragma once
#include <Eigen/Core>
#include <complex>

using Real     = double; 
using Complex  = std::complex<Real>;


using VectorXc = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using ArrayXc = Eigen::Array<Complex, Eigen::Dynamic, 1>;
using ArrayXr  = Eigen::Array<Real, Eigen::Dynamic, 1>;
using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

//This is optional, in case we want to measure runtimes.
using Time          = double;
using Clock         = std::chrono::high_resolution_clock;
using Evaluate_Time = std::chrono::duration<Time>;



