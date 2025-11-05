#pragma once 
#include <Eigen/Dense>
#include "PrecisionOfTypes.h"

using namespace Eigen;

VectorXc Ket0(int nQ);
VectorXc Ket1(int nQ);
VectorXc plus_state(int d);
VectorXc minus_state(int d);