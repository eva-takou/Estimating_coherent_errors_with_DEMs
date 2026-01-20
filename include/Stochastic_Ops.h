#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <unordered_set>
#include <cassert>
#include <utility>
#include <cmath>

#include "PrecisionOfTypes.h"
#include "constants.h"
#include "Unitary_Ops.h"

#include <random>

static std::mt19937 rng(std::random_device{}());

using namespace Eigen;
using std::vector;

inline void apply_stochastic_Z_on_qubits(VectorXc& psi, const std::vector<int>& qubits, const std::vector<Real>& prob_Z){
    
    std::vector<int> qubits_to_apply_Z;

    std::uniform_real_distribution<double> dist(0.0, 1.0);    
    for (int q = 0; q < qubits.size(); ++q) {
        double r = dist(rng);   
        if (r < prob_Z[q]) {
            qubits_to_apply_Z.push_back(qubits[q]);
        }
    }

    apply_Rz_on_qubits_inplace(psi,qubits_to_apply_Z, PI/2);


}


inline void apply_stochastic_X_on_qubits(VectorXc& psi, const std::vector<int>& qubits, const std::vector<Real>& prob_X, int nQ){
    
    std::vector<int> qubits_to_apply_X;

    std::uniform_real_distribution<double> dist(0.0, 1.0);    


    for (int q = 0; q < qubits.size(); ++q) {
        double r = dist(rng);   
        if (r < prob_X[q]) {
            qubits_to_apply_X.push_back(qubits[q]);
        }
    }


    Eigen::Index flip_mask = 0;

    for (int i = 0; i < qubits_to_apply_X.size(); ++i) {
        
        int q = qubits_to_apply_X[i];
        flip_mask |= (1ULL << (nQ - 1 - q)); //MSB ordering
        
    }

    if (flip_mask == 0) return; 

    
    for (Eigen::Index j = 0; j < dim; ++j) {
        Eigen::Index j_flip = j ^ flip_mask;
        if (j < j_flip) {
            std::swap(psi[j], psi[j_flip]);
        }
    }    


}




