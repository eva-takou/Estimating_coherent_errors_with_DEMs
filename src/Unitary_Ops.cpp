#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <unordered_set>
#include <cassert>
#include <utility>
#include <cmath>

#include "PrecisionOfTypes.h"


using namespace Eigen;
using std::vector;


std::vector<std::pair<size_t, size_t>> precompute_CNOT_swaps(int control, const std::vector<int>& targets, int nQ) {
    /*
    Compute the indices that need to be swapped to implement CNOTs on a state vector.
    Input:
    control: index of control qubit
    targets: vector of indices of target qubits
    nQ: total number of qubits
    
    Output:
    vector of pairs of indices that need to be swapped.
    */

    const size_t dim = 1ULL << nQ;
    const size_t ctrl_mask = 1ULL << (nQ - 1 - control);

    size_t total_target_mask = 0;
    for (int t : targets) {
        total_target_mask |= (1ULL << (nQ - 1 - t));
    }

    std::vector<std::pair<size_t, size_t>> swaps;
    swaps.reserve(dim / 2);  

    for (size_t i = 0; i < dim; ++i) {
        if ((i & ctrl_mask) == 0) continue;

        size_t j = i ^ total_target_mask;
        if (i < j) {
            swaps.emplace_back(i, j);
        }
    }

    return swaps;
}



ArrayXc precompute_Rz_phase_mask(int nQ, const std::vector<int>& qubits, const std::vector<Real>& thetas) {
    /*
    Compute the Rz phase mask, where Rz^{(j)} = e^{-i\theta_j Z} applied on each qubit.
    Input:
    nQ: total # of qubits
    qubits: vector of indices for the qubits where we apply the operations
    thetas: vector of angles \theta_j
    
    Output:
    the phase mask to be applied on the state vector.
    */

    const int dim = 1 << nQ;
    const int nq = static_cast<int>(qubits.size());
    assert(thetas.size() == nq && "thetas must match number of qubits");

    ArrayXc phase_mask(dim);
    const Complex I(Real(0), Real(1));

    
    std::vector<int> bit_shifts(nq);
    std::vector<uint64_t> bitmasks(nq);
    std::vector<Complex> phase0(nq), phase1(nq);
    for (int j = 0; j < nq; ++j) {
        bit_shifts[j] = nQ - 1 - qubits[j];  
        bitmasks[j] = 1ULL << bit_shifts[j];
        const Real theta = thetas[j];
        phase0[j] = std::exp(-I * theta);
        phase1[j] = std::exp(I * theta);
    }

    // Compute phase mask 
    for (int i = 0; i < dim; ++i) {
        Complex phase = Complex(1.0, 0.0);
        for (int j = 0; j < nq; ++j) {
            phase *= (i & bitmasks[j]) ? phase1[j] : phase0[j];
        }
        phase_mask[i] = phase;
    }

    return phase_mask;
}


ArrayXc compute_ZZ_phase_mask(int nQ, int q1, int q2, Real theta) {
    /*
    Compute phase mask for e^{i\theta ZZ} operation.
    Input:
    nQ: total # of qubits
    q1: index of the first qubit
    q2: index of the second qubit
    theta: angle for the operation
    
    Output:
    the phase mask to be applied on the state vector.
    */

    const int dim = 1 << nQ;
    const Complex I(Real(0), Real(1));

    int shift1 = nQ - 1 - q1;
    int shift2 = nQ - 1 - q2;

    ArrayXc phase_mask(dim);
    for (int i = 0; i < dim; ++i) {
        bool bit1 = (i >> shift1) & 1;
        bool bit2 = (i >> shift2) & 1;
        int z1 = bit1 ? +1 : -1;
        int z2 = bit2 ? +1 : -1;
        Real phase_angle = theta * z1 * z2;
        phase_mask[i] = std::exp(I * phase_angle);
    }

    return phase_mask;
}


void apply_Hadamard_on_all_qubits(VectorXc& psi) {
    /*
    Apply Hadamard gate on each qubit of the state using butterfly networks.
    Input:
    psi: State vector
    Output:
    psi (modified in place)
    */

    const size_t N = psi.size();
    Complex* psi_data = psi.data(); 

    for (size_t len = 1; len < N; len <<= 1) {
        for (size_t i = 0; i < N; i += 2 * len) {
            for (size_t j = 0; j < len; ++j) {
                
                Complex& a = psi_data[i + j]; 
                Complex& b = psi_data[i + j + len]; 
                
                Complex a_val = a;
                Complex b_val = b;
                a = (a_val + b_val) * SQRT2_INV;
                b = (a_val - b_val) * SQRT2_INV;                
            }
        }
    }
}


template <typename Derived>
inline void inplace_hadamard_on_rows(Eigen::MatrixBase<Derived>& M) {
    const int N = M.rows();
    const int cols = M.cols();

    for (int c = 0; c < cols; ++c) {
        auto* col_ptr = &M(0, c);  // pointer to start of column c

        for (int len = 1; len < N; len <<= 1) {
            for (int i = 0; i < N; i += 2 * len) {
                auto* top_ptr = col_ptr + i;
                auto* bot_ptr = top_ptr + len;

                for (int j = 0; j < len; ++j) {
                    auto tmp_top = top_ptr[j];
                    auto tmp_bot = bot_ptr[j];
             
                    top_ptr[j] = (tmp_top + tmp_bot) * SQRT2_INV;
                    bot_ptr[j] = (tmp_top - tmp_bot) * SQRT2_INV;
                }


            }
        }
    }
}



inline void apply_fast_hadamards_on_ancilla_qubits(VectorXc& psi, int d) {

    const int data_dim    = 1 << d;
    const int ancilla_dim = 1 << (d - 1);

    // Map state as 2D matrix: [ancilla_index][data_index]
    Eigen::Map<MatrixXc> psi_matrix(psi.data(), ancilla_dim, data_dim); //interpret ancilla as rows so that we apply the transform only there
    
    inplace_hadamard_on_rows(psi_matrix);
    
}









//------------------------- Unused ---------------------------------------------

//For Hadamard operations
// std::vector<std::pair<size_t, size_t>> precompute_Hadamard_flip_masks(const std::vector<int>& qubits, int nQ) {
//     const size_t dim = 1ULL << nQ;
//     std::vector<std::pair<size_t, size_t>> ij_pairs;
    
//     ij_pairs.reserve(qubits.size() * (dim / 2));  // conservative overestimate

//     for (int q : qubits) {
//         const size_t flip_bit = 1ULL << (nQ - 1 - q);
//         for (size_t i = 0; i < dim; ++i) {
//             if ((i & flip_bit) == 0) {
//                 ij_pairs.emplace_back(i, i ^ flip_bit);
//             }
//         }
//     }

//     return ij_pairs;
// }



// void apply_Hadamard_on_qubit_in_place(VectorXc& psi, int q) {
    
//     const Eigen::Index dim = psi.size();
//     const int n = static_cast<int>(std::log2(dim));
//     const Eigen::Index flip_bit = 1LL << (n - 1 - q);
    
   
//     Real inv_sqrt = static_cast<Real>(1.0 / std::sqrt(2.0));

//     for (Eigen::Index i = 0; i < dim; ++i) {
//         // Only apply once per pair (to avoid double updates)
//         if ((i & flip_bit) == 0) {
//             Eigen::Index j = i ^ flip_bit;

//             Complex temp_i = psi[i];
//             Complex temp_j = psi[j];
//             psi[i] = (temp_i + temp_j) * inv_sqrt;  // 1/sqrt(2)
//             psi[j] = (temp_i - temp_j) * inv_sqrt;
//         }
//     }

// }

// VectorXc apply_Hadamard_on_qubit(VectorXc& psi, int q) {
    
//     const Eigen::Index dim = psi.size();
//     const int n = static_cast<int>(std::log2(dim));
//     const Eigen::Index flip_bit = 1LL << (n - 1 - q);
    
//     VectorXc psi_f = psi;
//     Real inv_sqrt = static_cast<Real>(1.0 / std::sqrt(2.0));

//     for (Eigen::Index i = 0; i < dim; ++i) {
//         // Only apply once per pair (to avoid double updates)
//         if ((i & flip_bit) == 0) {
//             Eigen::Index j = i ^ flip_bit;

//             Complex temp_i = psi_f[i];
//             Complex temp_j = psi_f[j];
//             psi_f[i] = (temp_i + temp_j) * inv_sqrt;  // 1/sqrt(2)
//             psi_f[j] = (temp_i - temp_j) * inv_sqrt;
//         }
//     }

//     return psi_f;
// }

// VectorXc apply_Hadamard_on_qubits(VectorXc& psi, std::vector<int> inds) {
//     VectorXc psi_f=psi;
//     for (int i =0 ; i<inds.size(); ++i){

//         psi_f = apply_Hadamard_on_qubit(psi_f,inds[i]);
//     }

//     return psi_f;
// }






