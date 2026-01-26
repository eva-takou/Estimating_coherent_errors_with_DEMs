#pragma once
#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <unordered_set>
#include <cassert>
#include <utility>
#include <cmath>
#include "PrecisionOfTypes.h"
#include "constants.h"

using namespace Eigen;
using std::vector;



std::vector<std::pair<size_t, size_t>> precompute_CNOT_swaps(int control, const std::vector<int>& targets, int nQ);

VectorXc precompute_Rz_phase_mask(int nQ, const std::vector<int>& qubits, const std::vector<Real>& thetas);

VectorXc compute_ZZ_phase_mask(int nQ, int q1, int q2, Real theta);

std::vector<std::pair<size_t, size_t>> precompute_Hadamard_flip_masks(const std::vector<int>& qubits, int nQ);




template <typename Derived>
inline void inplace_hadamard_on_rows(Eigen::MatrixBase<Derived>& M) {

    const int N = M.rows();
    const int cols = M.cols();

    for (int c = 0; c < cols; ++c) {
        auto* col_ptr = &M(0, c);  

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

//USED FOR REPETITION CODE
inline void apply_hadamards_on_ancilla_qubits(VectorXc& psi, int d) {
    /*
    Apply Hadamard gates on the ancilla qubits only. Note the vector is treated as
    psi_{data} \otimes psi_{anc} (order of states, doesnt mean the state has to be product)
    
    Input:
    psi: the state vector before the Hadamards
    d: distance of the repetition code

    Output (in-place):
    psi: the state vector after the Hadamards
    */

    const int data_dim    = 1 << d;
    const int ancilla_dim = 1 << (d - 1);

    // Map state as 2D matrix: [ancilla_index][data_index]
    Eigen::Map<MatrixXc> psi_matrix(psi.data(), ancilla_dim, data_dim); //interpret ancilla as rows so that we apply the transform only there
    
    inplace_hadamard_on_rows(psi_matrix);
    
}
//Under development.
inline void apply_hadamards_on_ancilla_qubits_surf_code(VectorXc& psi, int d, int n_anc) {
    /*
    Need to assume a fictitious thing where data + anc_X are treated in columns
    Last ancilla are Z (to revert hadamards)
    
    Input:
    psi: the state vector before the Hadamards
    d: distance of the surface_code

    Output (in-place):
    psi: the state vector after the Hadamards
    */

    const int data_and_X_anc_dim    = 1 << ((d*d)+n_anc/2); //Assume
    const int Z_anc_dim = 1 << n_anc/2;

    // Map state as 2D matrix: [ancilla_index][data_index]
    Eigen::Map<MatrixXc> psi_matrix(psi.data(), Z_anc_dim, data_and_X_anc_dim); //interpret ancilla as rows so that we apply the transform only there
    
    inplace_hadamard_on_rows(psi_matrix);
    
}





inline void apply_CNOTs_from_precomputed_swaps(const std::vector<std::pair<size_t, size_t>>& swaps, VectorXc& psi){
    /*
    Apply CNOTs by swapping the corresponding indices. Note that SWAP order is constrained based on the CNOT order.

    Input:
    swaps: vector of pairs of indices to swap
    psi: the state vector before the CNOTs

    Output (in-place):
    psi: the state vector after the CNOTs

    */
    
    Complex* data = psi.data();

    for (const auto& [i, j] : swaps) {
        std::swap(data[i], data[j]);
    }
}


inline void apply_Hadamard_on_all_qubits(VectorXc& psi) {
    /*
    Apply Hadamard gate on each qubit of the state using butterfly networks (Walsh-Hadamard transform, cost O(Nlog(N)) operations).
    
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



inline void apply_precomputed_ZZ_mask(VectorXc& psi, const ArrayXc& phase_mask) {
    // const size_t N = psi.size();
    // for (size_t i = 0; i < N; ++i) {
    //     psi[i] *= phase_mask[i];
    // }
    psi.array() *= phase_mask; //.array()
}

inline void apply_precomputed_Rz_mask(VectorXc& psi, const ArrayXc& phase_mask) {
    
    psi.array() *= phase_mask;
}




inline void apply_X_on_qubits(VectorXc& psi, const std::vector<uint8_t>& outcome_bitstring, int offset, const Eigen::Index dim, int nQ) {
    /*
    Apply X gate on qubits to return from |1> -> |0> state, after their measurement outcome (conditional reset).

    Input:
    psi: the full state vector
    outcome_bitstring: the outcomes of the qubits that were measured
    offset: from which qubit we start (e.g., if we measure the ancilla we have an offset=d since the first d qubits are the data qubits)
    dim: dimension 2^n
    nQ: total # of qubits
    */
    
    Eigen::Index flip_mask = 0;

    for (int i = 0; i < static_cast<int>(outcome_bitstring.size()); ++i) {
        if (outcome_bitstring[i]) {
            int q = offset + i;
            flip_mask |= (1ULL << (nQ - 1 - q)); //MSB ordering
        }
    }

    if (flip_mask == 0) return; 

    
    for (Eigen::Index j = 0; j < dim; ++j) {
        Eigen::Index j_flip = j ^ flip_mask;
        if (j < j_flip) {
            std::swap(psi[j], psi[j_flip]);
        }
    }
}


inline void apply_Rx_on_qubits_inplace(VectorXc& psi, const std::vector<int>& qubits, Real theta) {
    //Note the convention: this does e^{-i\theta * X}
    const Eigen::Index dim = psi.size();
    const int nQ = static_cast<int>(std::log2(dim));
    const Complex I(0,1);
    const Complex cos_t = std::cos(theta);
    const Complex sin_t = std::sin(theta);

    for (int q : qubits) {
        uint64_t mask = 1ULL << (nQ - 1 - q);  //MSB order

        for (Eigen::Index i = 0; i < dim; ++i) {
            if ((i & mask) == 0) {  //apply only for bit q=1
                Eigen::Index i0 = i;
                Eigen::Index i1 = i | mask;
                Complex a0 = psi[i0];
                Complex a1 = psi[i1];
                psi[i0] = cos_t * a0 - I * sin_t * a1;
                psi[i1] = -I * sin_t * a0 + cos_t * a1;
            }
        }
    }
}

inline void apply_Ry_on_qubits_inplace(VectorXc& psi, const std::vector<int>& qubits, Real theta) {
    //Note the convention: this does e^{-i\theta * Y}
    const Eigen::Index dim = psi.size();
    const int nQ = static_cast<int>(std::log2(dim));
    const Complex I(0,1);
    const Complex cos_t = std::cos(theta);
    const Complex sin_t = std::sin(theta);

    for (int q : qubits) {
        uint64_t mask = 1ULL << (nQ - 1 - q);  //MSB order

        for (Eigen::Index i = 0; i < dim; ++i) {
            if ((i & mask) == 0) {  //apply only for bit q=1
                Eigen::Index i0 = i;
                Eigen::Index i1 = i | mask;
                Complex a0 = psi[i0];
                Complex a1 = psi[i1];
                psi[i0] = cos_t * a0 -  sin_t * a1;
                psi[i1] = sin_t * a0 + cos_t * a1;
            }
        }
    }
}

inline void apply_Rz_on_qubits_inplace(VectorXc& psi, const std::vector<int>& qubits, Real theta) {
    /*
    Apply e^{-i\theta Z} operation on a set of qubits.

    Inputs:
    psi: input state vector
    qubits: vector of indiecs on which to apply the operation
    theta: angle \theta of the operation

    Output (in-place):
    the updated state vector

    */
    const Eigen::Index dim = psi.size();
    const int nQ = static_cast<int>(std::log2(dim));
    const Complex I(Real(0), Real(1));
    const Complex e0 = std::exp(-I * theta);
    const Complex e1 = std::exp(I * theta);

    // Precompute bitmask (bitset) for each qubit
    uint64_t mask = 0;
    for (int q : qubits) {
        mask |= (1ULL << (nQ - 1 - q));   //MSB ordering
    }

    // Apply phase directly in-place
    for (Eigen::Index i = 0; i < dim; ++i) {
        int parity = __builtin_popcountll(i & mask);  // count how many e1s to apply
        Complex phase = std::pow(e1 / e0, parity) * std::pow(e0, static_cast<int>(qubits.size()));
        psi[i] *= phase;
    }
}


void apply_Hadamard_on_qubit_in_place(VectorXc& psi, int q);


inline void apply_Hadamard_on_qubits(VectorXc& psi, const std::vector<int> & qubits){

    for (int q: qubits){
        apply_Hadamard_on_qubit_in_place(psi,  q);

    }
    
}






// VectorXc apply_Hadamard_on_qubit(VectorXc& psi, int q);

// VectorXc apply_Hadamard_on_qubits(VectorXc& psi, std::vector<int> inds);


