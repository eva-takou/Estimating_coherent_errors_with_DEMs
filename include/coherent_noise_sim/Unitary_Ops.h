#pragma once
#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <unordered_set>
#include <cassert>
#include <utility>
#include <cmath>


#include "PrecisionOfTypes.h"
#include <xsimd/xsimd.hpp>


using namespace Eigen;
using std::vector;


std::vector<std::pair<size_t, size_t>> precompute_CNOT_swaps(int control, const std::vector<int>& targets, int nQ);

VectorXc precompute_Rz_phase_mask(int nQ, const std::vector<int>& qubits, const std::vector<Real>& thetas);

VectorXc compute_ZZ_phase_mask(int nQ, int q1, int q2, Real theta);

std::vector<std::pair<size_t, size_t>> precompute_Hadamard_flip_masks(const std::vector<int>& qubits, int nQ);


inline void apply_CNOTs_from_precomputed_swaps(const std::vector<std::pair<size_t, size_t>>& swaps, VectorXc& psi){

    Complex* data = psi.data();

    for (const auto& [i, j] : swaps) {
        // std::swap(psi[i], psi[j]);
        std::swap(data[i], data[j]);
    }
}


inline void apply_Hadamard_on_all_qubits(VectorXc& psi) {
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


//TODO: CLEANUP THE CODE

inline void apply_Rz_on_qubits_inplace_old(VectorXc& psi, const std::vector<int>& qubits, Real theta) {

    // using cplx = std::complex<double>;
    const Eigen::Index dim = psi.size();
    const int n = static_cast<int>(std::log2(dim));
    const Complex I(Real(0), Real(1));
    const Complex e0 = std::exp(-I * theta);
    const Complex e1 = std::exp(I * theta);

    const int nq = static_cast<int>(qubits.size());
    const int n_patterns = 1 << nq;


    // Precompute all possible phase combinations for bit patterns on target qubits
    std::vector<Complex> phase_table(n_patterns);
    for (int b = 0; b < n_patterns; ++b) {
        Complex phase = 1.0;
        for (int i = 0; i < nq; ++i) {
            int bit = (b >> (nq - 1 - i)) & 1;
            phase *= (bit == 0) ? e0 : e1;
        }
        phase_table[b] = phase;
    }

    // Create a mask vector indicating bit positions of target qubits
    std::vector<int> bit_shifts(nq);
    for (int i = 0; i < nq; ++i)
        bit_shifts[i] = n - 1 - qubits[i];

    // Apply precomputed phase using bit pattern lookup
    for (Eigen::Index i = 0; i < dim; ++i) {
        int pattern = 0;
        for (int j = 0; j < nq; ++j) {
            pattern |= ((i >> bit_shifts[j]) & 1) << (nq - 1 - j);
        }
        psi[i] *= phase_table[pattern];
    }
}

inline void apply_Rz_on_qubits_inplace_old_V2(VectorXc& psi, const std::vector<int>& qubits, Real theta) {
    
    // using cplx = std::complex<double>;

    const Eigen::Index dim = psi.size();
    const int nQ = static_cast<int>(std::log2(dim));
    const Complex I(Real(0), Real(1));
    const Complex e0 = std::exp(-I * theta);
    const Complex e1 = std::exp(I * theta);

    const int nq = static_cast<int>(qubits.size());
    std::vector<int> bit_shifts(nq);
    for (int i = 0; i < nq; ++i)
        bit_shifts[i] = nQ - 1 - qubits[i];

    ArrayXc phase_mask(dim);

    for (Eigen::Index i = 0; i < dim; ++i) {
        Complex phase = 1.0;
        for (int j = 0; j < nq; ++j) {
            bool bit = (i >> bit_shifts[j]) & 1;
            phase *= (bit ? e1 : e0);
        }
        phase_mask[i] = phase;
    }

    Eigen::Map<ArrayXc>(psi.data(), psi.size()) *= phase_mask;
}


inline void apply_Rz_on_qubits_inplace(VectorXc& psi, const std::vector<int>& qubits, Real theta) {
    const Eigen::Index dim = psi.size();
    const int nQ = static_cast<int>(std::log2(dim));
    const Complex I(Real(0), Real(1));
    const Complex e0 = std::exp(-I * theta);
    const Complex e1 = std::exp(I * theta);

    // Precompute bitmask (bitset) for each qubit
    uint64_t mask = 0;
    for (int q : qubits) {
        mask |= (1ULL << (nQ - 1 - q));  
    }

    // Apply phase directly in-place
    for (Eigen::Index i = 0; i < dim; ++i) {
        int parity = __builtin_popcountll(i & mask);  // count how many e1s to apply
        Complex phase = std::pow(e1 / e0, parity) * std::pow(e0, static_cast<int>(qubits.size()));
        psi[i] *= phase;
    }
}




// inline void apply_precomputed_Rz_mask(VectorXcd& psi, const ArrayXcd& phase_mask) {
//     // psi.array() *= phase_mask.array();

//     Eigen::Map<Eigen::ArrayXcd>(psi.data(), psi.size()) *= phase_mask; //This will avoid creating a temporary array for both sides. However requires defining phase_mask as ArrayXcd //TODO if 
    
// }

//This is element-wise multiplication
// inline void apply_precomputed_ZZ_mask(VectorXcd& psi, const ArrayXcd& phase_mask) {
//     // psi.array() *= phase_mask.array();
//     // Eigen::Map<Eigen::ArrayXcd>(psi.data(), psi.size()) *= phase_mask.array();
//     Eigen::Map<Eigen::ArrayXcd>(psi.data(), psi.size()) *= phase_mask;
// }


//TODO: FIX THIS.

inline void apply_precomputed_ZZ_mask(VectorXc& psi, const ArrayXc& phase_mask) {
    const size_t N = psi.size();
    for (size_t i = 0; i < N; ++i) {
        psi[i] *= phase_mask[i];
    }
}

inline void apply_precomputed_Rz_mask(VectorXc& psi, const ArrayXc& phase_mask) {
    
    auto* p = psi.data();
    auto* m = phase_mask.data();
    const size_t N = psi.size();
    for (size_t i = 0; i < N; ++i) {
        const Real re = p[i].real();
        const Real im = p[i].imag();
        const Real mr = m[i].real();
        const Real mi = m[i].imag();
        p[i].real(re * mr - im * mi);
        p[i].imag(re * mi + im * mr);
    }    
}




// inline void apply_precomputed_ZZ_mask(VectorXcd& psi, const ArrayXcd& phase_mask) {
    
//     auto* p = psi.data();
//     auto* m = phase_mask.data();
//     const size_t N = psi.size();
//     for (size_t i = 0; i < N; ++i) {
//         p[i] *= m[i];
//     }
// }


//For X gates
inline void apply_X_on_qubits(VectorXc& psi, const std::vector<uint8_t>& outcome_bitstring, int offset, const Eigen::Index dim, int nQ) {

    // Compute the combined X flip mask
    Eigen::Index flip_mask = 0;

    for (int i = 0; i < static_cast<int>(outcome_bitstring.size()); ++i) {
        if (outcome_bitstring[i]) {
            int q = offset + i;
            flip_mask |= (1ULL << (nQ - 1 - q));
        }
    }

    if (flip_mask == 0) return; // nothing to do

    // Use bitmask symmetry to avoid redundant swaps
    for (Eigen::Index j = 0; j < dim; ++j) {
        Eigen::Index j_flip = j ^ flip_mask;
        if (j < j_flip) {
            std::swap(psi[j], psi[j_flip]);
        }
    }
}






// void apply_Hadamard_on_qubit_in_place(VectorXc& psi, int q);

// VectorXc apply_Hadamard_on_qubit(VectorXc& psi, int q);

// VectorXc apply_Hadamard_on_qubits(VectorXc& psi, std::vector<int> inds);

