#pragma once
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <random>
#include "pcg_random.hpp"
#include "PrecisionOfTypes.h"

std::random_device rd;  // Seed
pcg32 rand_gen;

inline void cumSum_from_state_vector(const VectorXc& psi, std::vector<Real>& cdf_buffer) {

    const int dim = psi.size();
    ArrayXr norms = psi.array().abs2();

    Real cumulative = 0.0;
    for (int i = 0; i < dim; ++i) {
        cumulative += norms[i];
        cdf_buffer[i] = cumulative;
    

}
}

std::vector<Real> cumSum(const std::vector<Real>& input) {
    
    std::vector<Real> cumsum(input.size()); // Output vector
    std::partial_sum(input.begin(), input.end(), cumsum.begin());

    return cumsum;
}

inline void cumSum_inplace(const std::vector<Real>& input, std::vector<Real>& output) noexcept {
    if (output.size() != input.size()) output.resize(input.size());
    std::partial_sum(input.begin(), input.end(), output.begin());
}


void print_vector(const std::vector<Real>& vec, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "  [" << i << "] = " << vec[i] << "\n";
    }
}


void print_vector_uint8(const std::vector<uint8_t>& vec, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "  [" << i << "] = " << static_cast<int>(vec[i]) << "\n";
    }
}


inline std::vector<uint8_t> unpack_bitstring(uint64_t packed, size_t length) {
    std::vector<uint8_t> bits(length);
    for (size_t i = 0; i < length; ++i) {
        bits[i] = (packed >> i) & 1;
    }
    return bits;
}



inline void form_defects(std::vector<uint8_t>& anc_outcome, int n_anc, int rds, Real q_readout, int Reset_ancilla, int include_stab_reconstruction) {
    if (include_stab_reconstruction == 1) {
        rds += 1;
    }

    if (q_readout > 1e-20) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < n_anc * (rds - 1); ++i) {
            anc_outcome[i] ^= (dis(rand_gen) < q_readout) ? 1 : 0;
        }
    }

    int XOR_times = Reset_ancilla ? 1 : 2;

    for (int k = 0; k < XOR_times; ++k) {
        for (int anc = 0; anc < n_anc; ++anc) {
            for (int rd = rds - 1; rd >= 1; --rd) {
                int indx1 = anc + n_anc * rd;
                int indx2 = anc + n_anc * (rd - 1);
                anc_outcome[indx1] ^= anc_outcome[indx2];
            }
        }
    }
}


inline std::vector<uint8_t> get_outcome_per_rd(const std::vector<uint8_t>& anc_outcome, int n_anc, int rd){

    auto start_it = anc_outcome.begin() + rd * n_anc;
    return std::vector<uint8_t>(start_it, start_it + n_anc);    

}


inline bool logical_XL_flipped(const std::vector<uint8_t>& outcome,
                               const std::vector<int>& correction) {
    assert(outcome.size() == correction.size());
    int parity = 0;
    for (size_t i = 0; i < outcome.size(); ++i) {
        parity ^= (outcome[i] ^ correction[i]);
    }
    return parity;
}


std::vector<std::vector<int>> Hx_rep_code(int d){
    
    std::vector<std::vector<int>> Hx(d - 1, std::vector<int>(d, 0));

    for  (int k1=0; k1<d-1; ++k1){

        Hx[k1][k1] = 1;
        Hx[k1][k1+1]=1;
        
    }

    return Hx;

}




