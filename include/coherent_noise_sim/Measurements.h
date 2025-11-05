#pragma once
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include <random>
#include <iostream>
#include <cassert>


#include "PrecisionOfTypes.h"


using namespace Eigen;
using std::vector;

static thread_local std::mt19937 random_generator(std::random_device{}());

unsigned int pack_outcome(const std::vector<uint8_t>& outcome);

//Can fit up to 64 bits for the key (i.e., the size of the outcome vector should be less than that.)
inline uint64_t pack_outcome_inline(const std::vector<uint8_t>& outcome) {
    uint64_t key = 0;
    for (auto bit : outcome) {
        key = (key << 1) | bit;
    }
    return key;
}


inline std::vector<size_t> build_kept_indices(int n_total, const std::vector<int>& anc_idx, const std::vector<uint8_t>& outcome,  const std::vector<int>& shifted_anc_inds, const std::vector<int>& data_positions) {

    const int n_anc = anc_idx.size();
    const int n_data = n_total - n_anc;
    const size_t n_kept = 1ULL << n_data;

    std::vector<size_t> kept(n_kept);
    
    // Build ancilla pattern
    size_t ancilla_mask = 0;
    for (size_t i = 0; i < static_cast<size_t>(n_anc); ++i) {
        ancilla_mask |= static_cast<size_t>(outcome[i]) << shifted_anc_inds[i];
    }


    // Generate all full basis indices consistent with ancilla pattern
    for (size_t raw = 0; raw < n_kept; ++raw) {
        size_t bits = ancilla_mask;
        for (int j = 0; j < n_data; ++j) {
            bits |= ((raw >> j) & 1ULL) << data_positions[j]; //No branching -- maybe slightly better, and more opportunity for vectorization
        }
        
        kept[raw] = bits;
    }

    return kept;
}



inline unsigned short int project_on_indices_inplace(VectorXc& psi, const std::vector<size_t>& kept_indices, VectorXc& psi_buffer) {
    /*
    Project the state on a particular measurement outcome.
    
    Inputs:
    psi: The state vector before the projection
    kept_indices: the indices corresponding to entries of the state vector that we want to project
    psi_buffer: a state vector buffer 
    
    Outputs:
    flag 0/1: if 0 the projection worked w/o any problems.
    */                                                        
    Complex* psi_ptr = psi.data();

    Real norm_sq = 0.0;
    for (size_t idx : kept_indices) {

        norm_sq += std::norm(psi_ptr[idx]);
    }

    if (norm_sq < 1e-10) {
        // This should not happen unless we have a bug.
        throw std::runtime_error("Encountered zero norm when projecting on indices.");
    }

    
    psi_buffer.setZero(); //Might be costly if kept_indices is big, but it is cleaner.
    
    const Real norm_inv = static_cast<Real>(1.0 / std::sqrt(norm_sq));

    for (size_t idx : kept_indices) {
        psi_buffer[idx] = psi_ptr[idx] * norm_inv;
    }
    
    psi.swap(psi_buffer);

    return 0;
}


inline void measure_all_data(const int n_data, const std::vector<int>& shifted_data_bits_from_d, const std::vector<Real>& cumulative,  std::vector<uint8_t>& data_outcome){
                                               
    /*
    Measure all the data qubits. Assumes that the ancilla qubits have been traced out and we have a state vector of size 2^{n_data}.
    It samples and overwrites the data_outcome vector inplace.

    Inputs:
    n_data: # of data qubits
    shifted_data_bits_from_d: bit-shifted indices of data qubits (1...d for repetition code) which are shifted to represent the MSB ordering.
    cumulative: cumulative vector (has calculated the cumulative prob distribution from the state vector)
    data_outcome: a vector of length n_data, on which we overwrite the outcome

    */
    
    static thread_local std::uniform_real_distribution<Real> dist(0.0, 1.0);
    Real r = dist(random_generator);

    auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);   
    size_t sampled_idx = it - cumulative.begin();


    const int* bit_shifts = shifted_data_bits_from_d.data();
    uint8_t* out_ptr      = data_outcome.data();

    for (int i = 0; i < n_data; ++i) {
        out_ptr[i] = static_cast<uint8_t>((sampled_idx >> bit_shifts[i]) & 1);
    }

}

//TODO: Think about the bottleneck: O(dim) (worst-case) sampling here which is recomputed.
inline std::vector<uint8_t> measure_all_ancilla_NEW(int nQ,int n_anc,const std::vector<int>& idxs_anc, VectorXc& psi,
                                                               std::unordered_map<uint64_t, std::vector<size_t>>& kept_indices_cache, 
                                                               const std::vector<int>& shifted_anc_inds, const std::vector<int>& data_positions,
                                                               VectorXc& psi_buffer) {
    

    const size_t dim = psi.size();
    const Complex* psi_data = psi.data();

    // Sample a point in the cumulative distribution
    static thread_local std::uniform_real_distribution<Real> dist(0.0, 1.0);
    Real sample = dist(random_generator) ; //* total_prob

    Real partial = 0.0;
    size_t idx = 0; //idx is the observed basis state
    for (; idx < dim; ++idx) {
        partial += std::norm(psi_data[idx]);
        if (partial >= sample) break;
    }

    // Compute ancilla outcome bits
    std::vector<uint8_t> outcome(n_anc);
    for (int i = 0; i < n_anc; ++i) {
        outcome[i] = (idx >> shifted_anc_inds[i]) & 1; //interpret the sampled idx as a bitstring and extract the ancilla bit
    }

    // Cache lookup or insertion
    uint64_t key = pack_outcome_inline(outcome);
    auto it = kept_indices_cache.find(key);
    if (it == kept_indices_cache.end()) {
        it = kept_indices_cache.emplace(key, build_kept_indices(nQ, idxs_anc, outcome, shifted_anc_inds, data_positions)).first;
    }

    // Project the state
    unsigned short int dummy_flag;
    dummy_flag = project_on_indices_inplace(psi, kept_indices_cache[key], psi_buffer);

    
    return {outcome};
}                                                                           
                                                                       
// Difference in this function compared to the above, is that here we pass the cumulative distribution 
// vector which is fixed and we dont re-calculate it.
inline std::vector<uint8_t> measure_all_ancilla_first_rd(int nQ,int n_anc,const std::vector<int>& idxs_anc, VectorXc& psi,
                                                               std::unordered_map<uint64_t, std::vector<size_t>>& kept_indices_cache, 
                                                               const std::vector<int>& shifted_anc_inds, const std::vector<int>& data_positions,
                                                               const std::vector<Real>& cumulative, VectorXc& psi_buffer){
    
    //Measure all the data qubits. Assumes that the ancilla qubits have been traced out already.
    static thread_local std::uniform_real_distribution<Real> dist(0.0, 1.0);
    Real r = dist(random_generator);

    auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);   
    size_t sampled_idx = it - cumulative.begin();

    std::vector<uint8_t> outcome(n_anc);

    const int* bit_shifts = shifted_anc_inds.data();
    
    for (int i = 0; i < n_anc; ++i) {
        outcome[i] = static_cast<uint8_t>(
            (sampled_idx >> bit_shifts[i]) & 1
        );
    }

    // Cache lookup or insertion
    uint64_t key = pack_outcome_inline(outcome);
    auto IT = kept_indices_cache.find(key);
    if (IT == kept_indices_cache.end()) {
        IT = kept_indices_cache.emplace(key, build_kept_indices(nQ, idxs_anc, outcome, shifted_anc_inds, data_positions)).first;
    }

    // Project the state
    unsigned short int dummy_flag;
    dummy_flag = project_on_indices_inplace(psi, kept_indices_cache[key],psi_buffer);

    if (dummy_flag==1){

        throw std::runtime_error("Sampled from outcome that cannot happen.");
    }

    return outcome;
}


// This function assumes psi is supported only on basis states with ancilla = outcome_this_rd.
// It resets ancilla bits from outcome_this_rd to 0 by relabeling basis states.

inline void reset_ancillas_to_zero(VectorXc& psi,
                            const std::vector<uint8_t>& outcome_this_rd,
                            const std::vector<int>& idxs_anc,
                            int nQ) {

    const int dim = psi.size();
    VectorXc psi_new = VectorXc::Zero(dim);

    for (int i = 0; i < dim; ++i) {
        // Check if amplitude nonzero
        if (std::norm(psi[i]) < 1e-15) continue;

        // Extract ancilla bits from i
        uint64_t ancilla_bits = 0;
        for (size_t anc_i = 0; anc_i < idxs_anc.size(); ++anc_i) {
            int q = idxs_anc[anc_i];
            uint64_t bit_val = (i >> (nQ - 1 - q)) & 1ULL;
            ancilla_bits |= (bit_val << (idxs_anc.size() - 1 - anc_i));
        }

        // If ancilla bits do not match outcome_this_rd, skip (shouldn't happen if already projected)
        uint64_t outcome_val = 0;
        for (size_t anc_i = 0; anc_i < outcome_this_rd.size(); ++anc_i) {
            outcome_val |= (static_cast<uint64_t>(outcome_this_rd[anc_i]) << (outcome_this_rd.size() - 1 - anc_i));
        }
        if (ancilla_bits != outcome_val) continue;

        // Now compute new index j where ancilla bits are zero, data bits unchanged
        int j = i;
        for (size_t anc_i = 0; anc_i < idxs_anc.size(); ++anc_i) {
            int q = idxs_anc[anc_i];
            // clear ancilla bit q in j
            j &= ~(1 << (nQ - 1 - q));
        }

        psi_new[j] = psi[i];
    }

    psi = psi_new;
    // No need to normalize because this is just a permutation of basis states.
}


inline VectorXc partial_trace_pure_state(const VectorXc& psi,
                                  const std::vector<int>& idxs_data,
                                  const std::vector<int>& idxs_anc,
                                  int nQ){
    const int dim_reduced = 1 << idxs_data.size();
    VectorXc psi_reduced = VectorXc::Zero(dim_reduced);

    for (int i = 0; i < psi.size(); ++i) {
        int reduced_idx = 0;
        for (size_t j = 0; j < idxs_data.size(); ++j) {
            int q = idxs_data[j];
            int bit = (i >> (nQ - 1 - q)) & 1;
            reduced_idx |= (bit << (idxs_data.size() - 1 - j));
        }
        psi_reduced[reduced_idx] += psi[i];
    }

    psi_reduced.normalize(); // optional, depending on context
    return psi_reduced;
}



std::vector<std::pair<int, int>> precompute_index_map_for_ptrace(const std::vector<int>& qubits_to_keep,
                                                                 const std::vector<int>& qubits_discarded,
                                                                 int n_total);
                                                                 
std::vector<std::pair<int,int>> precompute_kept_index_map_for_ptrace_of_ancilla(int n_anc, int n_data);

VectorXc discard_measured_qubits(const VectorXc& psi_full,
                                   const std::vector<int>& qubits_to_keep,
                                   const std::vector<int>& qubits_discarded,
                                   const std::vector<uint8_t>& measured_values,
                                   int n_total);


VectorXc discard_measured_qubits_NEW(const VectorXc& psi_full,
                                      const std::vector<int>& qubits_to_keep,
                                      const std::vector<int>& qubits_discarded,
                                      const std::vector<uint8_t>& measured_values,
                                      int n_total);                                   
