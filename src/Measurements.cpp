#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include "PrecisionOfTypes.h"

using namespace Eigen;
using std::vector;





std::tuple<VectorXc, unsigned short int> project_on_indices(const VectorXc& psi, const std::vector<size_t>& kept_indices) {
    //Function to project on a particular subspace based on a measured outcome.
    //psi: The full state vector before the measurement
    //kept_indices: incides corresponding to a particular measurement outcome that we intend to project the state onto.
    //Output:
    //psi_f , encountered_zero_norm: the output state, and a flag 0/1 if we encountered zero norm.
    
    const size_t dim = psi.size();
    VectorXc psi_f = VectorXc::Zero(dim);
    Real norm_sq = 0.0;

    const Complex* psi_ptr = psi.data();
    Complex* psi_f_ptr = psi_f.data();    

    for (size_t idx : kept_indices) {
        psi_f_ptr[idx] = psi_ptr[idx];
        norm_sq += std::norm(psi_ptr[idx]);
    }

    if (norm_sq < 1e-28) {
        return {psi_f, 1};
    }

    psi_f /= std::sqrt(norm_sq);
    return {psi_f, 0};
}





//Function to do partial trace of some qubits, assuming they have been measured and we projected already on the 0000... state
//of the measured qubits. This returns the indices that we keep of the vector
//So if we create a new vector psi_new[i] = psi_old[j] then, here we return a list of vector (i,j)
std::vector<std::pair<int, int>> precompute_index_map_for_ptrace(const std::vector<int>& qubits_to_keep,
                                                                 const std::vector<int>& qubits_discarded,
                                                                 int n_total) {
    
    const int dim_full = 1 << n_total;
    const int n_keep   = qubits_to_keep.size();

    // Map from qubit index to keep position
    std::vector<int> keep_pos(n_total, -1);
    for (int j = 0; j < n_keep; ++j)
        keep_pos[qubits_to_keep[j]] = j;

    // Discarded bitmask
    std::vector<bool> is_discarded(n_total, false);
    for (int q : qubits_discarded)
        is_discarded[q] = true;

    std::vector<std::pair<int, int>> index_map;
    index_map.reserve(1 << n_keep); // Max number of valid entries

    for (int i = 0; i < dim_full; ++i) {
        bool valid = true;
        int reduced_index = 0;

        for (int k = 0; k < n_total; ++k) {
            bool bit = (i >> (n_total - 1 - k)) & 1;

            if (keep_pos[k] != -1) {
                reduced_index |= (bit << (n_keep - 1 - keep_pos[k]));
            } else if (is_discarded[k]) {
                if (bit != 0) {
                    valid = false;
                    break;
                }
            }
        }

        if (valid) {
            index_map.emplace_back(i, reduced_index);
        }
    }

    return index_map;
}

//When we trace out the ancilla qubits which are ordered last, we have a very obvious pattern for the index_map
//Note: we also assume that the ancilla qubits were reset, so this is valid if the state vector was projected
//to the 0 outcome for all the ancilla (or we reset it to all 0 via X gates).
//Also we assume that the state vector is ordered as |psi_data,psi_ancilla>
std::vector<std::pair<int,int>> precompute_kept_index_map_for_ptrace_of_ancilla(int n_anc, int n_data){

    std::vector<std::pair<int,int>> index_map;
    const int dim_anc  = 1 << n_anc; //same as std::pow(2,n);
    const int dim_data = 1 << n_data;

    for (int i=0; i<dim_data; ++i){

        index_map.emplace_back( i * dim_anc , i);

    }

    return index_map;

}



//Function to discard the qubits that were measured, given the measurement outcomes that they were already projected into.
//This is used for example, if we want to discard the ancilla qubits, and we haven't done a reset.
VectorXc discard_measured_qubits(const VectorXc& psi_full,
                                   const std::vector<int>& qubits_to_keep,
                                   const std::vector<int>& qubits_discarded,
                                   const std::vector<uint8_t>& measured_values,
                                   int n_total) {


    if (qubits_discarded.size() != measured_values.size()) {
        throw std::invalid_argument("qubits_discarded and measured_values must have the same size.");
    }

    // Build map from discarded qubit index to its measured value
    std::unordered_map<int, uint8_t> discard_map;
    for (size_t i = 0; i < qubits_discarded.size(); ++i) {
        discard_map[qubits_discarded[i]] = measured_values[i];
    }

    int n_keep = qubits_to_keep.size();
    int dim_keep = 1 << n_keep;

    VectorXc psi_reduced = VectorXc::Zero(dim_keep);

    for (int i = 0; i < psi_full.size(); ++i) {
        bool valid = true;
        int reduced_index = 0;

        for (int k = 0; k < n_total; ++k) {
            bool bit = (i >> (n_total - 1 - k)) & 1;

            auto it_keep = std::find(qubits_to_keep.begin(), qubits_to_keep.end(), k);
            if (it_keep != qubits_to_keep.end()) {
                int pos = std::distance(qubits_to_keep.begin(), it_keep);
                reduced_index |= (bit << (n_keep - 1 - pos));
            } else {
                auto it_discard = discard_map.find(k);
                if (it_discard == discard_map.end()) {
                    throw std::runtime_error("Qubit index not found in either keep or discard lists.");
                }
                if (bit != it_discard->second) {
                    valid = false;
                    break;
                }
            }
        }

        if (valid) {
            psi_reduced[reduced_index] = psi_full[i];
        }
    }

    return psi_reduced;
}


VectorXc discard_measured_qubits_NEW(const VectorXc& psi_full,
                                      const std::vector<int>& qubits_to_keep,
                                      const std::vector<int>& qubits_discarded,
                                      const std::vector<uint8_t>& measured_values,
                                      int n_total) {

    if (qubits_discarded.size() != measured_values.size()) {
        throw std::invalid_argument("qubits_discarded and measured_values must have the same size.");
    }

    int n_keep = qubits_to_keep.size();
    int dim_keep = 1 << n_keep;

    // Precompute: fast lookup table for which qubits are kept, and their positions
    std::vector<int> qubit_roles(n_total, -1);  // -1 = discard, >=0 = keep position
    for (int i = 0; i < n_keep; ++i)
        qubit_roles[qubits_to_keep[i]] = i;

    // Build discard bitmask
    size_t discard_mask = 0;
    size_t discard_pattern = 0;
    for (size_t i = 0; i < qubits_discarded.size(); ++i) {
        int bitpos = n_total - 1 - qubits_discarded[i];
        discard_mask |= (1ULL << bitpos);
        if (measured_values[i]) {
            discard_pattern |= (1ULL << bitpos);
        }
    }

    VectorXc psi_reduced = VectorXc::Zero(dim_keep);

    for (size_t full_idx = 0; full_idx < psi_full.size(); ++full_idx) {
        // Reject invalid indices based on discarded bits
        if ((full_idx & discard_mask) != discard_pattern)
            continue;

        // Build reduced index from kept bits
        size_t reduced_idx = 0;
        for (int i = 0; i < n_total; ++i) {
            int keep_pos = qubit_roles[i];
            if (keep_pos >= 0) {
                bool bit = (full_idx >> (n_total - 1 - i)) & 1;
                reduced_idx |= (bit << (n_keep - 1 - keep_pos));
            }
        }

        psi_reduced[reduced_idx] = psi_full[full_idx];
    }

    return psi_reduced;
}
