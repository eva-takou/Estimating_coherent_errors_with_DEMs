#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include "PrecisionOfTypes.h"

using namespace Eigen;
using std::vector;


std::vector<std::pair<int,int>> precompute_kept_index_map_for_ptrace_of_ancilla(int n_anc, int n_data){
    /*
    Compute the indices we will keep in the state vector after tracing out the ancilla. The state vector is ordered as |psi,data, psi_ancilla>.
    Note: This also assumes that the ancilla qubits were reset! So this is only valid if the state vector was projected to the 0 outcome for all ancilla (or X gates were used to return the ancilla to 0).
    
    Inputs:
    n_anc: # of ancilla qubits
    n_data: # of data qubits

    Output:
    index_map: a vector of pairs where the 1st index is which entry we select from the full vector, and the 2nd index is the entry in the reduced vector.
    */
    std::vector<std::pair<int,int>> index_map;
    const int dim_anc  = 1 << n_anc; //same as std::pow(2,n);
    const int dim_data = 1 << n_data;

    for (int i=0; i<dim_data; ++i){

        index_map.emplace_back( i * dim_anc , i);

    }

    return index_map;

}


VectorXc discard_measured_qubits(const VectorXc& psi_full, const std::vector<int>& qubits_to_keep, const std::vector<int>& qubits_discarded, const std::vector<uint8_t>& measured_values, int n_total) {

    /*
    Discard the qubits that were measured, given the measurement outcomes that they were already projected into. This is used for example, if we want to discard the ancilla qubits, and we haven't reset the ancilla.
    
    Inputs:
    psi_full: total state vector
    qubits_to_keep: vector of qubit indices to retain
    qubits_discarded: vector of qubit indices to remove
    measured_values: vector of measurement outcomes of the qubits to discard
    n_total: total number of qubits
    
    Output:
    the reduced state after removing the discarded qubits.
    */                                    
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



