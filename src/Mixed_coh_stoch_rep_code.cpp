#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <complex>
#include <queue>


#include "estimation_functions_rep_code.h"
#include "Measurements.h"
#include "Kets.h"
#include "Unitary_Ops.h"
#include "Stochastic_Ops.h"
#include "call_to_pymatching.h"
#include "utils.h"
#include <cstdint>


#include <utility>
#include <chrono>
#include <iostream>
#include <set>
#include <numeric>


#include <Eigen/Dense>
#include <stdexcept>

#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>  // Needed for custom strides


#include <Eigen/Eigenvalues>


#include <chrono>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "pcg_random.hpp"
#include <type_traits>


#include "PrecisionOfTypes.h"

#include "constants.h"

using std::vector;
using namespace Eigen;



struct DataOutcome {
    std::vector<uint8_t> bitstring;  // data qubit bitstring
    Real probability;
};

struct AncillaOutcome {
    std::vector<uint8_t> bitstring;  // ancilla qubit bitstring
    Real probability;
    std::vector<DataOutcome> data_outcomes;  // nested vector of data outcomes
};


struct VectorHash {
    size_t operator()(const std::vector<uint8_t>& v) const {
        std::hash<uint8_t> hasher;
        size_t seed = 0;
        for (uint8_t i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};





inline void reinitialize_ancilla(const VectorXc& psi_full, VectorXc& psi_full_out, int n_anc) {
    /*
    Prepare again the |psi_data>\otimes |+>^{n_anc} state after measuring and reseting the ancilla qubits.
    
    Inputs:
    psi_full: the total state after measuring and reseting ancilla
    psi_full_out: same as psi_full
    n_anc: number of ancilla qubits

    Output (inline):
    the updated state |psi_data>\otimes |+>^{n_anc}

    */
    const Eigen::Index dim_anc  = 1 << n_anc;
    const Eigen::Index dim_data = psi_full.size() / dim_anc;
    const Real scale = 1.0 / std::sqrt(static_cast<Real>(dim_anc));

    // Compute norm of data qubit state (outcome of ancilla is 000...0)
    Real norm_sq = 0.0;
    const Complex* src = psi_full.data();
    for (Eigen::Index i = 0; i < dim_data; ++i) {
        Complex val = src[i * dim_anc];
        norm_sq += std::norm(val);
    }
    const Real norm = std::sqrt(norm_sq);
    const Real overall_scale = scale / norm; // = 1/(norm * sqrt(2^n_anc))

   
    psi_full_out.resize(psi_full.size());

    // Create the output state
    Complex* dst = psi_full_out.data();
    src = psi_full.data();
    for (Eigen::Index i = 0; i < dim_data; ++i) {
        Complex val = src[i * dim_anc] * overall_scale;
        Complex* block = dst + (i * dim_anc);
        for (Eigen::Index j = 0; j < dim_anc; ++j) {
            block[j] = val;
        }
    }
}


inline std::vector<std::vector<std::pair<size_t, size_t>>> get_CNOTs_as_swap_layers(int d){


    int nQ = d + (d-1);

    std::vector<std::vector<std::pair<size_t, size_t>>> all_swaps;

    int control = d;
    for (int i = 0; i < d-1; ++i) {
        std::vector<int> targets{ i, i + 1 };
        auto swaps = precompute_CNOT_swaps(control, targets, nQ);
        control += 1;

        all_swaps.push_back(swaps);
    }


    return all_swaps;
    
}


inline std::vector<ArrayXc> get_ZZ_masks_as_layers(int d, Real theta_G){

    int nQ = d+d-1;
    std::vector<ArrayXc> ZZ_mask_per_layer;
    ZZ_mask_per_layer.reserve(d-1);
    ArrayXc ZZ_mask = VectorXc::Ones(1 << nQ);    

    for (int i=0; i<d-1; ++i){

        ArrayXc temp1 = compute_ZZ_phase_mask(nQ, d + i, i, theta_G);
        ZZ_mask      *= temp1;
        temp1         = compute_ZZ_phase_mask(nQ, d + i, i + 1, theta_G);
        ZZ_mask      *= temp1;

        ZZ_mask_per_layer.push_back(ZZ_mask);

        ZZ_mask = VectorXc::Ones(1 << nQ);    

    }    


    return ZZ_mask_per_layer;
    
}

inline ArrayXc prepare_reusable_structures(int d, int nQ, int n_anc, const std::vector<int>& idxs_all, 
                                                                                                        Real theta_data, Real theta_anc){


    /*
    Precompute structures that remain constant for the QEC memory experiment.
    
    Input: 
    d: distance of the repetition code
    nQ: total # of qubits
    n_anc: number of ancilla qubits
    idxs_all: vector of all the qubit indices
    theta_data: error angle for e^{-i\theta Z} operation for data qubits
    theta_anc: error angle for e^{-i\theta Z} operation for ancilla qubits
    theta_G: error angle for e^{i \theta ZZ} after CNOTs
    
    Output:
    phase_mask: phase mask for e^{-i\theta Z} errors
    */



    std::vector<Real> thetas(d, theta_data);        //Same \theta angle for all data qubits (d in total)
    thetas.insert(thetas.end(), d - 1, theta_anc);  //Same \theta angle for all ancilla qubits (d-1 in total)   
    
    ArrayXc phase_mask = precompute_Rz_phase_mask(nQ, idxs_all,  thetas);


    return phase_mask;
}


VectorXc prepare_pre_meas_state_for_circuit_level_mixed_stoch_coh(int d, const std::vector<int>& all_qubits, const Real prob_depol1, const Real prob_depol2,  const ArrayXc& phase_mask, const std::vector<std::vector<std::pair<size_t, size_t>>>& swap_layers , const std::vector<ArrayXc>& ZZ_mask_per_layer) { 
    /*
    
    Input: 
    d: distance of repetition code
    all_swaps: a vector of pairs of indices for swaps (CNOTs)
    phase_mask: the phase mask for the e^{i\theta_j Z}^{\otimes n} operations on the qubits
    ZZ_mask: the phase mask for the e^{i\theta ZZ} errors
    
    Output:
    psi: The state after the operations. 
    */

    int nQ       = d+(d-1);
    VectorXc psi = Ket0(nQ);
 
    apply_Hadamard_on_all_qubits(psi); //Put qubits in X-basis
    
    apply_precomputed_Rz_mask(psi, phase_mask); //Apply noise e^{-i\theta_j Z_j}



    apply_single_depol_on_qubits(psi,  all_qubits, prob_depol1,  nQ);

    int control = d;
    for (int i=0; i<d-1; ++i){
        
        apply_CNOTs_from_precomputed_swaps(swap_layers[i], psi); //Apply perfect CNOTs
        
        apply_precomputed_ZZ_mask(psi, ZZ_mask_per_layer[i]); 

        //Apply 2-qubit depolarizing
        std::vector<int> targets{ i, i + 1 };
        apply_twoQ_depol_on_qubits(psi,  {control,control},  targets, prob_depol2,  nQ);

        control +=1 ;


    }    
    
    apply_hadamards_on_ancilla_qubits(psi,d); //Rotate ancilla before Z-basis measurement


    return psi;
}

inline void prepare_state_again_for_circuit_level_mixed_stoch_coh(VectorXc &psi, int d, std::vector<int>& all_qubits, const Real prob_depol1, const Real prob_depol2, const ArrayXc& phase_mask, const std::vector<std::vector<std::pair<size_t, size_t>>> &swap_layers,
                                                                 const std::vector<ArrayXc>& ZZ_mask_per_layer){ 
    /*
    Re-prepare the state for every QEC round. The input state needs to be in |\psi>_{data} \otimes |+>_{ancilla}.
    Input:
    psi: The state vector
    d: the distance of the repetition code
    all_swaps: vector of pairs of indices to implement the swaps
    phase_mask: the phase mask for the e^{-i\theta_j Z_j} qubit errors
    ZZ_mask: the phase mas for the e^{-i\theta ZZ} errors after the CNOTs
    */
    int nQ = d +(d-1);
    apply_precomputed_Rz_mask(psi, phase_mask); //Rz errors
    apply_single_depol_on_qubits(psi,  all_qubits, prob_depol1,  nQ);

    int control = d;                                                                    
    for (int i=0; i<d-1; ++i){
        
        apply_CNOTs_from_precomputed_swaps(swap_layers[i], psi); //Apply perfect CNOTs
        apply_precomputed_ZZ_mask(psi, ZZ_mask_per_layer[i]); 

        std::vector<int> targets{ i, i + 1 };
        apply_twoQ_depol_on_qubits(psi,  {control,control},  targets, prob_depol2,  nQ);

        control+=1;

    }       
    
    apply_hadamards_on_ancilla_qubits(psi,d); //H on ancillas

    return;


}


std::tuple<std::vector<std::vector<uint8_t>>,std::vector<uint8_t>> sample_circ_level_mixed_coh_stoc_rep_code(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real theta_G, Real q_readout, Real prob_depol1, Real prob_depol2,  bool Reset_ancilla){
    
    /*
    Get the detection events and observable flips for a mixed stoch-coherent noise model for a repetition code memory.

    Inputs:
    d: distance of repetition code
    rds: the number of QEC rounds
    ITERS: the Monte Carlo iterations 
    theta_data: the error angle for data qubits e^{-i\theta_data Z}
    theta_anc:  the error angle for ancilla qubits e^{-i\theta_anc Z}
    theta_G:    the error angle for gate errors e^{i\theta_G Z_{control} Z_{target}}
    q_readout:  the classical readout error
    prob_depol1: probability for input single qubit depolarizing
    Reset_ancilla: to reset or not the ancilla qubits

    Output:
    Detection events and observable flips.
    */

    // Fixed values/vectors
    const int n_anc  = d - 1;
    const int n_data = d;    
    const int nQ     = n_data+n_anc;


    bool include_stab_reconstruction = true;    
    int rds_effective = rds + (include_stab_reconstruction ? 1 : 0);

    std::vector<int>  idxs_data(n_data);
    for (int i=0; i<d; ++i){ idxs_data[i]=i;}

    std::vector<int> idxs_anc(n_anc);
    for (int i = 0; i < n_anc; ++i) idxs_anc[i] = i + d;

    std::vector<int> idxs_all(nQ);
    for (int i = 0; i < nQ; ++i) idxs_all[i] = i;

    std::vector<int> shifted_anc_inds(n_anc);
    
    for (int i = 0; i < n_anc; ++i) {
        shifted_anc_inds[i] = nQ - 1 - idxs_anc[i];
    }    

    std::vector<int> shifted_data_bits_from_d(n_data);
    for (int i=0; i<n_data; ++i){
        shifted_data_bits_from_d[i] = n_data - 1 - idxs_data[i]; //Note this is shift from d -- if the state vector has d qubits
    }

    std::vector<int> data_positions;
    data_positions.reserve(n_data);

    std::vector<bool> is_anc(nQ, false);
    for (int i : idxs_anc) {
        is_anc[nQ - 1 - i] = true;
    }

    for (int bit = 0; bit < nQ; ++bit) {
        if (!is_anc[bit]) {
            data_positions.push_back(bit);
        }
    }    
    
    
    std::vector<uint8_t> outcome_of_data(n_data); 
    std::vector<uint8_t> outcome_this_rd(n_anc);
    std::vector<uint8_t> ancilla_bitstring;
    ancilla_bitstring.reserve(n_anc * rds_effective); 

    
    
    ArrayXc phase_mask;
    phase_mask = prepare_reusable_structures( d,  nQ,  n_anc, idxs_all, theta_data,  theta_anc,  theta_G);

    
    std::vector<std::vector<std::pair<size_t, size_t>>> swap_layers =  get_CNOTs_as_swap_layers(d);
    std::vector<ArrayXc> ZZ_mask_per_layer                           = get_ZZ_masks_as_layers( d,  theta_G);

    

    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, d);
    std::unordered_map<uint64_t, std::vector<size_t>> kept_indices_cache; 


    VectorXc psi_data(1 << d);
    VectorXc psi_buffer(1<<nQ);

    std::vector<Real> cumsum_data(1<<d);

    
    std::vector<std::vector<uint8_t>> all_data_outcomes;
    all_data_outcomes.resize(ITERS);

    
    std::vector<std::vector<uint8_t>> batch;
    batch.resize(ITERS);
    std::vector<uint8_t> obs_flips;
    obs_flips.resize(ITERS);

    
    const Eigen::Index dim = 1 << nQ; 

    for (int iter=0; iter<ITERS; ++iter){

        VectorXc psi  = prepare_pre_meas_state_for_circuit_level_mixed_stoch_coh(d,  idxs_all,  prob_depol1,  prob_depol2,  phase_mask,  swap_layers , ZZ_mask_per_layer);

        ancilla_bitstring.clear(); //Reset

        for (int r = 0; r < rds; ++r) {
            
            outcome_this_rd = measure_all_ancilla(nQ,n_anc,idxs_anc,psi,kept_indices_cache, shifted_anc_inds, data_positions,psi_buffer);

            if (Reset_ancilla==1){

                apply_X_on_qubits(psi, outcome_this_rd,d, dim, nQ); //"Reset" the ancilla (more efficient than tracing out and starting again in |0>)
            }

            // Store outcome
            ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());

            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {

                reinitialize_ancilla(psi,psi,n_anc);
                prepare_state_again_for_circuit_level_mixed_stoch_coh(psi, d, idxs_all,  prob_depol1,  prob_depol2,  phase_mask, swap_layers, ZZ_mask_per_layer);
            
            }
            
        }

        //Now measure data qubits

        if (Reset_ancilla==1){

            for (const auto& [i_full, i_reduced] : index_map)
                psi_data[i_reduced] = psi[i_full];           
            
        }
        else{
            psi_data = discard_measured_qubits(psi, idxs_data, idxs_anc, outcome_this_rd, nQ); //Need to discard based on measurement outcomes
        }
        
        psi_data.normalize();
        apply_Hadamard_on_all_qubits(psi_data);


        cumSum_from_state_vector(psi_data, cumsum_data);
        
        measure_all_data(d,shifted_data_bits_from_d,cumsum_data,outcome_of_data); 

        all_data_outcomes[iter] = outcome_of_data;

        if (include_stab_reconstruction==1){

            for (int k=0; k<d-1; ++k){
                ancilla_bitstring.push_back( outcome_of_data[k] ^ outcome_of_data[k+1]);
            }

            uint8_t total_xor = 0;
            for (uint8_t v : outcome_of_data) {
                total_xor ^= v;
            }            

            obs_flips.push_back( total_xor); //Logical can be defined on all the qubits
        }

        form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);


        batch[iter] = ancilla_bitstring;

    }


    return std::make_tuple(batch, obs_flips);
}
