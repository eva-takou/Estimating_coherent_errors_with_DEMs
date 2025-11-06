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

//Maybe unused?
// #include <Spectra/SymEigsSolver.h>
// #include <Spectra/MatOp/DenseSymMatProd.h>
// #include <Spectra/Util/SelectionRule.h>

#include <chrono>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "pcg_random.hpp"
#include <type_traits>


#include "PrecisionOfTypes.h"
#include "constants.h"

using std::vector;
// using namespace Spectra;
using namespace Eigen;



// using Time          = double;
// using Clock         = std::chrono::high_resolution_clock;
// using Evaluate_Time = std::chrono::duration<Time_Precision>;


// constexpr Real PI        = Real(3.1415926535897932384626);
// constexpr Real SQRT2     = Real(1.4142135623730951);
// constexpr Real SQRT2_INV = Real(0.7071067811865475);

// constexpr int mantissa_bits = std::numeric_limits<Real>::digits;  // mantissa bits



struct DataOutcome {
    std::vector<uint8_t> bitstring;  // data qubit bitstring
    Real probability;
};

struct AncillaOutcome {
    std::vector<uint8_t> bitstring;  // ancilla qubit bitstring
    Real probability;
    std::vector<DataOutcome> data_outcomes;  // nested vector of data outcomes
};



// std::random_device rd;  // Seed
// pcg32 gen;

//To measure time:
// auto t0 = Clock::now();
// auto t1 = Clock::now();
// time_for_CNOT += Evaluate_Time(t1-t0).count();

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


template <typename T>
T clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}



inline void expand_with_plus_state(const VectorXc& psi_data,
                            VectorXc& psi,
                            int n_anc) {
    const int dim_data = psi_data.size();
    const int dim_full = dim_data << n_anc; // 2^(d+n_anc)
    const Real scale = 1.0 / std::sqrt(1 << n_anc);

    psi.resize(dim_full);

    // Each entry in psi_data gets copied into a contiguous block
    for (int i = 0; i < dim_data; ++i) {
        Complex val = psi_data[i] * scale;
        Complex* block = psi.data() + (i << n_anc);
        for (int j = 0; j < (1 << n_anc); ++j) {
            block[j] = val;
        }
    }
}


//TODO: WORK ON THIS
//This gets psi_data x psi_anc where psi_anc is in |+> state.
inline void reinitialize_ancilla(const VectorXc& psi_full, VectorXc& psi_full_out, int n_anc) {
    
    const Eigen::Index dim_anc  = 1 << n_anc;
    const Eigen::Index dim_data = psi_full.size() / dim_anc;
    const Real scale = 1.0 / std::sqrt(static_cast<Real>(dim_anc));

    // Step 1: compute norm over kept components
    Real norm_sq = 0.0;
    const Complex* src = psi_full.data();
    for (Eigen::Index i = 0; i < dim_data; ++i) {
        Complex val = src[i * dim_anc];
        norm_sq += std::norm(val);
    }
    const Real norm = std::sqrt(norm_sq);
    const Real overall_scale = scale / norm;

    // Step 2: resize output |ψ⟩ (same size as input)
    psi_full_out.resize(psi_full.size());

    // Step 3: fill |ψ_data⟩⊗|+>ⁿ into psi_full_out
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


VectorXc prepare_pre_meas_state(int d, const std::vector<std::pair<size_t, size_t>>& all_swaps, const ArrayXc& phase_mask, const ArrayXc& ZZ_mask) { 
    /*
    Perform all unitary operations to prepare the state for the 1st QEC round (before measuring the ancilla qubits).
    
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
    
    apply_CNOTs_from_precomputed_swaps(all_swaps, psi); //Apply perfect CNOTs

    apply_precomputed_ZZ_mask(psi, ZZ_mask);  //Apply e^{-i\theta_j ZZ} errors after CNOTs                                    
    
    apply_hadamards_on_ancilla_qubits(psi,d); //Rotate ancilla before Z-basis measurement

    return psi;
}


inline void prepare_state_again(VectorXc &psi, int d, const std::vector<std::pair<size_t, size_t>>& all_swaps,
                                const ArrayXc& phase_mask, const ArrayXc& ZZ_mask){ 
    /*
    Re-prepare the state for every QEC round. The input state needs to be in |\psi>_{data} \otimes |+>_{ancilla}.
    Input:
    psi: The state vector
    d: the distance of the repetition code
    all_swaps: vector of pairs of indices to implement the swaps
    phase_mask: the phase mask for the e^{-i\theta_j Z_j} qubit errors
    ZZ_mask: the phase mas for the e^{-i\theta ZZ} errors after the CNOTs
    */
    
    apply_precomputed_Rz_mask(psi, phase_mask); //Rz errors
    
    apply_CNOTs_from_precomputed_swaps(all_swaps, psi); //CNOTs

    apply_precomputed_ZZ_mask(psi, ZZ_mask); //ZZ-errors after the CNOTs
    
    apply_hadamards_on_ancilla_qubits(psi,d); //H on ancillas

    return;


}


inline std::tuple<std::vector<std::pair<size_t, size_t>>, ArrayXc, ArrayXc> prepare_reusable_structures(int d, int nQ, int n_anc, const std::vector<int>& idxs_all, 
                                                                                                        Real theta_data, Real theta_anc, Real theta_G){


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
    all_swaps: vector of pairs of indices to be swapped
    phase_mask: phase mask for e^{-i\theta Z} errors
    ZZ_mask: phase mask for e^{i \theta ZZ} CNOT errors
    
    */

    //Precompute indices for SWAPs that implement CNOTs
    
    std::vector<std::pair<size_t, size_t>> all_swaps;
    int control=d;
    for (int k1=0; k1<d-1; ++k1){

        std::vector<int> targets{ k1, k1 + 1 };
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps( control, targets,  nQ);
        control +=1;    
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end()); //keep it a flattened vector

    }    


    std::vector<Real> thetas(d, theta_data);        //Same \theta angle for all data qubits (d in total)
    thetas.insert(thetas.end(), d - 1, theta_anc);  //Same \theta angle for all ancilla qubits (d-1 in total)   
    
    ArrayXc phase_mask = precompute_Rz_phase_mask(nQ, idxs_all,  thetas);

    ArrayXc ZZ_mask = VectorXc::Ones(1 << nQ);    

    for (int i = 0; i < n_anc; ++i) {

        ArrayXc temp1 = compute_ZZ_phase_mask(nQ, d + i, i, theta_G);
        ZZ_mask      *= temp1;
        temp1         = compute_ZZ_phase_mask(nQ, d + i, i + 1, theta_G);
        ZZ_mask      *= temp1;

    }    

    return std::make_tuple(all_swaps, phase_mask, ZZ_mask);
}


Real get_LER_from_estimated_DEM(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real theta_G, Real q_readout, 
                                bool Reset_ancilla,  bool include_higher_order, bool print_higher_order){
    
    /*
    Get the logical error rate by measuring the parity of data qubits in the end and comparing with the obtained correction.
    Here we use the estimated DEM to decode. The estimated edge does not contain hyper-edges. The hyper-edges are only used
    to redefine the probabilities of edges. Note that if Reset_ancilla option is false, then the estimation procedure might need to be adjusted.

    Inputs:
    d: distance of repetition code
    rds: the number of QEC rounds
    ITERS: the Monte Carlo iterations 
    theta_data: the error angle for data qubits e^{-i\theta_data Z}
    theta_anc:  the error angle for ancilla qubits e^{-i\theta_anc Z}
    theta_G:    the error angle for gate errors e^{i\theta_G Z_{control} Z_{target}}
    q_readout:  the classical readout error
    Reset_ancilla: to reset or not the ancilla qubits
    include_stab_reconstruction: to include or not last measurement of data qubits in the detection events 
    include_higher_order: to consider or not higher-order correlations (hyperedges) in the estimation  
    print_higher_orer: to print or not the estimatd values of hyperedges  

    Output:
    Logical error rate.
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
    
    std::vector<std::pair<size_t, size_t>> all_swaps;
    ArrayXc phase_mask;
    ArrayXc ZZ_mask;
    std::tie(all_swaps, phase_mask,ZZ_mask) = prepare_reusable_structures( d,  nQ,  n_anc, idxs_all, theta_data,  theta_anc,  theta_G);

 

    const VectorXc psi0    = prepare_pre_meas_state(d,  all_swaps, phase_mask, ZZ_mask);
    const Eigen::Index dim = psi0.size();    

    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, d);

    std::unordered_map<uint64_t, std::vector<size_t>> kept_indices_cache; 

    VectorXc psi;    
    psi.resize(psi0.size());

    VectorXc psi_data(1 << d);
    // VectorXc psi_plus_anc = plus_state(n_anc);


    std::vector<Real> cumsum_data(1<<d);
    std::vector<Real> cdf_buffer_total(1<<nQ);

    cumSum_from_state_vector(psi0,cdf_buffer_total); //Use this for 1st round of measurements

    VectorXc psi_buffer(psi0.size());

    std::vector<std::vector<uint8_t>> all_data_outcomes;
    all_data_outcomes.resize(ITERS);

    std::vector<std::vector<int>> Hx = Hx_rep_code(d);
    std::vector<std::vector<uint8_t>> batch;
    batch.resize(ITERS);


    for (int iter=0; iter<ITERS; ++iter){

        std::memcpy(psi.data(), psi0.data(), sizeof(Complex) * psi0.size());
        ancilla_bitstring.clear(); //Reset

        for (int r = 0; r < rds; ++r) {

            if (r==0){
                outcome_this_rd = measure_all_ancilla_first_rd(nQ, n_anc,  idxs_anc,  psi, kept_indices_cache, 
                                                                shifted_anc_inds, data_positions, cdf_buffer_total,psi_buffer);
            }
            else{
                outcome_this_rd = measure_all_ancilla(nQ,n_anc,idxs_anc,psi,kept_indices_cache, shifted_anc_inds, data_positions,psi_buffer);
            }

            if (Reset_ancilla==1){

                apply_X_on_qubits(psi, outcome_this_rd,d, dim, nQ); //"Reset" the ancilla (more efficient than tracing out and starting again in |0>)
            }

            // Store outcome
            ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());

            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {

                reinitialize_ancilla(psi,psi,n_anc);
                // for (const auto& [i_full, i_reduced] : index_map)
                //     psi_data[i_reduced] = psi[i_full];           

                // psi_data.normalize();    
                
                // expand_with_plus_state(psi_data, psi, n_anc); //This is a bit faster

                prepare_state_again(psi, d,  all_swaps, phase_mask, ZZ_mask); 
            
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
        }

        form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);


        batch[iter] = ancilla_bitstring;

    }


    std::vector<Real> p_space;
    std::vector<Real> p_time;
    std::vector<Real> p_diag;
    
    auto start = std::chrono::high_resolution_clock::now();
    std::tie(p_space,p_time,p_diag) = estimate_edges_rep_code(batch, d, n_anc, rds_effective, include_higher_order, print_higher_order);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time for estimation: " << elapsed.count() << " s\n";    

    print_vector(p_space,"space");
    print_vector(p_time,"time");
    print_vector(p_diag,"diag");

    auto corrections = decode_with_pymatching_create_graph(Hx, p_space, p_time, p_diag, batch,rds, include_stab_reconstruction);
    
    
    Real LER_sum = 0.0;
    for(int iter=0; iter<ITERS; ++iter){
        LER_sum += logical_XL_flipped(all_data_outcomes[iter], corrections[iter]) ? 1.0 : 0.0;
    }

    Real LER = LER_sum / ITERS;    
   

    return LER;
}

Real get_LER_from_uniform_DEM_circuit_level(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real theta_G, Real q_readout,  bool Reset_ancilla){
    
    /*
    Get the logical error rate by measuring the parity of data qubits in the end and comparing with the obtained correction.
    Here we use a uniform DEM to decode. Note that the decoder puts space, time and diagonal edges, as for a stochastic DEM.

    Inputs:
    d: distance of repetition code
    rds: the number of QEC rounds
    ITERS: the Monte Carlo iterations 
    theta_data: the error angle for data qubits e^{-i\theta_data Z}
    theta_anc:  the error angle for ancilla qubits e^{-i\theta_anc Z}
    theta_G:    the error angle for gate errors e^{i\theta_G Z_{control} Z_{target}}
    q_readout:  the classical readout error
    Reset_ancilla: to reset or not the ancilla qubits

    Output:
    Logical error rate.
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
    
    std::vector<std::pair<size_t, size_t>> all_swaps;
    ArrayXc phase_mask;
    ArrayXc ZZ_mask;
    std::tie(all_swaps, phase_mask,ZZ_mask) = prepare_reusable_structures( d,  nQ,  n_anc, idxs_all, theta_data,  theta_anc,  theta_G);

 

    const VectorXc psi0    = prepare_pre_meas_state(d,  all_swaps, phase_mask, ZZ_mask);
    const Eigen::Index dim = psi0.size();    

    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, d);

    std::unordered_map<uint64_t, std::vector<size_t>> kept_indices_cache; 

    VectorXc psi;    
    psi.resize(psi0.size());

    VectorXc psi_data(1 << d);
    // VectorXc psi_plus_anc = plus_state(n_anc);


    std::vector<Real> cumsum_data(1<<d);
    std::vector<Real> cdf_buffer_total(1<<nQ);

    cumSum_from_state_vector(psi0,cdf_buffer_total); //Use this for 1st round of measurements

    VectorXc psi_buffer(psi0.size());

    std::vector<std::vector<uint8_t>> all_data_outcomes;
    all_data_outcomes.resize(ITERS);

    std::vector<std::vector<int>> Hx = Hx_rep_code(d);
    std::vector<std::vector<uint8_t>> batch;
    batch.resize(ITERS);


    for (int iter=0; iter<ITERS; ++iter){

        std::memcpy(psi.data(), psi0.data(), sizeof(Complex) * psi0.size());
        ancilla_bitstring.clear(); //Reset

        for (int r = 0; r < rds; ++r) {

            if (r==0){
                outcome_this_rd = measure_all_ancilla_first_rd(nQ, n_anc,  idxs_anc,  psi, kept_indices_cache, 
                                                                shifted_anc_inds, data_positions, cdf_buffer_total,psi_buffer);
            }
            else{
                outcome_this_rd = measure_all_ancilla(nQ,n_anc,idxs_anc,psi,kept_indices_cache, shifted_anc_inds, data_positions,psi_buffer);
            }

            if (Reset_ancilla==1){

                apply_X_on_qubits(psi, outcome_this_rd,d, dim, nQ); //"Reset" the ancilla (more efficient than tracing out and starting again in |0>)
            }

            // Store outcome
            ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());

            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {

                reinitialize_ancilla(psi,psi,n_anc);
                // for (const auto& [i_full, i_reduced] : index_map)
                //     psi_data[i_reduced] = psi[i_full];           

                // psi_data.normalize();    
                
                // expand_with_plus_state(psi_data, psi, n_anc); //This is a bit faster

                prepare_state_again(psi, d,  all_swaps, phase_mask, ZZ_mask); 
            
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
        }

        form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);


        batch[iter] = ancilla_bitstring;

    }

    //Set an arbitrary weight for all probabilities (equal weights) 
    std::vector<Real> p_space(rds_effective * d, 0.1); 
    std::vector<Real> p_time(rds * n_anc, 0.1);
    std::vector<Real> p_diag(rds * (n_anc-1), 0.1);
    

    auto corrections = decode_with_pymatching_create_graph(Hx, p_space, p_time, p_diag, batch, rds, include_stab_reconstruction);
    
    Real LER_sum = 0.0;
    for(int iter=0; iter<ITERS; ++iter){
        LER_sum += logical_XL_flipped(all_data_outcomes[iter], corrections[iter]) ? 1.0 : 0.0;
    }

    Real LER = LER_sum / ITERS;    
   

    return LER;
}


Real get_LER_from_uniform_DEM_phenom_level(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real q_readout,  bool Reset_ancilla){
    
    /*
    Get the logical error rate by measuring the parity of data qubits in the end and comparing with the obtained correction.
    Here we use a uniform DEM to decode. Note that the decoder puts space, time edges only.

    Inputs:
    d: distance of repetition code
    rds: the number of QEC rounds
    ITERS: the Monte Carlo iterations 
    theta_data: the error angle for data qubits e^{-i\theta_data Z}
    theta_anc:  the error angle for ancilla qubits e^{-i\theta_anc Z}
    theta_G:    the error angle for gate errors e^{i\theta_G Z_{control} Z_{target}}
    q_readout:  the classical readout error
    Reset_ancilla: to reset or not the ancilla qubits

    Output:
    Logical error rate.
    */

    // Fixed values/vectors
    const int n_anc  = d - 1;
    const int n_data = d;    
    const int nQ     = n_data+n_anc;
    const Real theta_G = 0.0;

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
    
    std::vector<std::pair<size_t, size_t>> all_swaps;
    ArrayXc phase_mask;
    ArrayXc ZZ_mask;
    std::tie(all_swaps, phase_mask,ZZ_mask) = prepare_reusable_structures( d,  nQ,  n_anc, idxs_all, theta_data,  theta_anc,  theta_G);

 
    const VectorXc psi0    = prepare_pre_meas_state(d,  all_swaps, phase_mask, ZZ_mask);
    const Eigen::Index dim = psi0.size();    

    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, d);

    std::unordered_map<uint64_t, std::vector<size_t>> kept_indices_cache; 

    VectorXc psi;    
    psi.resize(psi0.size());

    VectorXc psi_data(1 << d);
    // VectorXc psi_plus_anc = plus_state(n_anc);


    std::vector<Real> cumsum_data(1<<d);
    std::vector<Real> cdf_buffer_total(1<<nQ);

    cumSum_from_state_vector(psi0,cdf_buffer_total); //Use this for 1st round of measurements

    VectorXc psi_buffer(psi0.size());

    std::vector<std::vector<uint8_t>> all_data_outcomes;
    all_data_outcomes.resize(ITERS);

    std::vector<std::vector<int>> Hx = Hx_rep_code(d);
    std::vector<std::vector<uint8_t>> batch;
    batch.resize(ITERS);


    for (int iter=0; iter<ITERS; ++iter){

        std::memcpy(psi.data(), psi0.data(), sizeof(Complex) * psi0.size());
        ancilla_bitstring.clear(); //Reset

        for (int r = 0; r < rds; ++r) {

            if (r==0){
                outcome_this_rd = measure_all_ancilla_first_rd(nQ, n_anc,  idxs_anc,  psi, kept_indices_cache, 
                                                                shifted_anc_inds, data_positions, cdf_buffer_total,psi_buffer);
            }
            else{
                outcome_this_rd = measure_all_ancilla(nQ,n_anc,idxs_anc,psi,kept_indices_cache, shifted_anc_inds, data_positions,psi_buffer);
            }

            if (Reset_ancilla==1){

                apply_X_on_qubits(psi, outcome_this_rd,d, dim, nQ); //"Reset" the ancilla (more efficient than tracing out and starting again in |0>)
            }

            // Store outcome
            ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());

            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {

                reinitialize_ancilla(psi,psi,n_anc);
                // for (const auto& [i_full, i_reduced] : index_map)
                //     psi_data[i_reduced] = psi[i_full];           

                // psi_data.normalize();    
                
                // expand_with_plus_state(psi_data, psi, n_anc); //This is a bit faster

                prepare_state_again(psi, d,  all_swaps, phase_mask, ZZ_mask); 
            
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
        }

        form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);


        batch[iter] = ancilla_bitstring;

    }


    auto corrections = decode_batch_with_pymatching(Hx, batch, rds_effective);
    
    Real LER_sum = 0.0;
    for(int iter=0; iter<ITERS; ++iter){
        LER_sum += logical_XL_flipped(all_data_outcomes[iter], corrections[iter]) ? 1.0 : 0.0;
    }

    Real LER = LER_sum / ITERS;    
   

    return LER;
}

//THIS TO BE TESTED...
Real get_logical_infidelity(int d, int rds, int ITERS, Real theta_data,  Real q_readout, bool Reset_ancilla){
    /*
    Used only for coherent data errors, or coherent data errors + classical readout errors. Calculates the logical infidelity
    P_L = \sum_s P(s)sin^2(\theta_s), where \theta_s is the logical angle. Thus, it can only be used when the state remains in the codespace after the correction.
    
    Inputs:
    d: distance of repetition code
    rds: number of QEC rounds
    ITERS: # of Monte Carlo shots
    theta_data: error angle for data qubits (e^{-i\theta_data Z})
    q_readout: classical readout error
    Reset_ancilla: true/false to reset or not the ancilla qubits

    Output:
    LER: logical infidelity calculate from P_L = \sum_s P(s)sin^2(theta_s), where theta_s is the angle of the logical state
    */                                                               

    const int n_anc      = d - 1;
    const int n_data     = d;    
    const int nQ         = n_data+n_anc;
    Real theta_G   = 0.0;
    Real theta_anc = 0.0;
    bool include_stab_reconstruction = false;


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

    ancilla_bitstring.reserve(n_anc * rds); 
    
    std::vector<std::pair<size_t, size_t>> all_swaps;
    ArrayXc phase_mask;
    ArrayXc ZZ_mask;
    std::tie(all_swaps, phase_mask,ZZ_mask) = prepare_reusable_structures( d,  nQ,  n_anc, idxs_all, theta_data,  theta_anc,  theta_G);
 
    const VectorXc psi0    = prepare_pre_meas_state(d,  all_swaps, phase_mask, ZZ_mask);
    const Eigen::Index dim = psi0.size();    

    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, d);

    std::unordered_map<uint64_t, std::vector<size_t>> kept_indices_cache; 

    VectorXc psi;    
    psi.resize(psi0.size());

    VectorXc psi_data(1 << d);
    // VectorXc psi_plus_anc = plus_state(n_anc);

    std::vector<Real> cumsum_data(1<<d);
    std::vector<Real> cdf_buffer_total(1<<nQ);

    cumSum_from_state_vector(psi0,cdf_buffer_total); //Use this for 1st round of measurements

    VectorXc psi_buffer(psi0.size());

    std::vector<std::vector<uint8_t>> all_data_outcomes;
    all_data_outcomes.resize(ITERS);

    std::vector<std::vector<int>> Hx = Hx_rep_code(d);
    std::vector<std::vector<uint8_t>> batch;
    batch.resize(ITERS);

    std::vector<VectorXc> psi_stored;
    psi_stored.resize(ITERS);

    for (int iter=0; iter<ITERS; ++iter){

        std::memcpy(psi.data(), psi0.data(), sizeof(Complex) * psi0.size());
        ancilla_bitstring.clear(); //Reset

        for (int r = 0; r < rds; ++r) {

            if (r==0){
                outcome_this_rd = measure_all_ancilla_first_rd(nQ, n_anc, idxs_anc, psi, kept_indices_cache, shifted_anc_inds, data_positions, cdf_buffer_total, psi_buffer);
            }
            else{
                outcome_this_rd = measure_all_ancilla(nQ,n_anc,idxs_anc,psi,kept_indices_cache, shifted_anc_inds, data_positions,psi_buffer);
            }

            if (Reset_ancilla==1){

                apply_X_on_qubits(psi, outcome_this_rd, d, dim, nQ); //"Reset" the ancilla (more efficient than tracing out and starting again in |0>)
            }

            // Store outcome
            ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());

            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {

                reinitialize_ancilla(psi, psi, n_anc);
                // for (const auto& [i_full, i_reduced] : index_map)
                //     psi_data[i_reduced] = psi[i_full];           

                // psi_data.normalize();    
                
                // expand_with_plus_state(psi_data, psi, n_anc); //This is a bit faster

                prepare_state_again(psi, d,  all_swaps, phase_mask, ZZ_mask); 
            
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


        form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);

        psi_stored[iter] = psi_data;
        batch[iter]      = ancilla_bitstring;

    }

    auto corrections = decode_batch_with_pymatching(Hx, batch, rds);

    VectorXc ket0L = plus_state(d);
    VectorXc ket1L = minus_state(d);

    MatrixXc Proj = ket0L * ket0L.adjoint() + ket1L * ket1L.adjoint() ;
    
    // std::vector<Real> phi(nsims,0.0);
    std::vector<Real> thetaL(ITERS,0.0);
    // std::vector<Real> infidelity(nsims,0.0);
    // std::vector<Real> leakage(nsims,0.0);

    Real LER = 0.0;

    for (int iter=0; iter<ITERS; ++iter){

        
        VectorXc psi =  psi_stored[iter];
        
        std::vector<int> qubits_to_correct;
        for (int m=0; m<d; ++m){
            if (corrections[iter][m]==1){
                qubits_to_correct.push_back(m);
            }
        }

        //Recovery : Z operations -- up to global phases

        if (!qubits_to_correct.empty()){
            apply_Rz_on_qubits_inplace(psi, qubits_to_correct, PI/2);
        }


        Complex overlap = psi.adjoint() * Proj * psi; // <\psi|Proj_{Codespace}|\psi>

        if (overlap.imag()>1e-10){
            throw std::runtime_error("Imaginary part in leakage > 1e-10");
        }

        if (overlap.real()<1e-10){
            std::cout << "State psi:\n" << psi.transpose() << "\n";
            throw std::runtime_error("State leaked outside of the codespace.");
        }

        //The state can be written as psi = a|0>_L +b|1>_L with |a|^2+|b|^2=1
        //so <0|psi> = a. Also, psi = cos(theta_L) |0>_L + e^{i\phi} sin(theta_L) |1>_L
        //Check if there is a factor of 2?
        
        Complex a     = ket0L.dot(psi);
        

        Complex b = ket1L.dot(psi);

        Real abs_a  = clamp(std::abs(a), Real(0.0), Real(1.0));
        thetaL[iter] = std::acos(abs_a);        

        // phi[iter]=std::arg(b) - std::arg(a);

        
        Real sin_theta = std::sin(thetaL[iter] );
        LER += sin_theta * sin_theta;  

    }


    return LER;




}








