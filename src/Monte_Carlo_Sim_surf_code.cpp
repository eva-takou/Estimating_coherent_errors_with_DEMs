#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <complex>
#include <queue>


#include "estimation_functions_surf_code.h"  //TODO: Make estimation functions for this.
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

struct DataOutcome {
    std::vector<uint8_t> bitstring;  // data qubit bitstring
    Real probability;
};

struct AncillaOutcome {
    std::vector<uint8_t> bitstring;  // ancilla qubit bitstring
    Real probability;
    std::vector<DataOutcome> data_outcomes;  // nested vector of data outcomes
};


template <typename T>
T clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}


VectorXc prepare_pre_meas_state(int d, const std::vector<std::pair<size_t, size_t>>& all_swaps, 
                                 const ArrayXc& phase_mask,
                                 const ArrayXc& ZZ_mask) { 
    
    
    
    
    int n_data = d*d;
    int n_anc  = n_data-1;
    int nQ     = n_data+n_anc; 
    
    
    VectorXc psi = Ket0(nQ);

    std::vector<int> idxs_data{0,1,2,3,4,5,6,7,8};
    std::vector<int> idxs_anc{9,10,11,12}; //X-type ancilla, measuring Z-type errors

    apply_Hadamard_on_qubits(psi,idxs_data);
    apply_Hadamard_on_qubits(psi,idxs_anc);

    //Next we want to apply the Rz operator
    apply_precomputed_Rz_mask(psi, phase_mask);
    
    //Next we want to apply the CNOTs
    apply_CNOTs_from_precomputed_swaps(all_swaps, psi);

    apply_precomputed_ZZ_mask(psi, ZZ_mask); //ZZ-errors after the CNOTs
    
    //Finally we apply the Hadamards on the ancilla qubits only
    
    //apply_fast_hadamards_on_ancilla_qubits(psi,n_data,nQ);

    apply_Hadamard_on_qubits(psi,idxs_anc);

    return psi;
}

inline std::tuple<Time,Time> reprepare_state(VectorXc &psi, int d,  const std::vector<std::pair<size_t, size_t>>& all_swaps,
                                                     const ArrayXc& phase_mask, 
                                                     const ArrayXc& ZZ_mask){ 

    const int n_data = d*d;
    const int n_anc  = n_data-1;                                                       
    
    Time time_for_Had = 0.0;
    Time time_for_CNOT = 0.0;

    std::vector<int> idxs_anc{9,10,11,12};

    // apply_fast_hadamards_on_ancilla_qubits(psi,d);
    
    apply_precomputed_Rz_mask(psi, phase_mask);
    
    auto t0 = Clock::now();

    apply_CNOTs_from_precomputed_swaps(all_swaps, psi);

    auto t1 = Clock::now();

    time_for_CNOT += Evaluate_Time(t1-t0).count();
    
    
    // t0 = std::chrono::high_resolution_clock::now();
    apply_precomputed_ZZ_mask(psi, ZZ_mask); //ZZ-errors after the CNOTs
    // t1 = std::chrono::high_resolution_clock::now();
    // std::cout << "Time for applying ZZ mask: " << std::chrono::duration<double>(t1 - t0).count() << "\n";
    

    //Apply the Hadamards on ancilla
    t0 = Clock::now();

    psi=apply_Hadamard_on_qubits(psi,idxs_anc);
    

    
    t1 = Clock::now();

    time_for_Had += Evaluate_Time(t1 - t0).count();

    return {time_for_Had,time_for_CNOT};


}




//Here we create the swap indices for the CNOTs


//    |  X |  
//    o----o----o---
//    |    |    |
//    |  Z |  X | Z
// ---o----o----o--- 
//    |    |    |
//  Z |  X | Z  |
// ---o----o----o
//         |  X |

//Ordering:
//    | 0X |  
//    0----3----6---
//    |    |    |
//    | 1Z | 2X | 3Z
// ---1----4----7--- 
//    |    |    |
// 0Z | 1X | 2Z |
// ---2----5----8
//         | 3X |

//For X we start from low right, to low left to top right to top left
//For Z we start from low right, to top right, to low left to top left

//Need: 
// CNOT_{1X,5}, CNOT_{1X,2}, CNOT_{1X,4}, CNOT_{1X,1}
// CNOT_{2X,7}, CNOT_{1X,4}, CNOT_{1X,6}, CNOT_{1X,3}
// CNOT_{0X,3}, CNOT_{0X,0}
// CNOT_{3X,8}, CNOT_{3X,5} 

// So we do:

// 1st: CNOT_{0X,3}, CNOT_{1X,5}, CNOT_{2X,7},               CNOT_{1Z,4}, CNOT_{0Z,2}, CNOT_{2Z,8}
// 2nd: CNOT_{0X,0}, CNOT_{1X,2}, CNOT_{2X,4},               CNOT_{1Z,3}, CNOT_{0Z,1}, CNOT_{2Z,7}
// 3rd:              CNOT_{1X,4}, CNOT_{2X,6},CNOT_{3X,8},   CNOT_{1Z,1},              CNOT_{2Z,5}, CNOT_{3Z,7}    
// 4th:              CNOT_{1X,1}, CNOT_{2X,3},CNOT_{3X,5},   CNOT_{1Z,0},              CNOT_{2Z,4}, CNOT_{3Z,6}

//Note for the Z-checks we need the data qubits to be the control

//Now let's order ancilla qubits as 
//    | 9  |  
//    0----3----6---
//    |    |    |
//    | 14 | 11 | 16
// ---1----4----7--- 
//    |    |    |
// 13 | 10 | 15 |
// ---2----5----8
//         | 12 |


//    | 9  |  
//    0----3----6---
//    |    |    |
//    | 14 | 11 | 16
// ---1----4----7--- 
//    |    |    |
// 13 | 10 | 15 |
// ---2----5----8
//         | 12 |


//This does the parallel zig-zag scheme (not entirely sure if I'm applying this correctly)
std::vector<std::pair<size_t, size_t>> find_CNOT_swaps_for_surface_code(){

    const int nQ=17;
    std::vector<std::pair<size_t, size_t>> all_swaps;

    // 1st: CNOT_{0X,3}, CNOT_{1X,5}, CNOT_{2X,7},               CNOT_{4,1Z}, CNOT_{2,0Z}, CNOT_{8,2Z}
    // 2nd: CNOT_{0X,0}, CNOT_{1X,2}, CNOT_{2X,4},               CNOT_{3,1Z}, CNOT_{1,0Z}, CNOT_{7,2Z}
    // 3rd:              CNOT_{1X,4}, CNOT_{2X,6},CNOT_{3X,8},   CNOT_{1,1Z},              CNOT_{5,2Z}, CNOT_{7,3Z}    
    // 4th:              CNOT_{1X,1}, CNOT_{2X,3},CNOT_{3X,5},   CNOT_{0,1Z},              CNOT_{4,2Z}, CNOT_{6,3Z}


    std::vector<int> TargetsX_1st{3,5,7}; //1st: CNOT_{0X,3}, CNOT_{1X,5}, CNOT_{2X,7},
    
    for (int i=0; i<3; ++i){
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(9+i,{TargetsX_1st[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    }

    std::vector<int> TargetsZ_1st{14, 13, 15}; //1st:  CNOT_{1Z,4}, CNOT_{0Z,2}, CNOT_{2Z,8}
    std::vector<int> controlsZ_1st{4,2,8};

    for (int i=0; i<3; ++i){
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(controlsZ_1st[i], {TargetsZ_1st[i]}, nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    }

    std::vector<int> TargetsX_2nd{0,2,4}; //2nd: CNOT_{0X,0}, CNOT_{1X,2}, CNOT_{2X,4},

    for (int i=0; i<3; ++i){
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(9+i,{TargetsX_2nd[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    }

     
    std::vector<int> controlsZ_2nd{3,1,7}; // 2nd: CNOT_{1Z,3}, CNOT_{0Z,1}, CNOT_{2Z,7}

    for (int i=0; i<3; ++i){
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(controlsZ_2nd[i], {TargetsZ_1st[i]}, nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    }

    std::vector<int> TargetsX_3rd{4,6,8};  // 3rd: CNOT_{1X,4}, CNOT_{2X,6},CNOT_{3X,8},  

    for (int i=0; i<3; ++i){
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(10+i,{TargetsX_3rd[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    }

   
    std::vector<int> TargetsZ_3rd{14,15,16}; // 3rd:  CNOT_{1Z,1},  CNOT_{2Z,5}, CNOT_{3Z,7}    
    std::vector<int> controlsZ_3rd{1,5,7}; 

    for (int i=0; i<3; ++i){
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(controlsZ_3rd[i], {TargetsZ_3rd[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    }

    
    std::vector<int> TargetsX_4th{1,3,5}; // 4th:   CNOT_{1X,1}, CNOT_{2X,3},CNOT_{3X,5},  

    for (int i=0; i<3; ++i){
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(10+i,{TargetsX_4th[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    }

    std::vector<int> TargetsZ_4th{14,15,16}; // 4th:   CNOT_{1Z,0},   CNOT_{2Z,4}, CNOT_{3Z,6}
    std::vector<int> controlsZ_4th{0,4,6}; 

    for (int i=0; i<3; ++i){
        std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(controlsZ_4th[i], {TargetsZ_4th[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    }    

    // std::vector<int> targets1{0,3};
    // std::vector<std::pair<size_t, size_t>> swaps = precompute_CNOT_swaps(9,targets1 , nQ);
    // all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());


    // std::vector<int> targets2{1,2,4,5};
    // swaps = precompute_CNOT_swaps(10,targets2 , nQ);
    // all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    
    // std::vector<int> targets3{3,4,6,7};
    // swaps = precompute_CNOT_swaps(11,targets3 , nQ);
    // all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());

    // std::vector<int> targets4{5,8};
    // swaps = precompute_CNOT_swaps(12,targets4 , nQ);
    // all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());



    return all_swaps;
}

//This is to apply ZZ errors after all gates (might not be the best, we should try other 2-qubit gate errors too.)
ArrayXc get_ZZ_phase_mask_for_surface_code(Real theta_G){
    
    const int nQ=17;
    ArrayXc ZZ_mask = VectorXc::Ones(1 << nQ);    


    // 1st: CNOT_{0X,3}, CNOT_{1X,5}, CNOT_{2X,7},               CNOT_{4,1Z}, CNOT_{2,0Z}, CNOT_{8,2Z}
    // 2nd: CNOT_{0X,0}, CNOT_{1X,2}, CNOT_{2X,4},               CNOT_{3,1Z}, CNOT_{1,0Z}, CNOT_{7,2Z}
    // 3rd:              CNOT_{1X,4}, CNOT_{2X,6},CNOT_{3X,8},   CNOT_{1,1Z},              CNOT_{5,2Z}, CNOT_{7,3Z}    
    // 4th:              CNOT_{1X,1}, CNOT_{2X,3},CNOT_{3X,5},   CNOT_{0,1Z},              CNOT_{4,2Z}, CNOT_{6,3Z}

    // 1st: CNOT_{9,3}, CNOT_{10,5}, CNOT_{11,7},               CNOT_{4,14}, CNOT_{2,13}, CNOT_{8,15}
    // 2nd: CNOT_{9,0}, CNOT_{10,2}, CNOT_{11,4},               CNOT_{3,14}, CNOT_{1,13}, CNOT_{7,15}
    // 3rd:              CNOT_{10,4}, CNOT_{11,6},CNOT_{12,8},   CNOT_{1,14},              CNOT_{5,15}, CNOT_{7,16}    
    // 4th:              CNOT_{10,1}, CNOT_{11,3},CNOT_{12,5},   CNOT_{0,14},              CNOT_{4,15}, CNOT_{6,16}


    std::vector<int> controls{9,10,11,        4,2,8,
                              9,10,11,        3,1,7,
                             10,11,12,        1,5,7,
                             10,11,12,        0,4,6}; 

    std::vector<int> targets{3,5,7,           14,13,15,
                             0,2,4,           14,13,15,
                             4,6,8,           14,15,16,
                             1,3,5,           14,15,16};

    for (int i = 0; i < controls.size(); ++i) {
        ArrayXc temp1 = compute_ZZ_phase_mask(nQ, controls[i], targets[i], theta_G);
        ZZ_mask *= temp1;
    }    

    return ZZ_mask;

}




std::tuple< std::vector<AncillaOutcome>, 
            std::vector<std::pair<size_t, size_t>> ,
            VectorXc,
            VectorXc,
            std::unordered_map<uint8_t, std::vector<size_t>>,
            ArrayXc
            > get_probs_d_rounds_Monte_Carlo(int d, int rds, int Nsamples, int Nsamples_data, Real theta_data, Real theta_anc, Real theta_G, int Reset_ancilla){


    // Fixed values/vectors
    
    int nQ      = 17;
    int n_anc   = 8;
    int n_anc_X = 4;
    int n_anc_Z = 4;
    int n_data  = 9;
    
    std::vector<int>  idxs_data(n_data);
    for (int i=0; i<n_data; ++i){ idxs_data[i]=i;}

    std::vector<int> idxs_anc_X{9,10,11,12};
    std::vector<int> idxs_anc_Z{13,14,15,16};
    std::vector<int> idxs_anc{9,10,11,12,13,14,15,16};

    std::vector<int> idxs_all(nQ);
    for (int i = 0; i < nQ; ++i) idxs_all[i] = i;

    std::vector<int> shifted_anc_inds_X(n_anc_X);
    std::vector<int> shifted_anc_inds_Z(n_anc_Z);
    std::vector<int> shifted_anc_inds(n_anc);
    
    for (int i = 0; i < n_anc_X; ++i) {
        shifted_anc_inds_X[i] = nQ - 1 - idxs_anc_X[i];
    }    

    for (int i = 0; i < n_anc_Z; ++i) {
        shifted_anc_inds_Z[i] = nQ - 1 - idxs_anc_Z[i];
    }    

    for (int i=0; i<n_anc; ++i){
        shifted_anc_inds[i] = nQ-1-idxs_anc[i];
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
    
    unsigned short int encountered_zero_norm;
    
    std::unordered_map<std::vector<uint8_t>, AncillaOutcome,VectorHash> outcome_hist;  //This means input (key) is a vector of ints, and it outputs the AncillaOutcome struct
    
    std::vector<uint8_t> outcome_of_data(n_data); 
    std::vector<uint8_t> outcome_this_rd(n_anc);
    std::vector<uint8_t> ancilla_bitstring;
    ancilla_bitstring.reserve(n_anc * rds);  
    std::unordered_map<std::vector<uint8_t>, Real, VectorHash> data_qubit_struct; //struct for the data qubit outcomes

    
    //Indices to implement the swaps
    std::vector<std::pair<size_t, size_t>> all_swaps = find_CNOT_swaps_for_surface_code();


    std::vector<Real> thetas;
    for (int i=0; i<n_data; ++i){
        thetas.push_back(theta_data);
    }   

    for (int i=0; i<n_anc; ++i){
        thetas.push_back(theta_anc);
    }
    
    ArrayXc phase_mask = precompute_Rz_phase_mask(nQ, idxs_all,  thetas);
    ArrayXc ZZ_mask    = get_ZZ_phase_mask_for_surface_code(theta_G);
    

    const VectorXc psi0    = prepare_pre_meas_state(d,  all_swaps, phase_mask, ZZ_mask);
    const Eigen::Index dim = psi0.size();    


    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, n_data);

    Time total_time_for_prepare_state_again = 0.0;
    Time total_time_for_meas_anc            = 0.0;
    Time total_time_for_meas_data           = 0.0;
    Time total_time_for_Hads                = 0.0;
    Time total_time_for_CNOTs               = 0.0;
    Time total_time_for_Hads_on_data        = 0.0;

    std::unordered_map<uint8_t, std::vector<size_t>> kept_indices_cache; 

    VectorXc psi;    
    psi.resize(psi0.size());

    VectorXc psi_data(1 << n_data );
    VectorXc psi_new(1<< nQ);
    VectorXc psi_plus_anc_X = plus_state(n_anc_X); //Only on X-type..
    VectorXc psi_plus_anc_Z = Ket0(n_anc_Z);
    VectorXc psi_anc = Eigen::kroneckerProduct(psi_plus_anc_X, psi_plus_anc_Z).eval();

    //TODO: Cache the cumulative distributions for next rounds
    //Need a map for particular rd and outcome.
    std::vector<Real> cumsum_data(1<< n_data);

    std::vector<Real> cdf_buffer_total(1<<nQ);


    cumSum_from_state_vector(psi0,cdf_buffer_total); //Use this for 1st round of measurements

    VectorXc psi_buffer(psi0.size());

    

    for (int sample = 0; sample < Nsamples; ++sample) {

        std::memcpy(psi.data(), psi0.data(), sizeof(Complex) * psi0.size());
        ancilla_bitstring.clear(); //Reset

        for (int r = 0; r < rds; ++r) {

            //Psi gets projected to the outcome
            std::chrono::time_point<std::chrono::high_resolution_clock> t0_meas_anc;
            std::chrono::time_point<std::chrono::high_resolution_clock> t1_meas_anc;

            if (r==0){
                t0_meas_anc = Clock::now();
                outcome_this_rd = measure_all_ancilla_first_rd(nQ, n_anc,  idxs_anc,  psi, kept_indices_cache, 
                                                               shifted_anc_inds, data_positions, cdf_buffer_total,psi_buffer);

                                
                t1_meas_anc = Clock::now();
            }
            else{
                t0_meas_anc = Clock::now();
                outcome_this_rd = measure_all_ancilla_NEW(nQ,n_anc,idxs_anc,psi,kept_indices_cache, shifted_anc_inds, data_positions,psi_buffer);
                
                t1_meas_anc  = Clock::now();
            }

            
            total_time_for_meas_anc       += Evaluate_Time(t1_meas_anc - t0_meas_anc).count();

            if (Reset_ancilla==1){
                
                apply_X_on_qubits(psi, outcome_this_rd, n_data, dim, nQ); //"Reset" the ancilla (more efficient than tracing out and starting again in |0>)

                
            }

            // Store outcome
            ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());

            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {

                auto t0_prep_state_again = Clock::now();
                Time time_for_Had;
                Time time_for_CNOT;

                //TODO: Add if statements.
                //Discard the ancilla qubits -- assuming that we reset also
                
                psi_data.setZero();
                for (const auto& [i_full, i_reduced] : index_map)
                    psi_data[i_reduced] = psi[i_full];           


                psi = Eigen::kroneckerProduct(psi_data, psi_anc).eval();

                psi.normalize();
                    
                std::tie(time_for_Had,time_for_CNOT) = reprepare_state(psi, d,  all_swaps, phase_mask,ZZ_mask); 

                
                total_time_for_Hads+=time_for_Had;
                total_time_for_CNOTs+=time_for_CNOT;

                auto t1_prep_state_again = Clock::now();
                total_time_for_prepare_state_again += Evaluate_Time(t1_prep_state_again - t0_prep_state_again).count();
            
            }
            
        }


        
        // Accumulate counts
        auto& anc_outcome = outcome_hist[ancilla_bitstring];

        if (anc_outcome.bitstring.empty())
            anc_outcome.bitstring = ancilla_bitstring;
        anc_outcome.probability += 1.0;

        // If data outcomes not computed yet, compute once per unique ancilla bitstring --  because of the top Monte Carlo,
        // the same ancilla outcome can occur multiple times, so compute only once the data qubit probs.
        
        if (anc_outcome.data_outcomes.empty()) {

            // VectorXcd psi_data;
            if (Reset_ancilla==1){

                psi_data.setZero();
                for (const auto& [i_full, i_reduced] : index_map)
                    psi_data[i_reduced] = psi[i_full];           
                
            }
            else{
                psi_data=discard_measured_qubits(psi, idxs_data, idxs_anc, outcome_this_rd, nQ); //Need to discard based on measurement outcomes
            }

            psi_data.normalize();
            
            auto t0 = Clock::now();
            apply_Hadamard_on_all_qubits(psi_data);
            auto t1 = Clock::now();
            total_time_for_Hads_on_data       += Evaluate_Time(t1 - t0).count();

            data_qubit_struct.clear();
            int N_valid_samples = Nsamples_data;
            
            //We should build the cumulative distribution only once here and sample multiple times over it.
            //I can also maybe store the vectors..
            cumSum_from_state_vector(psi_data, cumsum_data);

            for (int sample2=0; sample2<Nsamples_data; ++sample2){
                
                
                auto t0_meas_data = Clock::now();

                                                                                        
                std::tie(outcome_of_data,encountered_zero_norm) = measure_all_data(n_data,shifted_data_bits_from_d,cumsum_data); 

                auto t1_meas_data = Clock::now();

                total_time_for_meas_data += Evaluate_Time(t1_meas_data - t0_meas_data).count();

                if (encountered_zero_norm==1){
                    N_valid_samples = N_valid_samples-1;
                    continue;
                }
                
                //Accumulate counts
                data_qubit_struct[outcome_of_data] +=1.0;

            }

            const Real inv_N_valid_samples  = 1.0 / static_cast<Real>(N_valid_samples);
            auto& data_vec =  anc_outcome.data_outcomes; //reference to anc_outcome.data_outcomes

            data_vec.reserve(data_qubit_struct.size());
            for (const auto& [key, value] : data_qubit_struct) {
                data_vec.push_back({key, value * inv_N_valid_samples});
            }            

        }
    }

    std::cout << "Took: " << total_time_for_prepare_state_again << " sec, in total for prepare state again. \n";
    std::cout << "Took: " << total_time_for_meas_anc << " sec, in total for measuring ancilla. \n";
    std::cout << "Took: " << total_time_for_meas_data << " sec, in total for measuring data. \n";
    std::cout << "Took: " << total_time_for_Hads << " sec, for all Hads. \n";
    std::cout << "Took: " << total_time_for_CNOTs << " sec, for all CNOTs. \n";
    std::cout << "Took: " << total_time_for_Hads_on_data << " sec, for Hads on data qubits. \n";

    
    // Normalize probabilities of ancilla qubits

    const Real inv_Nsamples  = 1.0 / static_cast<Real>(Nsamples);

    std::vector<AncillaOutcome> all_ancilla_outcomes;
    all_ancilla_outcomes.reserve(outcome_hist.size());
    for (auto& [bits, outcome] : outcome_hist) {
        outcome.probability *= inv_Nsamples;
        all_ancilla_outcomes.push_back(std::move(outcome));
    }


    return std::make_tuple(all_ancilla_outcomes,all_swaps,phase_mask, psi0,kept_indices_cache,ZZ_mask);
}



//TODO: WRITE A NEW ESTIMATION FUNCTION FOR THE NOISE
//TODO: CHECK IF THIS: project_on_indices_inplace IS CORRECT



inline std::vector<uint8_t> get_outcome_per_rd(const std::vector<uint8_t>& anc_outcome, int n_anc, int rd){

    auto start_it = anc_outcome.begin() + rd * n_anc;
    return std::vector<uint8_t>(start_it, start_it + n_anc);    

}



//These are OK
std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>> get_parity_check_matrices(){

    std::vector<std::vector<int>> Hz(4, std::vector<int>(9, 0));

    Hz[0][1]=1;
    Hz[0][2]=1;

    Hz[1][0]=1;
    Hz[1][1]=1;
    Hz[1][3]=1;
    Hz[1][4]=1;

    Hz[2][4]=1;
    Hz[2][5]=1;
    Hz[2][7]=1;
    Hz[2][8]=1;

    Hz[3][6]=1;
    Hz[3][7]=1;

    std::vector<std::vector<int>> Hx(4, std::vector<int>(9, 0));

    Hx[0][0]=1;
    Hx[0][3]=1;

    Hx[1][1]=1;
    Hx[1][2]=1;
    Hx[1][4]=1;
    Hx[1][5]=1;

    Hx[2][3]=1;
    Hx[2][4]=1;
    Hx[2][6]=1;
    Hx[2][7]=1;

    Hx[3][5]=1;
    Hx[3][8]=1;

 
    return {Hx,Hz};


}


std::tuple<std::vector<std::vector<int>>, 
           std::vector<std::vector<int>>, 
           std::vector<Real>,std::vector<Real>,
           std::vector<Real>,std::vector<Real>> sample_outcomes_MC_surface_code(int d, int rds, int Nsamples, int Nsamples_data, int nsims,
                                                                             Real theta_data, Real theta_anc, Real theta_G, Real q_readout,
                                                                             int Reset_ancilla, 
                                                                             int include_stab_reconstruction){

    // Fixed values/vectors
    int nQ      = 17;
    int n_anc   = 8;
    int n_anc_X = 4;
    int n_anc_Z = 4;
    int n_data  = 9;
    
    std::vector<int>  idxs_data(n_data);
    for (int i=0; i<n_data; ++i){ idxs_data[i]=i;}

    std::vector<int> idxs_anc_X{9,10,11,12};
    std::vector<int> idxs_anc_Z{13,14,15,16};
    std::vector<int> idxs_anc{9,10,11,12,13,14,15,16};

    std::vector<int> idxs_all(nQ);
    for (int i = 0; i < nQ; ++i) idxs_all[i] = i;
    
    std::vector<AncillaOutcome> all_ancilla_outcomes;
    std::vector<std::pair<size_t, size_t>> all_swaps;
    ArrayXc phase_mask;
    VectorXc psi0;
    std::unordered_map<uint8_t, std::vector<size_t>> kept_indices_cache;
    ArrayXc ZZ_mask;

    auto t0 = Clock::now();
    
    std::tie(all_ancilla_outcomes, all_swaps, 
             phase_mask, 
            psi0, kept_indices_cache, ZZ_mask) = get_probs_d_rounds_Monte_Carlo(d,rds,Nsamples, Nsamples_data, theta_data, theta_anc, theta_G, Reset_ancilla);

    auto t1 = Clock::now();
    std::cout << "Took: " << Evaluate_Time(t1 - t0).count() << " sec, to get probs from Monte Carlo sim. \n";

    const Eigen::Index dim = psi0.size();    

    //TODO: THIS WILL NOT WORK NOW.

    // std::vector<double> p_space;
    // std::vector<double> p_time;
    // std::vector<double> p_diag;

    // t0 = std::chrono::high_resolution_clock::now();  
    // std::tie(p_space,p_time,p_diag) = estimate_probs_from_syndrome(d, rds, q_readout, Reset_ancilla, include_stab_reconstruction, all_ancilla_outcomes, nsims_for_est);
    // t1 = std::chrono::high_resolution_clock::now();
    // std::cout << "Took: " << std::chrono::duration<double>(t1 - t0).count() << " sec, to run the whole estimate probs function. \n";

    // print_vector(p_space, "p_space");
    // print_vector(p_time, "p_time");
    // print_vector(p_diag, "p_diag");


    std::vector<Real> prob_vec_anc;
    std::vector<std::vector<uint8_t>> anc_bit_pattern;
    std::vector<std::vector<DataOutcome>> all_data_outcomes;

    size_t N = all_ancilla_outcomes.size();
    anc_bit_pattern.reserve(N);
    prob_vec_anc.reserve(N);
    all_data_outcomes.reserve(N);    

    for (const auto& outcome : all_ancilla_outcomes) {
        anc_bit_pattern.push_back(std::move(outcome.bitstring));  //()
        prob_vec_anc.push_back(std::move(outcome.probability));
        all_data_outcomes.push_back(std::move(outcome.data_outcomes));
    }

    std::vector<Real> cumsum = cumSum(prob_vec_anc);
    
    //For ancilla
    std::vector<std::vector<uint8_t>> anc_patterns_sampled(nsims);
    std::vector<std::vector<uint8_t>> data_patterns_sampled(nsims);

    //For data
    std::vector<Real> probs_data;
    std::vector<const std::vector<uint8_t>*> data_bit_pointers;
    std::vector<Real> cumsum_data;

    //For decoding
    //NEED TO UPDATE THIS
    // std::vector<std::vector<int>> Hx = Hx_rep_code(d);

    
    std::vector<std::vector<uint8_t>> batch_X;
    std::vector<std::vector<uint8_t>> batch_Z;

    
    batch_X.resize(nsims);
    batch_Z.resize(nsims);

    
    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, n_data);
    
    
    
    t0 = Clock::now();
    //Sample based on probabilities of ancilla_outcomes
    std::vector<uint8_t> anc_outcome;
    int rds_effective = rds + (include_stab_reconstruction ? 1 : 0);
    anc_outcome.reserve(n_anc * rds_effective);


    //key-value pair
    std::unordered_map<std::vector<uint8_t>,VectorXc, VectorHash> psi_stored;

    VectorXc psi;
    VectorXc psi_buffer(psi0.size());

    VectorXc psi_plus_anc_X = plus_state(n_anc_X); //Only on X-type..
    VectorXc psi_anc_Z = Ket0(n_anc_Z);
    VectorXc psi_anc = Eigen::kroneckerProduct(psi_plus_anc_X, psi_anc_Z).eval();

    
    for (int sim=0; sim<nsims; ++sim){

        // double r_anc = std::generate_canonical<double, 53>(gen); //dis(gen);        
        Real r_anc = std::generate_canonical<Real, mantissa_bits>(gen3);
        auto it = std::lower_bound(cumsum.begin(), cumsum.end(), r_anc);
        int idx = std::distance(cumsum.begin(), it); //Find indx where r_anc<C(idx)

        anc_outcome.clear();  // keep capacity
        anc_outcome.insert(anc_outcome.end(), anc_bit_pattern[idx].begin(), anc_bit_pattern[idx].end());        

        anc_patterns_sampled[sim] = anc_outcome; 

        psi = psi0;
        std::vector<uint8_t> outcome_this_rd;
        unsigned short int encountered_zero_norm;

        VectorXc psi_data(1 << n_data);
        VectorXc psi_new(1<< nQ);

        for (int rd=0; rd<rds; ++rd){

            outcome_this_rd       = get_outcome_per_rd(anc_outcome, n_anc, rd);
            encountered_zero_norm = project_on_indices_inplace(psi, kept_indices_cache[pack_outcome_inline(outcome_this_rd)], psi_buffer);

            if (Reset_ancilla==1){apply_X_on_qubits(psi , outcome_this_rd, n_data, dim, nQ);}

            if (rd != rds - 1) { 

                psi_data.setZero();
                for (const auto& [i_full, i_reduced] : index_map)
                    psi_data[i_reduced] = psi[i_full];           

                psi = Eigen::kroneckerProduct(psi_data, psi_anc).eval(); // |psi_data> \otimes |+>^{n_anc/2} \otimes |0>^{n_anc/2}


                std::tie(t0H,t0CNOT)=reprepare_state(psi, d,  all_swaps,phase_mask, ZZ_mask); }
            
        }

        

        //Store this for the correction part
        VectorXc psi_reduced;

        if (Reset_ancilla==1){  psi_reduced = VectorXc::Zero(1 << idxs_data.size());

            for (const auto& [i_full, i_reduced] : index_map) 
                {psi_reduced[i_reduced] = psi[i_full];}        
            }
        else{ psi_reduced = discard_measured_qubits(psi, idxs_data, idxs_anc, outcome_this_rd, nQ); }
        
        psi_reduced.normalize(); 

        


        psi_stored.insert({anc_outcome,psi_reduced}); //Will only insert if the key doesnt exist


        //Given the ancilla outcome, get the probabilities of possible data bitstrings and the possible bitstrings
        const auto& data_qubit_struct = all_data_outcomes[idx];
        probs_data.clear();
        data_bit_pointers.clear();
        for (const auto& data_outcome_temp : data_qubit_struct) {
            probs_data.emplace_back(data_outcome_temp.probability); 
            data_bit_pointers.emplace_back(&data_outcome_temp.bitstring);
        }

        //Now sample data qubit outcomes:
        cumSum_inplace(probs_data,cumsum_data);
        
        Real r_data   = std::generate_canonical<Real, mantissa_bits>(gen3);     
        auto it_data  = std::lower_bound(cumsum_data.begin(), cumsum_data.end(), r_data);
        int idx_data  = std::distance(cumsum_data.begin(), it_data); 

        const auto& data_outcome = *data_bit_pointers[idx_data];
        data_patterns_sampled[sim] = data_outcome;

        if (include_stab_reconstruction==1){

            anc_outcome.push_back( data_outcome[0] ^ data_outcome[3]);
            anc_outcome.push_back( data_outcome[1] ^ data_outcome[2] ^ data_outcome[4] ^ data_outcome[5]);
            anc_outcome.push_back( data_outcome[3] ^ data_outcome[4] ^ data_outcome[6] ^ data_outcome[7]);
            anc_outcome.push_back( data_outcome[5] ^ data_outcome[8]);

            anc_outcome.push_back( data_outcome[1] ^ data_outcome[2]);
            anc_outcome.push_back( data_outcome[0] ^ data_outcome[1] ^ data_outcome[3] ^ data_outcome[4]);
            anc_outcome.push_back( data_outcome[4] ^ data_outcome[5] ^ data_outcome[7] ^ data_outcome[8]);
            anc_outcome.push_back( data_outcome[6] ^ data_outcome[7]);
            
        }

        form_defects(anc_outcome,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);

        
        //Now, we want to pick the X dets and Z dets per number of rounds

        std::vector<uint8_t> anc_outcome_X(n_anc_X * rds_effective);
        std::vector<uint8_t> anc_outcome_Z(n_anc_Z * rds_effective);

        for (int rd=0; rd<rds_effective; ++rd){

            for (int l=0; l<n_anc_X; ++l){

                anc_outcome_X[l+ rd * n_anc_X] = anc_outcome[l + rd * n_anc];
                anc_outcome_Z[l+ rd * n_anc_Z] = anc_outcome[l+n_anc_X + rd * n_anc];

            }

        }

        batch_X[sim] = std::move(anc_outcome_X);
        batch_Z[sim] = std::move(anc_outcome_Z);
        
    }


    const int dim_data = 1 << n_data;
    

    std::vector<std::vector<int>> Hz;
    std::vector<std::vector<int>> Hx;

    std::tie(Hx,Hz)=get_parity_check_matrices();


    t1 = Clock::now();

    std::cout << "Took: " << Evaluate_Time(t1 - t0).count() << " sec, to run sampling loop from sample_outcomes_MC_final. \n";


    t0 = Clock::now();
    // auto corrections = decode_with_pymatching_create_graph_V2(Hx, p_space, p_time, p_diag, batch,rds, include_stab_reconstruction);
    
    

    //For the corrections_X we apply Rz operations
    auto corrections_X = decode_batch_with_pymatching(Hx, batch_X, rds);  //use this only if we have include_stab_reconstruction=0 and up to phenomenological noise
    auto corrections_Z = decode_batch_with_pymatching(Hz, batch_Z, rds);  //use this only if we have include_stab_reconstruction=0 and up to phenomenological noise
    
    std::cout << "Full contents of corrections_Z:" << std::endl;
    for (size_t i = 0; i < corrections_Z.size(); ++i) {
        std::cout << "corrections[" << i << "] = [";
        for (size_t j = 0; j < corrections_Z[i].size(); ++j) {
            std::cout << corrections_Z[i][j];
            if (j + 1 < corrections_Z[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }     

    std::cout << "Full contents of corrections_X:" << std::endl;
    for (size_t i = 0; i < corrections_X.size(); ++i) {
        std::cout << "corrections[" << i << "] = [";
        for (size_t j = 0; j < corrections_X[i].size(); ++j) {
            std::cout << corrections_X[i][j];
            if (j + 1 < corrections_X[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }     

    t1 = Clock::now();

    std::cout << "Took: " << Evaluate_Time(t1 - t0).count() << " sec, to decode. \n";

    if (corrections_X[0].size()!=n_data){
        std::cout << "Full contents of corrections:" << std::endl;
        for (size_t i = 0; i < corrections_X.size(); ++i) {
            std::cout << "corrections[" << i << "] = [";
            for (size_t j = 0; j < corrections_X[i].size(); ++j) {
                std::cout << corrections_X[i][j];
                if (j + 1 < corrections_X[i].size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }        
        throw std::runtime_error("Incorrect shape: corrections.size() != d");
    }

    
    VectorXc ket0L = plus_state(n_data); 
    VectorXc ket1L = minus_state(n_data);

    //The XL operator can be chosen as: X0*X1*X2, or X3*X4*X5
    MatrixXc X = Ket0(1) * Ket1(1).adjoint() + Ket1(1) * Ket0(1).adjoint();
    
    MatrixXc XL_temp = Eigen::kroneckerProduct(X,X).eval();
    MatrixXc XL = Eigen::kroneckerProduct(XL_temp,X).eval();

    MatrixXc Id = Eigen::MatrixXd::Identity(1<<6,1<<6); 
    MatrixXc Id3 = Eigen::MatrixXd::Identity(1<<3,1<<3); 

    MatrixXc temp = Eigen::kroneckerProduct(Id3,XL).eval();
    
    
    MatrixXc Proj_Alt = Eigen::kroneckerProduct(temp,Id3).eval();
    MatrixXc Proj  = Eigen::kroneckerProduct(XL,Id).eval();



    // MatrixXc Proj = ket0L * ket0L.adjoint() + ket1L * ket1L.adjoint();
    
    std::vector<Real> phi(nsims,0.0);
    std::vector<Real> thetaL(nsims,0.0);
    std::vector<Real> infidelity(nsims,0.0);
    std::vector<Real> leakage(nsims,0.0);

    for (int sim=0; sim<nsims; ++sim){

       
        VectorXc psi =  psi_stored[anc_patterns_sampled[sim]];

        std::cout<< "size of psi_stored:" << psi.size() << "\n";
        

        std::vector<int> qubits_to_correct;
        for (int m=0; m<n_data; ++m){
            if (corrections_X[sim][m]==1){
                qubits_to_correct.push_back(m);
            }
        }

        //Recovery : Z operations -- up to global phases

        if (!qubits_to_correct.empty()){
            apply_Rz_on_qubits_inplace(psi, qubits_to_correct, PI/2);
        }


        std::vector<uint8_t> corrections_Z_uint8;

        for (int val : corrections_Z[sim]) {
            corrections_Z_uint8.push_back(static_cast<uint8_t>(val));
        }        
        
        apply_X_on_qubits(psi,corrections_Z_uint8, 0, dim_data, n_data);


        Complex overlap = psi.adjoint() * Proj * psi; 

        Complex overlap2 = psi.adjoint() * Proj_Alt * psi; 

        std::cout << "overlap method 1:" << overlap.real() << "\n";
        std::cout << "overlap method 2:" << overlap2.real() << "\n";

        if (overlap.imag()>1e-10){
            throw std::runtime_error("Imaginary part in leakage > 1e-10");
        }

        leakage[sim]=1.0 - overlap.real();
        
        psi = Proj * psi; //Project into the subspace
        psi.normalize();

        Complex a     = ket0L.dot(psi);
        Real fidelity = std::norm(a);  
        
        infidelity[sim] = 1.0 - fidelity;        
        
        //Calculate leakage, infidelity w/ target state, logical rotation angle.

        Complex b = ket1L.dot(psi);

        Real abs_a  = clamp(std::abs(a), Real(0.0), Real(1.0));
        thetaL[sim] = safe_acos(abs_a);        

        phi[sim]=std::arg(b) - std::arg(a);

    }


    // return std::make_tuple(anc_patterns_sampled,data_patterns_sampled, phi, thetaL, infidelity,leakage);


    // Cast anc_patterns_sampled and data_patterns_sampled to vector<vector<int>> for Python
    std::vector<std::vector<int>> anc_patterns_sampled_int(anc_patterns_sampled.size());
    std::vector<std::vector<int>> data_patterns_sampled_int(data_patterns_sampled.size());

    for (size_t i = 0; i < anc_patterns_sampled.size(); ++i) {
        anc_patterns_sampled_int[i].assign(anc_patterns_sampled[i].begin(), anc_patterns_sampled[i].end());
    }
    for (size_t i = 0; i < data_patterns_sampled.size(); ++i) {
        data_patterns_sampled_int[i].assign(data_patterns_sampled[i].begin(), data_patterns_sampled[i].end());
    }

    return std::make_tuple(
        anc_patterns_sampled_int,
        data_patterns_sampled_int,
        phi,
        thetaL,
        infidelity,
        leakage
    );




}


std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>> get_total_pcm(){

    std::vector<std::vector<int>> Hx(8, std::vector<int>(9, 0));
    //X part
    H[0][0]=1;
    H[0][3]=1;

    H[1][1]=1;
    H[1][2]=1;
    H[1][4]=1;
    H[1][5]=1;

    H[2][3]=1;
    H[2][4]=1;
    H[2][6]=1;
    H[2][7]=1;

    H[3][5]=1;
    H[3][8]=1;

    //Z part
    
    H[4][1]=1;
    H[4][2]=1;

    H[5][0]=1;
    H[5][1]=1;
    H[5][3]=1;
    H[5][4]=1;

    H[6][4]=1;
    H[6][5]=1;
    H[6][7]=1;
    H[6][8]=1;

    H[7][6]=1;
    H[7][7]=1;


    return {H};


}



Real get_LER_from_uniform_DEM_phenom_level(int d, int rds, int ITERS, Real theta_data, Real theta_anc, Real theta_G, Real q_readout,  bool Reset_ancilla){
   
    // Fixed values/vectors

    const int n_anc  = d*d-1;
    int n_anc_X = 4;
    int n_anc_Z = 4;
    const int n_data = d*d;    
    const int nQ  = n_data+n_anc;

    
    std::vector<int>  idxs_data(n_data);
    for (int i=0; i<n_data; ++i){ idxs_data[i]=i;}

    std::vector<int> idxs_anc_X{9,10,11,12};
    std::vector<int> idxs_anc_Z{13,14,15,16};
    std::vector<int> idxs_anc{9,10,11,12,13,14,15,16};

    std::vector<int> idxs_all(nQ);
    for (int i = 0; i < nQ; ++i) idxs_all[i] = i;

    std::vector<int> shifted_anc_inds_X(n_anc_X);
    std::vector<int> shifted_anc_inds_Z(n_anc_Z);
    std::vector<int> shifted_anc_inds(n_anc);
    
    for (int i = 0; i < n_anc_X; ++i) {
        shifted_anc_inds_X[i] = nQ - 1 - idxs_anc_X[i];
    }    

    for (int i = 0; i < n_anc_Z; ++i) {
        shifted_anc_inds_Z[i] = nQ - 1 - idxs_anc_Z[i];
    }    

    for (int i=0; i<n_anc; ++i){
        shifted_anc_inds[i] = nQ-1-idxs_anc[i];
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

    VectorXc psi_data(1 << n_data);
    std::vector<Real> cumsum_data(1<<n_data);
    std::vector<Real> cdf_buffer_total(1<<nQ);

    cumSum_from_state_vector(psi0,cdf_buffer_total); //Use this for 1st round of measurements

    VectorXc psi_buffer(psi0.size());

    std::vector<std::vector<uint8_t>> all_data_outcomes;
    all_data_outcomes.resize(ITERS);

    
    std::vector<std::vector<int>> Hx = get_total_pcm(); 
    std::vector<std::vector<uint8_t>> batch;
    batch.resize(ITERS);

    VectorXc psi_plus_anc_X = plus_state(n_anc_X); 
    VectorXc psi_anc_Z = Ket0(n_anc_Z);
    VectorXc psi_anc = Eigen::kroneckerProduct(psi_plus_anc_X, psi_anc_Z).eval();


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

                apply_X_on_qubits(psi, outcome_this_rd, n_data, dim, nQ); //"Reset" the ancilla (more efficient than tracing out and starting again in |0>)
            }

            // Store outcome
            ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());

            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {


                psi_data.setZero();
                for (const auto& [i_full, i_reduced] : index_map)
                    psi_data[i_reduced] = psi[i_full];           


                psi = Eigen::kroneckerProduct(psi_data, psi_anc).eval();

                psi.normalize();
                    
                std::tie(time_for_Had,time_for_CNOT) = reprepare_state(psi, d,  all_swaps, phase_mask,ZZ_mask); 

            
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
    std::vector<Real> p_space(rds_effective * n_data, 0.1); 
    std::vector<Real> p_time(rds * n_anc, 0.1);
    std::vector<Real> p_diag(rds * (n_anc-1), 0.0); 
    
    auto corrections = decode_with_pymatching_create_graph(Htot, p_space, p_time, p_diag, batch, rds, include_stab_reconstruction);
    
    Real LER_sum = 0.0;
    for(int iter=0; iter<ITERS; ++iter){
        LER_sum += logical_XL_flipped(all_data_outcomes[iter], corrections[iter]) ? 1.0 : 0.0;
    }

    Real LER = LER_sum / ITERS;    
   

    return LER;
}