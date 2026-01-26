#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <complex>
#include <queue>

#include "Measurements.h"
#include "Kets.h"
#include "Unitary_Ops.h"
#include "call_to_pymatching.h"
#include "Stochastic_Ops.h"
#include "utils.h"
#include "utils_sc.h"
#include "estimation_functions_surf_code.h"
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


template <typename T>
T clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}


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

std::vector<std::vector<int>> get_Hx_sc(){

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

 
    return Hx;


}


std::vector<std::vector<int>> get_total_pcm(){


    std::vector<std::vector<int>> H(8, std::vector<int>(9, 0));
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

// d=3 rotated surface code
//    |  X |  
//    o----o----o---
//    |    |    |
//    |  Z |  X | Z
// ---o----o----o--- 
//    |    |    |
//  Z |  X | Z  |
// ---o----o----o
//         |  X |
// Ordering:
//    | 0X |  
//    0----3----6---
//    |    |    |
//    | 1Z | 2X | 3Z
// ---1----4----7--- 
//    |    |    |
// 0Z | 1X | 2Z |
// ---2----5----8
//         | 3X |
// Ordering of data + ancilla
//    | 9  |  
//    0----3----6---
//    |    |    |
//    | 14 | 11 | 16
// ---1----4----7--- 
//    |    |    |
// 13 | 10 | 15 |
// ---2----5----8
//         | 12 |

std::vector<std::pair<size_t, size_t>> get_CNOT_swaps_for_surface_code(){

    //Following schedule pattern from this paper: https://arxiv.org/pdf/2511.06758
    //This is NW, NE, SW, SE
    //Note X-type ancilla are control qubits, Z-type ancilla are target qubits, since we do X-memory.

    std::vector<std::pair<size_t, size_t>> all_swaps;
    const int nQ=17;

    std::vector<int> X{9,10,11,12};
    std::vector<int> Z{13,14,15,16};

    std::vector<int> controls{X[1],X[2],X[3],0,4,6};
    std::vector<int> targets{1,3,5,Z[1],Z[2],Z[3]};

    for (int i =0; i<controls.size(); ++i){

        auto swaps = precompute_CNOT_swaps(controls[i],{targets[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());

    }

    std::vector<int> controls2{X[1],X[2],X[3],1,5,7};
    std::vector<int> targets2{4,6,8,Z[1],Z[2],Z[3]};

    for (int i =0; i<controls2.size(); ++i){

        auto swaps = precompute_CNOT_swaps(controls2[i],{targets2[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());

    }    


    std::vector<int> controls3{X[0],X[1],X[2],1,3,7};
    std::vector<int> targets3{0,2,4,Z[0],Z[1],Z[2]};

    for (int i =0; i<controls3.size(); ++i){

        auto swaps = precompute_CNOT_swaps(controls3[i],{targets3[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());

    }        


    std::vector<int> controls4{X[0],X[1],X[2],2,4,8};
    std::vector<int> targets4{3,5,7,Z[0],Z[1],Z[2]};

    for (int i =0; i<controls4.size(); ++i){

        auto swaps = precompute_CNOT_swaps(controls4[i],{targets4[i]} , nQ);
        all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());

    }            

    return all_swaps;
}

void apply_data_correction(int nQ, VectorXc& psi0, const std::vector<uint8_t>& outcome_this_rd){
    //Apply X corrections on data qubits of d=3 rotated surface code, so that we bring the state
    //into the +1 eigenspace for the |+>_L logical state. We do this to avoid tracking a Pauli frame.

    //Inputs:
    //nQ: total number of data + ancilla qubits
    //psi0: the state after perfect measurement of the X and Z checks
    //outcome_this_rd: the ancilla qubits outcome (first 4 are X checks all 0, last 4 are Z checks)

    //Output (inplace): the updated state after corrections on data qubits.

    int offset = 0; //data qubit offset is 0
    const Eigen::Index dim = psi0.size();    

    //Stabilizers are Z1Z2, Z0Z1Z3Z4, Z4Z5Z7Z8, Z6Z7

    std::vector<uint8_t> s0001 = {0,0,0,0, 0,0,0,1}; //Z6Z7     -> pick 6 
    std::vector<uint8_t> s0010 = {0,0,0,0, 0,0,1,0}; //Z4Z5Z7Z8 -> pick 8 
    std::vector<uint8_t> s0100 = {0,0,0,0, 0,1,0,0}; //Z0Z1Z3Z4 -> pick 0 
    std::vector<uint8_t> s1000 = {0,0,0,0, 1,0,0,0}; //Z1Z2     -> pick 2 

    std::vector<uint8_t> s0011 = {0,0,0,0, 0,0,1,1}; //Z4Z5Z7Z8 & Z6Z7      -> pick 7 
    std::vector<uint8_t> s0101 = {0,0,0,0, 0,1,0,1}; //Z0Z1Z3Z4 & Z6Z7      -> pick 0 and 6 
    std::vector<uint8_t> s1001 = {0,0,0,0, 1,0,0,1}; //Z1Z2 & Z6Z7          -> pick 2 and 6 
    std::vector<uint8_t> s0110 = {0,0,0,0, 0,1,1,0}; //Z0Z1Z3Z4 & Z4Z5Z7Z8  -> pick 4 
    std::vector<uint8_t> s1010 = {0,0,0,0, 1,0,1,0}; //Z1Z2 & Z4Z5Z7Z8      -> pick 2 and 8 
    std::vector<uint8_t> s1100 = {0,0,0,0, 1,1,0,0}; //Z1Z2 & Z0Z1Z3Z4      -> pick 1 

    std::vector<uint8_t> s0111 = {0,0,0,0, 0,1,1,1}; //Z0Z1Z3Z4 & Z4Z5Z7Z8 & Z6Z7  -> 4 and 6 
    std::vector<uint8_t> s1101 = {0,0,0,0, 1,1,0,1}; //Z1Z2 & Z0Z1Z3Z4 & Z6Z7      -> 1 and 6 
    std::vector<uint8_t> s1011 = {0,0,0,0, 1,0,1,1}; //Z1Z2 & Z4Z5Z7Z8 & Z6Z7      -> 2 and 7 
    std::vector<uint8_t> s1110 = {0,0,0,0, 1,1,1,0}; //Z1Z2 & Z0Z1Z3Z4 & Z4Z5Z7Z8  -> 2 and 4 

    std::vector<uint8_t> s1111 = {0,0,0,0, 1,1,1,1}; //Z1Z2 & Z0Z1Z3Z4 & Z4Z5Z7Z8 & Z6Z7 -> 1 and 7 

    if (outcome_this_rd == s0001) { //Qubit 6

        apply_X_on_qubits(psi0, {0,0,0,0,0,0,1}, offset, dim, nQ); 

    }
    else if (outcome_this_rd == s0010) { //Qubit 8
        apply_X_on_qubits(psi0, {0,0,0,0,0,0,0,0,1}, offset, dim, nQ); 
    }
    else if (outcome_this_rd == s0100) { //Qubit 0
        apply_X_on_qubits(psi0, {1}, offset, dim, nQ); 
    }
    else if (outcome_this_rd == s1000) { //Qubit 2
        apply_X_on_qubits(psi0, {0,0,1}, offset, dim, nQ); 
    }
    else if (outcome_this_rd == s1000) { //Qubit 2
        apply_X_on_qubits(psi0, {0,0,1}, offset, dim, nQ); 
    }
    else if (outcome_this_rd == s0011) { //Qubit 7
        apply_X_on_qubits(psi0, {0,0,0,0,0,0,0,1}, offset, dim, nQ); 

    }
    else if (outcome_this_rd == s0101) { //Qubit 0 and 6
        apply_X_on_qubits(psi0, {1,0,0,0,0,0,1}, offset, dim, nQ); 

    }
    else if (outcome_this_rd == s1001) { //Qubit 2 and 6
        apply_X_on_qubits(psi0, {0,0,1,0,0,0,1}, offset, dim, nQ); 

    }
    else if (outcome_this_rd == s0110) { //Qubit 4
        apply_X_on_qubits(psi0, {0,0,0,0,1}, offset, dim, nQ); 

    }
    else if (outcome_this_rd == s1010) { //Qubit 2 and 8
        apply_X_on_qubits(psi0, {0,0,1,0,0,0,0,0,1}, offset, dim, nQ); 

    }
    else if (outcome_this_rd == s1100) { //Qubit 1
        apply_X_on_qubits(psi0, {0,1}, offset, dim, nQ); 

    }    
    else if (outcome_this_rd == s0111) { //Qubit 4 and 6
        apply_X_on_qubits(psi0, {0,0,0,0,1,0,1}, offset, dim, nQ); 

    }    
    else if (outcome_this_rd == s1101) { //Qubit 1 and 6
        apply_X_on_qubits(psi0, {0,1,0,0,0,0,1}, offset, dim, nQ); 

    }    
    else if (outcome_this_rd == s1011) { //Qubit 2 and 7
        apply_X_on_qubits(psi0, {0,0,1,0,0,0,0,1}, offset, dim, nQ); 

    }    
    else if (outcome_this_rd == s1110) { //Qubit 2 and 4
        apply_X_on_qubits(psi0, {0,0,1,0,1}, offset, dim, nQ); 

    }        
    else if (outcome_this_rd == s1111) { //Qubit 1 and 7
        apply_X_on_qubits(psi0, {0,1,0,0,0,0,0,1}, offset, dim, nQ); 

    }    
    
    
    

}


VectorXc prepare_logical_plus_state(int d, const std::vector<int>& shifted_anc_inds, const std::vector<int>& data_positions,
                                    std::vector<std::pair<int, int>>& index_map,std::vector<std::pair<size_t, size_t>> all_swaps){

    //For the surface code, we need to first prepare the |0>_L or |+>_L either by measurement or unitarily, to do the memory experiment.
    //Here we prepare using a perfect round of stabilizer measurements to project into |+>_L.
    //To measure the stabilizers for X memory, we start from |+>_{data} |+>_{X type ancilla} |0>_{Z type ancilla}
    //Then, we apply the CNOT schedule, and then we measure all data qubits.
    //Output: state |+>_L |+>_{X type ancilla} |0>_{Z type ancilla}

    int n_data  = d*d;
    int n_anc   = (n_data-1);
    int n_anc_X = n_anc/2;
    int n_anc_Z = n_anc/2;
    int nQ      = n_data+n_anc; 

    std::vector<int> idxs_data{0,1,2,3,4,5,6,7,8};
    std::vector<int> idxs_anc_X{9,10,11,12}; //X-type ancilla, measuring Z-type errors
    std::vector<int> idxs_anc{9,10,11,12,13,14,15,16}; //All ancilla

    VectorXc psi_buffer(1<<nQ);
    VectorXc psi_data(1 << n_data);

    VectorXc psi = Ket0(nQ);

    apply_Hadamard_on_qubits(psi,idxs_data);
    apply_Hadamard_on_qubits(psi,idxs_anc_X); // State so far: |+>_{data} |+>_{Xtype ancilla} |0>_{Z type ancilla}

    //Apply CNOT gates
    apply_CNOTs_from_precomputed_swaps(all_swaps, psi);

    //Prepare for X-basis measurement the X-type ancilla
    apply_Hadamard_on_qubits(psi,idxs_anc_X);

    std::unordered_map<uint64_t, std::vector<size_t>> kept_indices; //We dont care about storing something now so this is not returned

    //Measure ancilla (Z stabilizer values will be random)
    std::vector<uint8_t> outcome_this_rd(n_anc);
    outcome_this_rd = measure_all_ancilla(nQ,n_anc,idxs_anc,psi,kept_indices, shifted_anc_inds, data_positions,psi_buffer);
 
    
    //Some Z type ancilla are 1: Want to select the +1 subspace for all stabs, so flip ancilla AND apply data qubit corrections 
    const Eigen::Index dim = psi.size();    
    
    apply_data_correction(nQ, psi, outcome_this_rd);          //Data qubit correction to start from all +1 space
    apply_X_on_qubits(psi, outcome_this_rd, n_data, dim, nQ); //Flip the Z-ancilla to |0> as well

    //Apply Hadamard on X-type ancilla to have them ready for next operations
    apply_Hadamard_on_qubits(psi,idxs_anc_X);

    return psi;

}


void prepare_pre_meas_state(int d, const std::vector<std::pair<size_t, size_t>>& all_swaps, 
                                const ArrayXc& phase_mask, const ArrayXc& ZZ_mask, const std::vector<Real>& prob_Z, VectorXc& psi0) { 
    
    //The input psi0 is the state |+>_L |+>_{X type ancilla}  |0>_{Z type ancilla}
    std::vector<int> idxs_anc_X{9,10,11,12};                                    

    // apply_stochastic_Z_on_qubits(psi,idxs_data, prob_Z); //Test only on data for now

    //Apply the coherent noise error
    apply_precomputed_Rz_mask(psi0, phase_mask);

    //Apply the CNOTs
    apply_CNOTs_from_precomputed_swaps(all_swaps, psi0);

    // apply_precomputed_ZZ_mask(psi, ZZ_mask); //ZZ-errors after the CNOTs
    
    apply_Hadamard_on_qubits(psi0,idxs_anc_X); //Had on the X-type ancilla (will follow with Z-basis measurement)


}

inline std::tuple<Time,Time> reprepare_state(VectorXc &psi, int d,  const std::vector<std::pair<size_t, size_t>>& all_swaps,
                                            const ArrayXc& phase_mask, const ArrayXc& ZZ_mask,const std::vector<Real>& prob_Z){ 
                                                  
    
    Time time_for_Had = 0.0;
    Time time_for_CNOT = 0.0;

    std::vector<int> idxs_anc{9,10,11,12};

    apply_precomputed_Rz_mask(psi, phase_mask);
    std::vector<int> idxs_data{0,1,2,3,4,5,6,7,8};
    // apply_stochastic_Z_on_qubits(psi,idxs_data, prob_Z);                                                        
    
    // auto t0 = Clock::now();

    apply_CNOTs_from_precomputed_swaps(all_swaps, psi);
    
    // apply_precomputed_ZZ_mask(psi, ZZ_mask); //ZZ-errors after the CNOTs

    //Apply the Hadamards on ancilla
    // t0 = Clock::now();

    apply_Hadamard_on_qubits(psi,idxs_anc);
    

    
    // t1 = Clock::now();

    // time_for_Had += Evaluate_Time(t1 - t0).count();

    return {time_for_Had,time_for_CNOT};


}



//TODO: FIX THIS
ArrayXc get_ZZ_phase_mask_for_surface_code(Real theta_G){
    
    const int nQ=17;
    ArrayXc ZZ_mask = VectorXc::Ones(1 << nQ);    


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


inline std::tuple<std::vector<std::pair<size_t, size_t>>, ArrayXc, ArrayXc> prepare_reusable_structures(int d, const std::vector<int>& idxs_all, Real theta_data, Real theta_anc, Real theta_G){

    /*
    Precompute structures that remain constant for the QEC memory experiment.
    
    Input: 
    d: distance of the repetition code
    idxs_all: vector of all the qubit indices
    theta_data: error angle for e^{-i\theta Z} operation for data qubits
    theta_anc: error angle for e^{-i\theta Z} operation for ancilla qubits
    theta_G: error angle for e^{i \theta ZZ} after CNOTs
    
    Output:
    all_swaps: vector of pairs of indices to be swapped
    phase_mask: phase mask for e^{-i\theta Z} errors
    ZZ_mask: phase mask for e^{i \theta ZZ} CNOT errors
    
    */

    int n_data = d*d;
    int n_anc  = (n_data-1);
    int nQ     = n_data+n_anc;

    std::vector<std::pair<size_t, size_t>> all_swaps = get_CNOT_swaps_for_surface_code();

    
    std::vector<Real> thetas(n_data, theta_data);   //Same \theta angle for all data qubits
    thetas.insert(thetas.end(), n_anc, theta_anc);  //Same \theta angle for all ancilla qubits 
    
    ArrayXc phase_mask = precompute_Rz_phase_mask(nQ, idxs_all,  thetas);
    ArrayXc ZZ_mask    = get_ZZ_phase_mask_for_surface_code(theta_G);


    return std::make_tuple(all_swaps, phase_mask, ZZ_mask);
}





Real get_LER_from_uniform_DEM_code_capacity_level(int d, int rds, int ITERS, Real theta_data, Real q_readout, Real pz, bool Reset_ancilla){
   
    // Fixed values/vectors

    if (d>3){ throw std::invalid_argument("Cannot simulate d=5 surface code right now."); }

    Real theta_G = 0.0;
    Real theta_anc = 0.0;


    int n_anc_X = 4;
    int n_anc_Z = 4;
    const int n_data = d*d;    

    const int n_anc  = (n_data-1);
    const int nQ  = n_data+n_anc;

    std::vector<Real> prob_Z;
    for (int k=0; k<n_data; ++k){
        prob_Z.push_back(pz);
    }    

    bool include_stab_reconstruction = true;    
    int rds_effective = rds + (include_stab_reconstruction ? 1 : 0);


    std::vector<int>  idxs_data(n_data);
    for (int i=0; i<n_data; ++i){ idxs_data[i]=i;}

    std::vector<int> idxs_anc(n_anc);
    for (int i = 0; i < n_anc; ++i) {idxs_anc[i] = i + n_data;}

    std::vector<int> idxs_all(nQ);
    for (int i = 0; i < nQ; ++i) {idxs_all[i] = i;}

    std::vector<int> shifted_anc_inds(n_anc);
    for (int i=0; i<n_anc; ++i){
        shifted_anc_inds[i] = nQ-1-idxs_anc[i];
    }


    std::vector<int> shifted_data_bits_from_d(n_data);
    for (int i=0; i<n_data; ++i){
        shifted_data_bits_from_d[i] = n_data - 1 - idxs_data[i]; 
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
    
    std::tie(all_swaps, phase_mask,ZZ_mask) = prepare_reusable_structures( d, idxs_all, theta_data,  theta_anc,  theta_G);
    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, n_data);

    //This prepares |psi0>=|+>_L |+>_{X ancilla} |0>_{Z anc}
    VectorXc psi0 = prepare_logical_plus_state(d, shifted_anc_inds, data_positions,  index_map, all_swaps); //Right after this, (in the absence of errors) probs of X-type anc to be 1 is 50%, probs of Z-type anc to be 1 is 0. OK

    //This calculates H_{X ancilla} * U_{CNOTs} * Error |psi0>
    prepare_pre_meas_state(d,  all_swaps,  phase_mask, ZZ_mask,  prob_Z,  psi0); 

    const Eigen::Index dim = psi0.size();    
    std::unordered_map<uint64_t, std::vector<size_t>> kept_indices_cache; 

    VectorXc psi;    
    psi.resize(1<<nQ); 
    VectorXc psi_buffer(1<<nQ); 
    VectorXc psi_data(1 << n_data);
    std::vector<Real> cumsum_data(1<<n_data);
    std::vector<Real> cdf_buffer_total(1<<nQ);

    cumSum_from_state_vector(psi0,cdf_buffer_total); //Use this for 1st round of measurements


    std::vector<std::vector<uint8_t>> all_data_outcomes;
    all_data_outcomes.resize(ITERS);

    
    std::vector<std::vector<int>> H = get_total_pcm(); 
    std::vector<std::vector<uint8_t>> batch;
    batch.resize(ITERS);

    VectorXc psi_plus_anc_X = plus_state(n_anc_X);     
    VectorXc psi_anc_Z = Ket0(n_anc_Z);
    VectorXc psi_anc = Eigen::kroneckerProduct(psi_plus_anc_X, psi_anc_Z).eval();


    for (int iter=0; iter<ITERS; ++iter){
            
        // psi0    = prepare_pre_meas_state(d,  all_swaps, phase_mask, ZZ_mask, prob_Z);
        // cumSum_from_state_vector(psi0,cdf_buffer_total); //Use this for 1st round of measurements
        
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

            //If it's only 1 round, then the Z-type measurements are random and should not be stored.
            // if (rds==1){
                
            //     //Append only X-type measurements
            //     ancilla_bitstring.insert(ancilla_bitstring.end(),outcome_this_rd.begin(),outcome_this_rd.begin() + 4 );                
                
      
                
            // }
            // else{//Store everything
            //     ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());    
            // }
            


            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {

                Time time_for_Had;
                Time time_for_CNOT;

                psi_data.setZero();
                for (const auto& [i_full, i_reduced] : index_map)
                    psi_data[i_reduced] = psi[i_full];           

                psi_data.normalize();

                psi = Eigen::kroneckerProduct(psi_data, psi_anc).eval();
                
                    
                std::tie(time_for_Had,time_for_CNOT) = reprepare_state(psi, d,  all_swaps, phase_mask, ZZ_mask, prob_Z); 

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
        
        measure_all_data(n_data,shifted_data_bits_from_d,cumsum_data,outcome_of_data); 


        all_data_outcomes[iter] = outcome_of_data;

        if (include_stab_reconstruction==1){ 
            
            //Reconstruct the X-type stabilizer measurements

            ancilla_bitstring.push_back( outcome_of_data[0] ^ outcome_of_data[3] );
            ancilla_bitstring.push_back( outcome_of_data[1] ^ outcome_of_data[2] ^ outcome_of_data[4] ^ outcome_of_data[5] );
            ancilla_bitstring.push_back( outcome_of_data[3] ^ outcome_of_data[4] ^ outcome_of_data[6] ^ outcome_of_data[7] );
            ancilla_bitstring.push_back( outcome_of_data[5] ^ outcome_of_data[8] );
            
            //Pad with extra 0s for the Z-type anc (this helps the formation of defects)
            //Note this is artificial, and we never actually use the last data qubit measurements to
            //reconstruct Z-stabilizer values because we cannot do that (we run X-memory)

            ancilla_bitstring.insert(ancilla_bitstring.end(), 4, 0);
            // if (rds>1){
            //     ancilla_bitstring.insert(ancilla_bitstring.end(), 4, 0);

            // }

        }

            form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);

            //Remove the last Z-round which we artificially put as 0s
            ancilla_bitstring.resize(ancilla_bitstring.size() - n_anc/2);

        // if (rds>1){ //use all n_anc

            
        //     form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);

        //     //Remove the last Z-round which we artificially put as 0s
        //     ancilla_bitstring.resize(ancilla_bitstring.size() - n_anc/2);

        //     //Remove the first Z-round now which is random.
        //     ancilla_bitstring.erase(ancilla_bitstring.begin() + n_anc/2, ancilla_bitstring.begin() + n_anc);


        // }
        // else{//Use half the ancilla (since we only store X-values)

         
        //     form_defects(ancilla_bitstring,  n_anc/2, rds, q_readout, Reset_ancilla,include_stab_reconstruction);
        //     // form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);

        // }


        batch[iter] = ancilla_bitstring;



    }



    //TODO: Figure out which diagonals we want to estimate.

    std::vector<Real> p_space(rds_effective * n_data, 0.1); 
    std::vector<Real> p_time;
    std::vector<Real> p_diag(rds * (4-1), 0.0); 

    if (rds==1){
        //Need to change the Hmatrix, and pass only Hx

        for (int i=0; i< rds * 4; ++i) {
            p_time.push_back(0.0);
        }        


    }
    else{

        for (int i=0; i< rds * n_anc; ++i) {
            p_time.push_back(0.0);
        }                

    }


    ProbDictXZ p_space_o;
    ProbDictXZ p_time_o;
    ProbDictXZ p_bd_o;
    ProbDictXZ p_diag_o;
    bool include_higher_order = false; 
    bool print_higher_order = false;
    std::tie(p_space_o,p_time_o,p_bd_o,p_diag_o) = estimate_edges_surf_code(batch,  d,  n_anc, rds_effective, include_higher_order,  print_higher_order);
    //Estimation of space bulk edges for only stochastic Z data errors is good.

    ProbDict p_space_X = p_space_o["X"];
    ProbDict p_bd_X = p_bd_o["X"];
    ProbDict p_time_X = p_time_o["X"];

    std::cout << "Space X:" << "\n";

    for (const auto& [key, val] : p_space_X.values) {
        std::cout << "key: [";
        for (size_t i = 0; i < key.size(); ++i) {
            std::cout << key[i];
            if (i + 1 < key.size()) std::cout << ", ";
        }
        std::cout << "] ";
        std::cout << "value: " << val << "\n";
    }

    std::cout << "Time X:" << "\n";

    for (const auto& [key, val] : p_time_X.values) {
        std::cout << "key: [";
        for (size_t i = 0; i < key.size(); ++i) {
            std::cout << key[i];
            if (i + 1 < key.size()) std::cout << ", ";
        }
        std::cout << "] ";
        std::cout << "value: " << val << "\n";
    }    
    
    std::cout << "BD X:" << "\n";

    for (const auto& [key, val] : p_bd_X.values) {
        std::cout << "key: [";
        for (size_t i = 0; i < key.size(); ++i) {
            std::cout << key[i];
            if (i + 1 < key.size()) std::cout << ", ";
        }
        std::cout << "] ";
        std::cout << "value: " << val << "\n";
    }    


    
    auto corrections = decode_with_pymatching_create_graph_for_sc_XZ(H, p_space, p_time, p_diag, batch, rds, include_stab_reconstruction);



    Real LER_sum = 0.0;
    for(int iter = 0; iter < ITERS; ++iter){

        //We can do transversal measurement of all qubits to infer parity in the surface code
        //Or just pick one of the logical as below

        int parity = 0;
        // int logical_X_qubits[3] = {0, 1,2}; // left column
        parity ^= all_data_outcomes[iter][3] ^ corrections[iter][0]; //I select the middle qubits only
        parity ^= all_data_outcomes[iter][4] ^ corrections[iter][1];
        parity ^= all_data_outcomes[iter][5] ^ corrections[iter][2];
        // for (int q = 0; q < 9; ++q){
        //     parity ^= (all_data_outcomes[iter][q] ^ corrections[iter][q]);
        // }
        LER_sum += (parity != 0) ? 1.0 : 0.0;
    }

    Real LER = LER_sum / ITERS;    
   

    return LER;
}