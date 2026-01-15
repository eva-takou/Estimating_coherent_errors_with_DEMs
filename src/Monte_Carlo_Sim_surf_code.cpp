#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <complex>
#include <queue>


// #include "estimation_functions_surf_code.h"  //TODO: Make estimation functions for this.
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

    //Next we want to apply the Rz operator only on X-type ancilla
    apply_precomputed_Rz_mask(psi, phase_mask);
    
    //Next we want to apply the CNOTs
    apply_CNOTs_from_precomputed_swaps(all_swaps, psi);

    apply_precomputed_ZZ_mask(psi, ZZ_mask); //ZZ-errors after the CNOTs
    
    //Finally we apply the Hadamards on the ancilla qubits only
    
    //apply_fast_hadamards_on_ancilla_qubits(psi,n_data,nQ);

    apply_Hadamard_on_qubits(psi,idxs_anc); //Had again only on the X-type ancilla

    return psi;
}

inline std::tuple<Time,Time> reprepare_state(VectorXc &psi, int d,  const std::vector<std::pair<size_t, size_t>>& all_swaps,
                                                     const ArrayXc& phase_mask, 
                                                     const ArrayXc& ZZ_mask){ 

    // const int n_data = d*d;
    // const int n_anc  = n_data-1;                                                       
    
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

    apply_Hadamard_on_qubits(psi,idxs_anc);
    

    
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
//TODO: Do tests to make sure this is correct.
std::vector<std::pair<size_t, size_t>> find_CNOT_swaps_for_surface_code(){

    //Following schedule pattern from this paper: https://arxiv.org/pdf/2511.06758

    //Fix the stuff below...

    //Note X-type ancilla are control qubits, Z-type ancilla are target qubits.

    std::vector<std::pair<size_t, size_t>> all_swaps;
    const int nQ=17;

    // controls_1st = [X1, X2, X3, 0,4, 6]
    // targets_1st  = [1, 3, 5, Z1, Z2, Z3]


    int X_shift = 9;   //X_shift+3 = 12 (9,10,11,12) Xchecks
    int Z_shift = 9+4; //9+4 = 13 (13,14,15,16) Zchecks

    //Step 1
    auto swaps = precompute_CNOT_swaps(X_shift+1,{1} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(X_shift+2,{3} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(X_shift+3,{5} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(0,{Z_shift+1} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(4,{Z_shift+2} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(6,{Z_shift+3} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());

    // controls_2nd = [X1, X2, X3, 1, 5, 7 ]    
    // targets_2nd = [4, 6, 8, Z1, Z2, Z3]
    auto swaps = precompute_CNOT_swaps(X_shift+1,{4} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(X_shift+2,{6} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(X_shift+3,{8} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(1,{Z_shift+1} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(5,{Z_shift+2} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(7,{Z_shift+3} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());

    // controls_3rd = [X0, X1, X2, 1, 3, 7]    
    // targets_3rd = [0, 2, 4, Z0, Z1, Z2]
    auto swaps = precompute_CNOT_swaps(X_shift+0,{0} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(X_shift+1,{2} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(X_shift+2,{4} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(1,{Z_shift+0} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(3,{Z_shift+1} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(7,{Z_shift+2} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());

    // controls_4th = [X0, X1, X2, 2, 4, 8]    
    // targets_4th = [3, 5, 7, Z0, Z1, Z2]
    auto swaps = precompute_CNOT_swaps(X_shift+0,{3} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(X_shift+1,{5} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(X_shift+2,{7} , nQ); //X-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(2,{Z_shift+0} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(4,{Z_shift+1} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());
    swaps = precompute_CNOT_swaps(8,{Z_shift+2} , nQ); //Z-type
    all_swaps.insert(all_swaps.end(), swaps.begin(), swaps.end());






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



//TODO:  It has to be that i follow column-major order of first the X and then the Z-checks?
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

    int n_data = nQ-n_anc; 
    


    std::vector<std::pair<size_t, size_t>> all_swaps = find_CNOT_swaps_for_surface_code();

    
    std::vector<Real> thetas(n_data, theta_data);        //Same \theta angle for all data qubits
    thetas.insert(thetas.end(), n_anc, theta_anc);  //Same \theta angle for all ancilla qubits 
    
    ArrayXc phase_mask = precompute_Rz_phase_mask(nQ, idxs_all,  thetas);


    ArrayXc ZZ_mask = get_ZZ_phase_mask_for_surface_code(theta_G);


    return std::make_tuple(all_swaps, phase_mask, ZZ_mask);
}

//TODO: Also not entirely sure, if I'm forming the defects correctly..
Real get_LER_from_uniform_DEM_code_capacity_level(int d, int rds, int ITERS, Real theta_data, Real q_readout,  bool Reset_ancilla){
   
    // Fixed values/vectors

    if (d>3){ throw std::invalid_argument("Cannot simulate d=5 surface code right now."); }

    int n_anc_X = 4;
    int n_anc_Z = 4;
    const int n_data = d*d;    
    const int n_anc  = n_data-1;
    const int nQ  = n_data+n_anc;

    Real theta_G = 0.0;
    Real theta_anc = 0.0;

    std::vector<int>  idxs_data(n_data);
    for (int i=0; i<n_data; ++i){ idxs_data[i]=i;}

    std::vector<int> idxs_anc{9,10,11,12, 13,14,15,16}; //First 4 are Xtype, next are Ztype

    std::vector<int> idxs_all(nQ);
    for (int i = 0; i < nQ; ++i) idxs_all[i] = i;

    std::vector<int> shifted_anc_inds(n_anc);
    

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

    std::vector<std::pair<int, int>> index_map = precompute_kept_index_map_for_ptrace_of_ancilla(n_anc, n_data);

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

    
    std::vector<std::vector<int>> H = get_total_pcm(); 
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

            //If it's only 1 round, then the Z-type measurements are random and should not be stored.
            if (rds==1){
                //Only 4 X-type measurements
                for (int i=0; i< 4; ++i) {
                    ancilla_bitstring.push_back(outcome_this_rd[i]);
                }                
            }
            else{//Store everything
                ancilla_bitstring.insert(ancilla_bitstring.end(), outcome_this_rd.begin(), outcome_this_rd.end());    
            }
            

            // Prepare state for next round, unless we are done with QEC rds 
            if (r != rds - 1) {

                Time time_for_Had;
                Time time_for_CNOT;

                psi_data.setZero();
                for (const auto& [i_full, i_reduced] : index_map)
                    psi_data[i_reduced] = psi[i_full];           

                psi = Eigen::kroneckerProduct(psi_data, psi_anc).eval();

                psi.normalize();
                    
                std::tie(time_for_Had,time_for_CNOT) = reprepare_state(psi, d,  all_swaps, phase_mask, ZZ_mask); 

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
            
            //Reconstruct the X-type stabilizer measurements (because we do Z-basis)

            ancilla_bitstring.push_back( outcome_of_data[0] ^ outcome_of_data[3] );
            ancilla_bitstring.push_back( outcome_of_data[1] ^ outcome_of_data[2] ^ outcome_of_data[4] ^ outcome_of_data[5] );
            ancilla_bitstring.push_back( outcome_of_data[3] ^ outcome_of_data[4] ^ outcome_of_data[6] ^ outcome_of_data[7] );
            ancilla_bitstring.push_back( outcome_of_data[5] ^ outcome_of_data[8] );
            
            //Pad with extra 0s for the Z-type anc (this helps the formation of defects)
            //Note this is artificial, and we never actually use the last data qubit measurements to
            //reconstruct Z-stabilizer values because we cannot do that (we run X-memory)

            ancilla_bitstring.push_back(0);
            ancilla_bitstring.push_back(0);
            ancilla_bitstring.push_back(0);
            ancilla_bitstring.push_back(0);
        }

        if (rds>1){ //use all n_anc

            //This will be incorrect, because X-type ancilla have a distance of 2 rds instead of just 1 rd
            //same for Z-type ancilla, + we postpone 1 round of detectors..

            //So i have something like an array of outcomes 
            //[X X X X , Z Z Z Z, X X X X, Z Z Z Z... ]
            
            //for Xdets, we let anc \in[n_anc_X]

            //indx1 = anc + n_anc * rd, indx2 = anc + n_anc * (rd-1)
            //So, w/ n_anc = 8
            //if rd=1, anc=0 -> indx1 = 8, indx2 = 0 (OK)
            //if rd=1, anc=1 -> indx1 = 9, indx2 = 1 (OK) ... etc so it seems ok

            //for Zdets, we need to skip one round, so i think again we have
            
            //indx1 = anc + n_anc * rd, indx2 = anc + n_anc * (rd-1)
            
            //but now we start with rd=1, and we also have anc \in [n_anc_X,n_anc_X+n_anc_Z]
            //for example, if we have anc = 4, and rd=1
            //indx1 = 4 + 8*1 = 12, indx2 = 4 + 0 = 4 
            //ok so we do the same, but we need to remove the 1st Z-round after,
            //so it means we go to locations n_anc/2 till n_anc and just pop them out

            //TODO: I think the Z-type need to end also 1 round sooner
            //since there is no stabilizer reconstruction.

            //We don't do stab reconstruction for final data qubit measurements
            //to create Z-type stabs. But to ensure correct counting
            //I should pad with extra 0s so i can apply the transformation
            //and then remove the last entries

            
            form_defects(ancilla_bitstring,  n_anc, rds, q_readout, Reset_ancilla,include_stab_reconstruction);

            //Remove the last Z-round which we artificially put as 0s
            ancilla_bitstring.resize(ancilla_bitstring.size() - n_anc/2);

            //Remove the first Z-round now which is random.
            ancilla_bitstring.erase(ancilla_bitstring.begin() + n_anc/2, ancilla_bitstring.begin() + n_anc);


        }
        else{//Use half the ancilla (since we only store X-values)
            
            //because n_anc is the total number, and an int, then in c++ n_anc/2 will remain an int
            //This is fine, we use only X-type ancilla, and bitxor directly w/ previous round outcomes
            form_defects(ancilla_bitstring,  n_anc/2, rds, q_readout, Reset_ancilla,include_stab_reconstruction);
        }

        
        
        //I'm taking the indx1 = anc + n_anc * rd
        //then       the indx2 = anc + n_anc * (rd-1)
        //so if let's say we have only the X-type then
        // rd=1, anc=0 -> indx1 = 4, indx2 = 0 
        // rd=1, anc=1 -> indx1 = 5, indx2 = 1 (so looks correct)


        batch[iter] = ancilla_bitstring;

    }

    //TODO: Fix the diagonal probs

    std::vector<Real> p_space(rds_effective * n_data, 0.1); 
    std::vector<Real> p_time;
    std::vector<Real> p_diag(rds * (4-1), 0.0); 

    if (rds==1){
        //Need to change the Hmatrix, and pass only Hx

        for (int i=0; i< rds * 4; ++i) {
            p_time.push_back(0.0);
        }        
        

        H =  get_Hx_sc();

    }
    else{

        for (int i=0; i< rds * n_anc; ++i) {
            p_time.push_back(0.0);
        }                

    }
    
    std::cout << "Rows,cols of pcm: " << H.size() << ", " << H[0].size() << "\n";
    std::cout << "Rows,cols of batch: " << batch.size() << ", " << batch[0].size() << "\n";
    
    auto corrections = decode_with_pymatching_create_graph(H, p_space, p_time, p_diag, batch, rds, include_stab_reconstruction);

    Real LER_sum = 0.0;
    for(int iter = 0; iter < ITERS; ++iter){

        //We can do transversal measurement of all qubits to infer parity in the surface code
        //Or just pick one of the logical as below

        int parity = 0;
        int logical_X_qubits[3] = {0, 3, 6}; // left column
        for (int q = 0; q < 3; ++q){
            parity ^= (all_data_outcomes[iter][logical_X_qubits[q]] ^ corrections[iter][logical_X_qubits[q]]);
        }
        LER_sum += (parity != 0) ? 1.0 : 0.0;
    }

    Real LER = LER_sum / ITERS;    
   

    return LER;
}