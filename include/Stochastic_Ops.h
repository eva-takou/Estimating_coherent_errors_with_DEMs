#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <unordered_set>
#include <cassert>
#include <utility>
#include <cmath>

#include "PrecisionOfTypes.h"

#include <set>
#include <random>

static std::mt19937 rng(std::random_device{}());

using namespace Eigen;
using std::vector;


inline void apply_stochastic_Z_on_qubits(VectorXc& psi,const std::vector<int>& qubits,const Real prob_Z,int nQ){
    std::vector<int> qubits_to_apply_Z;
    qubits_to_apply_Z.reserve(qubits.size());

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t q = 0; q < qubits.size(); ++q) {
        if (dist(rng) < prob_Z) {
            qubits_to_apply_Z.push_back(qubits[q]);
        }
    }

    Eigen::Index phase_mask = 0;
    for (int q : qubits_to_apply_Z) {
        phase_mask |= (1ULL << (nQ - 1 - q));   // MSB ordering
    }

    if (phase_mask == 0) return;

    const Eigen::Index dim = psi.size();

    for (Eigen::Index j = 0; j < dim; ++j) {
        if (j & phase_mask) {
            psi[j] = -psi[j];
        }
    }
}

inline void apply_stochastic_X_on_qubits(VectorXc& psi, const std::vector<int>& qubits, const Real prob_X, int nQ){
    
    std::vector<int> qubits_to_apply_X;
    qubits_to_apply_X.reserve(qubits.size());
    std::uniform_real_distribution<double> dist(0.0, 1.0);    

    for (size_t q = 0; q < qubits.size(); ++q) {

        if (dist(rng) < prob_X) {
            qubits_to_apply_X.push_back(qubits[q]);
        }
    }


    Eigen::Index flip_mask = 0;

    for (int q : qubits_to_apply_X) {
            flip_mask |= (1ULL << (nQ - 1 - q));   // MSB ordering
        }    

    if (flip_mask == 0) return; 

    const Eigen::Index dim = psi.size();
    
    for (Eigen::Index j = 0; j < dim; ++j) {
        Eigen::Index j_flip = j ^ flip_mask;
        if (j < j_flip) {
            std::swap(psi[j], psi[j_flip]);
        }
    }    


}


inline void apply_stochastic_Y_on_qubits(VectorXc& psi,const std::vector<int>& qubits,const Real prob_Y,int nQ){
    std::vector<int> qubits_to_apply_Y;
    qubits_to_apply_Y.reserve(qubits.size());

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t q = 0; q < qubits.size(); ++q) {
        if (dist(rng) < prob_Y) {
            qubits_to_apply_Y.push_back(qubits[q]);
        }
    }

    Eigen::Index flip_mask = 0;
    for (int q : qubits_to_apply_Y) {
        flip_mask |= (1ULL << (nQ - 1 - q));   // MSB ordering
    }

    if (flip_mask == 0) return;

    const Eigen::Index dim = psi.size();

    const std::complex<double> I(0.0, 1.0);

    for (Eigen::Index j = 0; j < dim; ++j) {

        Eigen::Index j_flip = j ^ flip_mask;

        if (j < j_flip) {

            bool bit_j      = (j & flip_mask);
            bool bit_jflip  = (j_flip & flip_mask);

            auto a = psi[j];
            auto b = psi[j_flip];

            // |0> ->  i|1>
            // |1> -> -i|0>

            psi[j]      = bit_jflip ? -I * b :  I * b;
            psi[j_flip] = bit_j     ? -I * a :  I * a;
        }
    }
}


inline void apply_single_depol_on_qubits(VectorXc& psi, const std::vector<int>& qubits, Real prob_depol1, int nQ){


    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> dist2(0, 2);

    for (int qubit: qubits){

        if (dist(rng) < prob_depol1){

            //We appply X or Y or Z with equal prob

            int sample = dist2(rng);    
            if (sample == 0){
                apply_stochastic_X_on_qubits( psi,  {qubit}, 1.1,  nQ);
            }
            else if (sample==1){
                apply_stochastic_Y_on_qubits( psi,  {qubit}, 1.1,  nQ);
            }
            else if (sample==2){
                apply_stochastic_Z_on_qubits( psi,  {qubit}, 1.1,  nQ);
            }


        }



    }
    


}

inline void apply_two_qubit_pauli(VectorXc& psi, int q1, int q2, char P1, char P2, int nQ){

    // 'I','X','Y','Z'
    auto has_flip  = [](char P){ return P=='X' || P=='Y'; };
    auto has_Z     = [](char P){ return P=='Z' || P=='Y'; };
    auto is_Y      = [](char P){ return P=='Y'; };

    Eigen::Index mask1 = 1ULL << (nQ - 1 - q1);
    Eigen::Index mask2 = 1ULL << (nQ - 1 - q2);

    Eigen::Index flip_mask = 0;
    if (has_flip(P1)) flip_mask |= mask1;
    if (has_flip(P2)) flip_mask |= mask2;

    const Eigen::Index dim = psi.size();

    for (Eigen::Index j = 0; j < dim; ++j) {

        Eigen::Index j_flip = j ^ flip_mask;
        if (j > j_flip) continue;

        // original bits
        bool b1 = (j & mask1);
        bool b2 = (j & mask2);

        std::complex<double> phase = 1.0;

        // Z phases
        if (has_Z(P1) && b1) phase *= -1.0;
        if (has_Z(P2) && b2) phase *= -1.0;

        // Y extra ±i phase (depends on original bit)
        if (is_Y(P1)) phase *= (b1 ? std::complex<double>(0,-1)
                                   : std::complex<double>(0, 1));
        if (is_Y(P2)) phase *= (b2 ? std::complex<double>(0,-1)
                                   : std::complex<double>(0, 1));

        if (j == j_flip) {
            psi[j] *= phase;
        } else {
            auto tmp = psi[j];
            psi[j] = phase * psi[j_flip];
            psi[j_flip] = std::conj(phase) * tmp; 
        }
    }
}


inline void apply_twoQ_depol_on_qubits(VectorXc& psi, 
                                       const std::vector<int>& qubits1,
                                       const std::vector<int>& qubits2, 
                                       Real prob_depol2, int nQ) {
    static const std::vector<std::pair<char,char>> pauli_combs = {
        {'X','I'},{'Y','I'},{'Z','I'},
        {'I','X'},{'I','Y'},{'I','Z'},
        {'X','X'},{'X','Y'},{'X','Z'},
        {'Y','X'},{'Y','Y'},{'Y','Z'},
        {'Z','X'},{'Z','Y'},{'Z','Z'}
    };

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<size_t> pick(0, pauli_combs.size()-1);

    for (size_t k=0; k<qubits1.size(); ++k){
        if (dist(rng) < prob_depol2) {
            auto PP = pauli_combs[pick(rng)]; // pick a two-qubit Pauli randomly
            apply_two_qubit_pauli(psi, qubits1[k], qubits2[k], PP.first, PP.second, nQ);
        }
    }
}