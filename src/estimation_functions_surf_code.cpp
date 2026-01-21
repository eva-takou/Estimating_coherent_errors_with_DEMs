#include <vector>
#include <cmath>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <string>
#include "PrecisionOfTypes.h"
#include "utils_sc.h"

using namespace Eigen;


struct VectorHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t h = 0;
        for (int x : v) {
            h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

void print_key(const std::vector<int>& key) {
    std::cout << "(";
    for (size_t i = 0; i < key.size(); i++) {
        std::cout << key[i];
        if (i + 1 < key.size()) {
            std::cout << ",";
        }
    }
    std::cout << ")";
}

struct ProbDict {
    std::unordered_map<std::vector<int>, double, VectorHash> values;
};

// struct ProbDictXZ {
//     ProbDict X;
//     ProbDict Z;

//     ProbDict& operator[](const std::string& type) {
//         if (type == "X") return X;
//         if (type == "Z") return Z;
//         throw std::invalid_argument("Invalid type, must be 'X' or 'Z'");
//     }

//     const ProbDict& operator[](const std::string& type) const {
//         if (type == "X") return X;
//         if (type == "Z") return Z;
//         throw std::invalid_argument("Invalid type, must be 'X' or 'Z'");
//     }
// };




Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> cast_as_Eigen_batch(const std::vector<std::vector<uint8_t>>& batch) {
    /*
    Convert batch to Eigen array so that we can count detection events faster.
    
    Input:
    batch: array of detection events (nsims x num_dets)
    
    Output:
    eigen_batch: array of detection events (converted to Eigen array)
    */

    int nsims     = batch.size();
    int total_len = batch[0].size();

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> eigen_batch(nsims, total_len);
    for (int i = 0; i < nsims; ++i) {
        for (int j = 0; j < total_len; ++j) {
            eigen_batch(i, j) = static_cast<bool>(batch[i][j]);
        }
    }

    return eigen_batch;
}

std::vector<Real> calculate_vi_mean(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, int nsims){
    /*
    Get the average detection counts <vi>.
    
    Input:
    eigen_batch: array of detector counts (nsims x num_dets)
    nsims: total # of shots
    n_stabs: # of stabilizers
    rds: total # of rds (including the effective last data qubit measurement)

    Output:
    vi_mean: the average detector counts
    */

    int num_dets = eigen_batch.cols();
    std::vector<Real> vi_mean(num_dets);

    for (int col = 0; col < num_dets; ++col) {
        vi_mean[col] = static_cast<Real>(eigen_batch.col(col).count()) / nsims;
    }

    return vi_mean;    
    

}

inline Real get_prob_pij(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, const std::vector<Real>& vi_mean, int indx1, int indx2, int nsims){
    /*
    Calculate the probability p_{ij} given <vi> and <vivj> detection events.

    Input:
    eigen_batch: batch of detection events
    vi_mean: vector of averages <vi>
    indx1: label of detector 1
    indx2: label of detector 2
    nsims: total # of shots
    nstabs: number of stabilizers

    Output:
    pij: probability 1/2 - \sqrt{1/4 - (<vivj>-<vivj>)/(1-2(<vi>+<vj>)+4<vivj>)}
    */

    Real v1   = vi_mean[indx1];
    Real v2   = vi_mean[indx2];

    auto joint_counts =  eigen_batch.col(indx1) && eigen_batch.col(indx2);

    Real v1v2 = static_cast<Real> (joint_counts.count())/nsims;
    
    Real denom  = 1.0 - 2.0 * (v1+v2) + 4.0 * v1v2;

    if (std::abs(denom) < 1e-12) {

        std::cout << "Skipped due to small denom." ;
        return 0.0;
    }

    Real numer       = v1v2 - v1*v2;
    Real sq_argument = 0.25 - numer/denom;

    //If sq_argument<0, then we get imaginary part, so return 0
    if (sq_argument<0){

        std::cout << "Negative sq_argument in p_{ij} for indx1 = " << indx1 << "for indx2 = " << indx2 
                    << " (sq_argument = " << sq_argument << ")" << std::endl;        

        return 0.0;
    }

    //If sq_argument is positive, but bigger than 0.5, then the probability can be negative so we return 0 again

    Real square_root = std::sqrt(sq_argument);
    Real p = 0.5 - square_root;

    if (p<0.0){
        std::cout << "FOUND NEGATIVE p_{ij}!\n";
        p=0.0;

    }

    if (p>1.0){
        std::cout << "p_{ij} above 1!\n";
        std::cout << "numer:" << numer << std::endl;      
        std::cout << "denom:" << denom << std::endl;        
    }

    return p;
}



inline bool is_subset(const std::vector<int>& subset, const std::vector<int>& superset) {
    // print_key(subset);
    // print_key(superset);
    for (int x : subset) {
        if (std::find(superset.begin(), superset.end(), x) == superset.end()) {
            
            return false;
        }
    }
    
    return true;
}

inline bool contains(int x, const std::vector<int>& v) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

bool has_overlap(const std::vector<int>& a, const std::vector<int>& b) {
    for (int x : a) {
        if (std::find(b.begin(), b.end(), x) != b.end()) {
            return true;  
        }
    }
    return false;
}

inline Real redefine_lowest_order_prob(const std::vector<int>& lower_order_key, Real lower_order_prob, const ProbDict& higher_order_probs){
    /*
    Modify probability of edge by removing higher order contribution, via the equation: p = (p-p_{high})/(1-2p_{high}).
    We can choose to remove contributions from all higher order events which overlap with the lower order event,
    or we can just choose to remove selected events.
    
    Input:
    lower_order_key: indices defining p_{ij} or p_{i} for boundary (e.g. p_{01} has the key={0,1})
    lower_order_prob: input probability of an edge
    higher_order_probs: dictionary with higher order probabilities
    
    Output:
    updated_prob: modified probability after subtracting a higher order event.
    */
    
    
    Real updated_prob = lower_order_prob;

    // for (const auto& [higher_order_key, higher_order_val] : higher_order_probs.values) {

    //     if (std::search(higher_order_key.begin(), higher_order_key.end(), lower_order_key.begin(), lower_order_key.end()) != higher_order_key.end()) {

    //         // std::cout << "REFINED bulk/boundary prob!\n";
    //         updated_prob = (updated_prob - higher_order_val )/(1.0-2.0*higher_order_val);

    //         if (updated_prob<0.0){
    //             std::cout<< "In the updated prob, i have to set 0 \n" ;
    //             updated_prob=0.0;
    //         }
    //     }        
    // }

    for (const auto& [higher_order_key, higher_order_val] : higher_order_probs.values) {

        if (is_subset(lower_order_key,higher_order_key)){
        
            updated_prob = (updated_prob - higher_order_val)/(1.0-2.0*higher_order_val);

            if (updated_prob<0.0){
                std::cout<< "In the updated prob, i have to set 0 \n" ;
                updated_prob=0.0;
            }
            return updated_prob;        
        }        
    }    
    return updated_prob;
}


inline Real multiply_exclusion_factor(const std::vector<int>& lower_order_key, Real term_to_correct, const ProbDict& four_point_probs){
    /*
    Multiply by 1/(1-2*p_{ijkl}) correction factor to the 3-point probability term.
    */

    Real updated_term = term_to_correct;

    for (const auto& [four_key, four_val] : four_point_probs.values) {

        if (std::search(four_key.begin(), four_key.end(), lower_order_key.begin(), lower_order_key.end()) != four_key.end()) {
            
            updated_term *= 1.0/(1.0-2.0*four_val);

        }        
    }
    return updated_term;
}


// Higher-order correlations are calculated based on the formulas of Ref https://arxiv.org/pdf/2502.17722.

inline Real get_four_point_prob(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, const std::vector<Real>& vi_mean, int nsims, int n_stabs, int rds,
                                int anc1, int anc2, int anc3, int anc4, int rd1, int rd2, int rd3, int rd4){
    

    int indx1 = anc1 + n_stabs * rd1;
    int indx2 = anc2 + n_stabs * rd2;
    int indx3 = anc3 + n_stabs * rd3;
    int indx4 = anc4 + n_stabs * rd4;

    Real v1 = vi_mean[indx1];
    Real v2 = vi_mean[indx2];
    Real v3 = vi_mean[indx3];
    Real v4 = vi_mean[indx4];

    auto col1 = eigen_batch.col(indx1);
    auto col2 = eigen_batch.col(indx2);
    auto col3 = eigen_batch.col(indx3);    
    auto col4 = eigen_batch.col(indx4);    

    // Two-point:

    //v1v2
    auto joint_counts12 = col1 && col2;
    Real v1v2 = static_cast<Real> (joint_counts12.count())/nsims;

    //v1v3 
    auto joint_counts13 = col1 && col3;
    Real v1v3 = static_cast<Real> (joint_counts13.count())/nsims;

    //v1v4 
    auto joint_counts14 = col1 && col4;
    Real v1v4 = static_cast<Real> (joint_counts14.count())/nsims;

    //v2v3
    auto joint_counts23 = col2 && col3;
    Real v2v3 = static_cast<Real> (joint_counts23.count())/nsims;

    //v2v4
    auto joint_counts24 = col2 && col4;
    Real v2v4 = static_cast<Real> (joint_counts24.count())/nsims;

    //v3v4
    auto joint_counts34 = col3 && col4;
    Real v3v4 = static_cast<Real> (joint_counts34.count())/nsims;

    // Three-point:

    //v1v2v3
    auto joint_counts123 = joint_counts12 && col3;
    Real v1v2v3 = static_cast<Real>(joint_counts123.count()) / nsims;
    

    //v1v2v4
    auto joint_counts124 = joint_counts12 && col4;
    Real v1v2v4 = static_cast<Real>(joint_counts124.count()) / nsims;
    

    //v1v3v4
    auto joint_counts134 = joint_counts13 && col4;
    Real v1v3v4 = static_cast<Real>(joint_counts134.count()) / nsims;

    //v2v3v4
    auto joint_counts234 = joint_counts23 && col4;
    Real v2v3v4 = static_cast<Real>(joint_counts234.count()) / nsims;

    //Four point:

    //v1v2v3v4
    auto joint_counts1234 = joint_counts123 && col4;
    Real v1v2v3v4 = static_cast<Real>(joint_counts1234.count()) / nsims;
    std::cout << "v1v2v3v4= " << v1v2v3v4 << std::endl;

    Real DENOM = 1.0;
    DENOM *= 1.0 - 2.0 * (v1+v2) + 4.0 * v1v2; //two-point
    DENOM *= 1.0 - 2.0 * (v1+v3) + 4.0 * v1v3; 
    DENOM *= 1.0 - 2.0 * (v1+v4) + 4.0 * v1v4; 
    DENOM *= 1.0 - 2.0 * (v2+v3) + 4.0 * v2v3; 
    DENOM *= 1.0 - 2.0 * (v2+v4) + 4.0 * v2v4; 
    DENOM *= 1.0 - 2.0 * (v3+v4) + 4.0 * v3v4; 
    DENOM *= 1.0 - 2.0 * (v1+v2+v3+v4) + 4.0 * (v1v2 + v1v3 + v1v4 + v2v3 + v2v4 + v3v4) - 8.0 * (v1v2v3 + v1v2v4 + v2v3v4+ v1v3v4) + 16.0 * v1v2v3v4;                                    


    Real NUMER = 1.0;
    NUMER *= (1.0 - 2.0 * v1) * (1.0 - 2.0 * v2) * (1.0 - 2.0 * v3) * (1.0 - 2.0 * v4); // single-point 
    NUMER *= 1.0 - 2.0 * (v1+v2+v3) + 4.0 * (v1v2 + v1v3 + v2v3) - 8.0 * v1v2v3; //three-point
    NUMER *= 1.0 - 2.0 * (v1+v2+v4) + 4.0 * (v1v2 + v1v4 + v2v4) - 8.0 * v1v2v4;
    NUMER *= 1.0 - 2.0 * (v1+v3+v4) + 4.0 * (v1v3 + v1v4 + v3v4) - 8.0 * v1v3v4;
    NUMER *= 1.0 - 2.0 * (v2+v3+v4) + 4.0 * (v2v3 + v2v4 + v3v4) - 8.0 * v2v3v4;


    Real p = 0.5 - 0.5 * std::pow(NUMER/DENOM, 1.0/8.0);

    if (p<0.0){
        std::cout << "negative 4pnt prob: " << p << "\n";
        p=0.0;
    }

    return p;


}

inline Real get_three_point_prob(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, const std::vector<Real>& vi_mean, int nsims, int n_stabs, int rds,
                                 int anc1, int anc2, int anc3, int rd1, int rd2, int rd3, ProbDict four_point_probs){
    

    

    int indx1 = anc1 + n_stabs * rd1;
    int indx2 = anc2 + n_stabs * rd2;
    int indx3 = anc3 + n_stabs * rd3;
    
    Real v1 = vi_mean[indx1];
    Real v2 = vi_mean[indx2];
    Real v3 = vi_mean[indx3];
    
    auto col1 = eigen_batch.col(indx1);
    auto col2 = eigen_batch.col(indx2);
    auto col3 = eigen_batch.col(indx3);        

    // Two-point:

    //v1v2
    auto joint_counts12 = col1 && col2;
    Real v1v2 = static_cast<Real> (joint_counts12.count())/nsims;

    //v1v3 
    auto joint_counts13 = col1 && col3;
    Real v1v3 = static_cast<Real> (joint_counts13.count())/nsims;

    //v2v3
    auto joint_counts23 = col2 && col3;
    Real v2v3 = static_cast<Real> (joint_counts23.count())/nsims;

    // Three-point:

    //v1v2v3
    auto joint_counts123 = joint_counts12 && col3;
    Real v1v2v3 = static_cast<Real>(joint_counts123.count()) / nsims;
    std::cout << "v1v2v3= " << v1v2v3 << std::endl;


    Real DENOM = 1.0;


    DENOM *= (1.0 - 2.0 * (v1+v2) + 4.0 * v1v2);
    DENOM *= (1.0 - 2.0 * (v1+v3) + 4.0 * v1v3);
    DENOM *= (1.0 - 2.0 * (v2+v3) + 4.0 * v2v3);
    
    Real NUMER = 1.0;
    NUMER *= (1.0 - 2.0 * v1) * (1.0 - 2.0 * v2) * (1.0 - 2.0 * v3);
    NUMER *= (1.0 - 2.0 * (v1 + v2 + v3) + 4.0 * (v1v2 + v1v3 + v2v3) - 8.0 * v1v2v3) ; 

    //Extra is 1/(1-2*p_{ijkl}) where l \not in {ijk} 
    

    std::vector<int> lower_order_key = {indx1,indx2,indx3};
    Real term_to_correct = 0.5 * std::sqrt(std::sqrt(NUMER/DENOM)); 
    

    term_to_correct = multiply_exclusion_factor(lower_order_key, term_to_correct, four_point_probs);

    Real p = 0.5 - term_to_correct; //this is p = p_{ijk}


    if (p<0.0){
        std::cout << "negative 3pnt prob: " << p << "\n";
        p=0.0;
    }
    return p;


}

//TODO: UPDATE
ProbDict get_all_four_point_probs(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, const std::vector<Real>& vi_mean, 
                                  int nsims, int n_stabs, int rds){

    
    ProbDict four_point_probs; 

    for (int rd=0; rd<rds-1; rd++){

        int rd1 = rd;
        int rd2 = rd1;
        int rd3 = rd1+1;
        int rd4 = rd3;

        for (int anc=0; anc<n_stabs-1; anc++){

            int anc1 = anc;
            int anc2 = anc1+1;
            int anc3 = anc1;
            int anc4 = anc2;

            int indx1 = anc1 + n_stabs * rd1;
            int indx2 = anc2 + n_stabs * rd2;
            int indx3 = anc3 + n_stabs * rd3;
            int indx4 = anc4 + n_stabs * rd4;



            std::vector<int> key = {indx1, indx2, indx3, indx4};



            Real pijkl = get_four_point_prob(eigen_batch, vi_mean, nsims, n_stabs, rds,
                                             anc1,  anc2,  anc3,  anc4,  rd1,  rd2,  rd3,  rd4);
                
            four_point_probs.values[key] = pijkl;
         
        }

    }   

    return four_point_probs;

}

//TODO: UPDATE
ProbDict get_all_three_point_probs(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, const std::vector<Real>& vi_mean, int nsims, int n_stabs, int rds,
                                    ProbDict four_point_probs){
    
    
                                     
    ProbDict three_point_probs; 

    //Get 3pnt probs according to upper diagonals  (anc1 and anc2 in same rd, other anc in next rd)
    for (int rd=0; rd<rds-1; ++rd){

        int rd1 = rd;
        int rd2 = rd;
        int rd3 = rd+1;

        for (int anc=0; anc<n_stabs-1; ++anc){

            int anc1 = anc;
            int anc2 = anc1+1;
            int anc3 = anc;

            int indx1 = anc1 + n_stabs * rd1;
            int indx2 = anc2 + n_stabs * rd2;
            int indx3 = anc3 + n_stabs * rd3;


            Real pijk=get_three_point_prob(eigen_batch, vi_mean,  nsims,  n_stabs,  rds,
                                            anc1,  anc2,  anc3,  rd1,  rd2,  rd3,  four_point_probs);                            

            std::vector<int> key = {indx1, indx2, indx3};
    
            three_point_probs.values[key] = pijk;

        }

    }

    //Get 3pnt probs according to lower diagonals  (anc1 in one rd, anc2 and 3 in next rd)
    // for (int rd=0; rd<rds-1; ++rd){

    //     int rd1 = rd;
    //     int rd2 = rd+1;
    //     int rd3 = rd+1;

    //     for (int anc=1; anc<n_stabs; ++anc){

    //         int anc1 = anc;
    //         int anc2 = anc1-1;
    //         int anc3 = anc;

    //         int indx1 = anc1 + n_stabs * rd1;
    //         int indx2 = anc2 + n_stabs * rd2;
    //         int indx3 = anc3 + n_stabs * rd3;

    //         std::cout << "Triplet: (" << indx1 << "," << indx2 << "," << indx3 << ")\n";


    //         Real pijk=get_three_point_prob(eigen_batch, vi_mean,  nsims,  n_stabs,  rds,
    //                                         anc1,  anc2,  anc3,  rd1,  rd2,  rd3,  four_point_probs);                            

    //         std::vector<int> key = {indx1, indx2, indx3};
                
    //         three_point_probs.values[key] = pijk;

    //         std::cout << "3pnt value:" << pijk << "\n";


    //     }

    // }
    

    return three_point_probs;


}


std::tuple<ProbDict,ProbDict> estimate_time_edges(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, const std::vector<Real>& vi_mean, 
                                                                    int nsims,  const ProbDict& four_point_probs, const ProbDict& three_point_probs,
                                                                    const std::vector<std::vector<int>>& X_det_inds_per_rd, const std::vector<std::vector<int>>& Z_det_inds_per_rd){

    /*
    Calculate the probability p_{ij} of time edges. If four_point_probs or three_point_probs is empty, we truncate the estimation
    to 2-point correlators. If those dictionaries are non-empty, we redefine the edge probabilities using higher-order events.

    Input:
    eigen_batch: batch of detection events
    vi_mean: vector of averages <vi>
    nsims: total # of shots
    four_point_probs: probability dictionary of 4-point events
    three_point_probs: probability dictionary of 3-point events
    X_det_inds_per_rd: indices of X type detectors per round (vector of vector, 0th entry is indices for 1st round and so on)
    Z_det_inds_per_rd: indices of Z type detectors per round (vector of vector, 0th entry is indices for 1st round and so on)

    Output:
    p_time_Xdets: probability dictionary of X-type time edges
    p_time_Zdets: probability dictionary of Z-type time edges
    */    

    ProbDict p_time_Zdets;
    ProbDict p_time_Xdets;
    std::vector<int> lower_order_key(2);
    
    // X-type time edges                                                
    for (size_t k=0; k<X_det_inds_per_rd.size()-1; ++k){

        const auto& dets_this_rd = X_det_inds_per_rd[k];
        const auto& dets_next_rd = X_det_inds_per_rd[k+1];
        
        
        for (size_t m=0; m<dets_next_rd.size(); ++m){

            int indx1 = dets_this_rd[m];
            int indx2 = dets_next_rd[m];

            Real p = get_prob_pij(eigen_batch, vi_mean, indx1, indx2,  nsims);   

            lower_order_key[0] = indx1;
            lower_order_key[1] = indx2;

            p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs); //lower_order_key
            p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs); //lower_order_key

            p_time_Xdets.values[lower_order_key] = p;

        }
    }    
    
    // Z-type time edges
    for (size_t k=0; k<Z_det_inds_per_rd.size()-1; ++k){

        const auto& dets_this_rd = Z_det_inds_per_rd[k];
        const auto& dets_next_rd = Z_det_inds_per_rd[k+1];

        
        for (int m=0; m<dets_next_rd.size(); ++m){

            int indx1 = dets_this_rd[m];
            int indx2 = dets_next_rd[m];

            Real p = get_prob_pij(eigen_batch, vi_mean, indx1, indx2,  nsims);   

            lower_order_key[0] = indx1;
            lower_order_key[1] = indx2;

            p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs); //lower_order_key
            p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs); //lower_order_key

            p_time_Zdets.values[lower_order_key] = p;

        }
    }    
    
    return std::make_tuple(p_time_Xdets,p_time_Zdets);
}

//This is hard-coded for d=3 rotated surface code only
//TODO: Think about which higher-order correlations I need to subtract.
std::tuple<ProbDict,ProbDict> estimate_space_edges(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, const std::vector<Real>& vi_mean, 
                                                                     int nsims,  const ProbDict& four_point_probs, const ProbDict& three_point_probs,
                                                                     const std::vector<std::vector<int>>& X_det_inds_per_rd, const std::vector<std::vector<int>>& Z_det_inds_per_rd){

    /*
    Calculate the probability p_{ij} of space (data) edges. If four_point_probs or three_point_probs is empty, we truncate the estimation
    to 2-point correlators. If those dictionaries are non-empty, we redefine the edge probabilities using higher-order events.

    Input:
    eigen_batch: batch of detection events
    vi_mean: vector of averages <vi>
    nsims: total # of shots
    four_point_probs: probability dictionary of 4-point events
    three_point_probs: probability dictionary of 3-point events
    X_det_inds_per_rd: indices of X type detectors per round (vector of vector, 0th entry is indices for 1st round and so on)
    Z_det_inds_per_rd: indices of Z type detectors per round (vector of vector, 0th entry is indices for 1st round and so on)

    Output:
    p_space_Xdets: probability dictionary for X-type bulk space edges
    p_space_Zdets: probability dictionary for Z-type bulk space edges
    */    

    ProbDict p_space_Xdets;
    ProbDict p_space_Zdets;
    std::vector<int> lower_order_key(2);
    
    //X-type bulk edges within the same time-layer
    for (size_t k=0; k<X_det_inds_per_rd.size(); ++k){

        const auto& temp = X_det_inds_per_rd[k];

        //0-2 edge
        int indx1 = temp[0];
        int indx2 = temp[2];
        
        Real p = get_prob_pij(eigen_batch, vi_mean, indx1, indx2,  nsims);

        lower_order_key[0] = indx1;
        lower_order_key[1] = indx2;
        
        p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs); //lower_order_key
        p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs);


        p_space_Xdets.values[lower_order_key] = p;

        //1-2 edge
        indx1 = temp[1];
        indx2 = temp[2];

        p = get_prob_pij(eigen_batch, vi_mean, indx1, indx2,  nsims);

        lower_order_key[0] = indx1;
        lower_order_key[1] = indx2;
        
        p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs); //lower_order_key
        p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs);

        p_space_Xdets.values[lower_order_key] = p;


        //1-3 edge

        indx1 = temp[1];
        indx2 = temp[3];
        lower_order_key[0] = indx1;
        lower_order_key[1] = indx2;

        p = get_prob_pij(eigen_batch, vi_mean, indx1, indx2,  nsims);
        
        p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs); //lower_order_key
        p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs);

        p_space_Xdets.values[lower_order_key] = p;

    }    

    //Z-type bulk edges within the same time-layer
    for (size_t k=0; k<Z_det_inds_per_rd.size(); ++k){

        const auto& temp = Z_det_inds_per_rd[k];

        //0-2 edge
        int indx1 = temp[0];
        int indx2 = temp[2];
        
        Real p = get_prob_pij(eigen_batch, vi_mean, indx1, indx2,  nsims);

        lower_order_key[0] = indx1;
        lower_order_key[1] = indx2;
        
        p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs); //lower_order_key
        p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs);

        p_space_Zdets.values[lower_order_key] = p;


        //1-2 edge
        indx1 = temp[1];
        indx2 = temp[2];

        p = get_prob_pij(eigen_batch, vi_mean, indx1, indx2,  nsims);

        lower_order_key[0] = indx1;
        lower_order_key[1] = indx2;
        
        p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs); //lower_order_key
        p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs);

        p_space_Zdets.values[lower_order_key] = p;

        //1-3 edge

        indx1 = temp[1];
        indx2 = temp[3];
        lower_order_key[0] = indx1;
        lower_order_key[1] = indx2;

        p = get_prob_pij(eigen_batch, vi_mean, indx1, indx2,  nsims);
        
        p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs); //lower_order_key
        p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs);

        p_space_Zdets.values[lower_order_key] = p;

    }    



    return std::make_tuple(p_space_Xdets,p_space_Zdets);

}

//This will also be hard-coded 
//TODO: Test this also...
// std::vector<Real> estimate_diag_edges(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& eigen_batch, const std::vector<Real>& vi_mean, 
//                                      int nsims, int n_stabs, int rds, const ProbDict& four_point_probs, const ProbDict& three_point_probs){
   
    
//     /*
//     Calculate the probability p_{ij} of diagonal space edges. If four_point_probs or three_point_probs is empty, we truncate the estimation
//     to 2-point correlators. If those dictionaries are non-empty, we redefine the edge probabilities using higher-order events.

//     Input:
//     eigen_batch: batch of detection events
//     vi_mean: vector of averages <vi>
//     nsims: total # of shots
//     nstabs: number of stabilizers
//     rds: total # of rounds (including last data qubit measurement round)
//     four_point_probs: probability dictionary of 4-point events
//     three_point_probs: probability dictionary of 3-point events

//     Output:
//     p_diag: probability dictionary of diagonal space edges 
//     */ 

//     std::vector<Real> p_diag;
//     p_diag.reserve((rds - 1) * (n_stabs - 1));
//     std::vector<int> lower_order_key(2);
    

//     //diagonal @ (t,anc) - (t+1,anc-1)
//     for (int rd1=0; rd1<rds-1; ++rd1){

//         int rd2 = rd1+1;

//         for (int anc1=1; anc1<n_stabs; ++anc1){
            
//             int anc2 = anc1-1;

//             Real p = get_prob_pij(eigen_batch, vi_mean,  indx1,indx2  nsims);
            
//             int indx1 = anc1 + n_stabs * rd1;
//             int indx2 = anc2 + n_stabs * rd2;
//             lower_order_key[0] = indx1;
//             lower_order_key[1] = indx2;

//             p = redefine_lowest_order_prob(lower_order_key, p, four_point_probs);
//             p = redefine_lowest_order_prob(lower_order_key, p, three_point_probs);
//             p_diag.push_back(p);

            
//         }
//     }

//     //This is for the other diagonal:

//     // for (int rd1=0; rd1<rds-1; ++rd1){

//     //     int rd2 = rd1+1;

//     //     for (int anc1=0; anc1<n_stabs-1; ++anc1){
            
//     //         int anc2 = anc1+1;

//     //         Real p = get_prob_pij(eigen_batch, vi_mean,  rd1,  rd2,  anc1,  anc2,  nsims,n_stabs);
            
//     //         // p_diag.push_back(p);
//     //         std::cout << "Other diag:" << p << "\n";
//     //     }
//     // }    

//     return p_diag;

// }

//TODO: Think about which higher-order 3 and 4 pnt correlators need to be subtracted...
std::tuple<ProbDict,ProbDict> estimate_bd_edges(const std::vector<Real>& vi_mean,  const ProbDict& four_point_probs, const ProbDict& three_point_probs, 
                                                const ProbDict& p_space_Xdets, const ProbDict& p_space_Zdets, const ProbDict& p_time_Xdets, const ProbDict& p_time_Zdets,
                                                const ProbDict& p_diag_Xdets, const ProbDict& p_diag_Zdets,
                                                const std::vector<std::vector<int>>& X_det_inds_per_rd, const std::vector<std::vector<int>>& Z_det_inds_per_rd){
    
    /*
    Calculate the probability p_{ii} of boundary space edges. If four_point_probs or three_point_probs is empty, we truncate the estimation
    to 2-point correlators. If those dictionaries are non-empty, we redefine the edge probabilities using higher-order events.

    Input:
    vi_mean: vector of averages <vi>
    four_point_probs: probability dictionary of 4-point events
    three_point_probs: probability dictionary of 3-point events
    p_space_Xdets: 
    p_space_Zdets:
    p_time_Xdets:
    p_time_Zdets:
    p_diag_Xdets:
    p_diag_Zdets:
    X_det_inds_per_rd:
    Z_det_inds_per_rd:
    

    Output:
    p_bd: probability dictionary of boundary space edges 
    */     
    
    ProbDict p_bd_Xdets;
    ProbDict p_bd_Zdets;

    for (size_t k=0; k<X_det_inds_per_rd.size(); ++k){

        const auto& temp = X_det_inds_per_rd[k];
        
        //For d=3 rotated surface code, we have 4 boundary edges for X DEM (= the number of X type detectors)
        
        for (size_t l=0; l<temp.size(); ++l){

            int indx1 = temp[l];
            Real denom = 1.0;

            //Subtract nearest X-type space bulk edge
            int cnt=0;
            for (const auto& [higher_order_key, higher_order_val] : p_space_Xdets.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;
                    //There are up to 2 so skip counting
                    cnt+=1;
                    if (cnt==2){
                        break;
                    }


                }
            }

            //Subtract nearest X-type time edge
            cnt=0;
            for (const auto& [higher_order_key, higher_order_val] : p_time_Xdets.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;

                    cnt+=1;
                    if (cnt==2){
                        break;
                    }

                }
            }

            //Subtract nearest X-type diagonal edge (space-time)
            for (const auto& [higher_order_key, higher_order_val] : p_diag_Xdets.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;

                }
            }

            //Subtract nearest 4-point events
            for (const auto& [higher_order_key, higher_order_val] : four_point_probs.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;

                }
            }

            for (const auto& [higher_order_key, higher_order_val] : three_point_probs.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;

                }
            }

            //Calculate prob
            Real p = 0.5 + (vi_mean[indx1]-0.5)/denom ;

            p_bd_Xdets.values[{indx1}] = p;            

        }


    }    
    
    for (size_t k=0; k<Z_det_inds_per_rd.size(); ++k){

        const auto& temp = Z_det_inds_per_rd[k];
        
        //For d=3 rotated surface code, we have 4 boundary edges for X DEM (= the number of X type detectors)
        
        for (size_t l=0; l<temp.size(); ++l){

            int indx1 = temp[l];
            Real denom = 1.0;

            //Subtract nearest Z-type space bulk edge
            int cnt=0;
            for (const auto& [higher_order_key, higher_order_val] : p_space_Zdets.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;
                    cnt+=1;
                    if (cnt==2){
                        break;
                    }

                }
            }

            //Subtract nearest Z-type time edge
            cnt=0;
            for (const auto& [higher_order_key, higher_order_val] : p_time_Zdets.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;
                    cnt+=1;
                    if (cnt==2){
                        break;
                    }

                }
            }

            //Subtract nearest Z-type diagonal edge (space-time)
            for (const auto& [higher_order_key, higher_order_val] : p_diag_Zdets.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;

                }
            }

            //Subtract nearest 4-point events
            for (const auto& [higher_order_key, higher_order_val] : four_point_probs.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;

                }
            }

            for (const auto& [higher_order_key, higher_order_val] : three_point_probs.values) {

                if (contains(indx1, higher_order_key)){

                    denom *= 1.0 - 2.0 * higher_order_val;

                }
            }
            
            //Calculate prob
            Real p = 0.5 + (vi_mean[indx1]-0.5)/denom ;

            p_bd_Zdets.values[{indx1}] = p;            

        }


    }       

    return std::make_tuple(p_bd_Xdets,p_bd_Zdets);

}


std::tuple<ProbDictXZ,ProbDictXZ,ProbDictXZ,ProbDictXZ> estimate_edges_surf_code(const std::vector<std::vector<uint8_t>>& batch, int d, int n_anc, int rds, 
                                                                                 bool include_higher_order, bool print_higher_order){
    
    /*
    Estimate edge/hyper-edge probabilities for a d=3 rotated surface code.

    Input:
    batch: array of detection events (nsims x num_dets)
    n_anc: total # of ancilla per round (both X and Z, i.e., for d=3, it will be 4 or 8)
    rds: total # of rounds (including the last measurement of data qubits)
    include_higher_order: 0/1 to calculate up to 3rd or 4th order event probabilities
    print_higher_order: 0/1 to print the higher order probabilities

    Output:
    p_space: probability dictionary of data edges (access as p_bd["X"] and p_bd["Z"])
    p_time: probability dictionary of time edges (access as p_bd["X"] and p_bd["Z"])
    p_bd: probability dictionary of boundary edges for X and Z (access as p_bd["X"] and p_bd["Z"])
    p_diag: probability dictionary of diagonal edges (access as p_bd["X"] and p_bd["Z"])
    */

    std::vector<int> X_det_inds;
    std::vector<int> Z_det_inds;
    std::vector<std::vector<int>> X_det_inds_per_rd;
    std::vector<std::vector<int>> Z_det_inds_per_rd;

    std::tie(X_det_inds,Z_det_inds,X_det_inds_per_rd,Z_det_inds_per_rd)= build_X_and_Z_detector_indices(n_anc,rds); //rds is effective (i.e., including last data qubit measurements)
                                               
    int nsims = batch.size();
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> eigen_batch = cast_as_Eigen_batch(batch); 
    
    std::vector<Real>  vi_mean = calculate_vi_mean(eigen_batch, nsims); //n_stabs and rds should be the total

    ProbDict three_point_probs;
    ProbDict four_point_probs; //Use  empty struct if we do not want to redefine the probs to include higher order events.

    if (include_higher_order){

        // four_point_probs  = get_all_four_point_probs(eigen_batch, vi_mean,  nsims,  n_stabs,  rds);
        // three_point_probs = get_all_three_point_probs(eigen_batch, vi_mean,  nsims,  n_stabs,  rds, four_point_probs);
        
        throw std::invalid_argument("Cannot handle right now higher-order correlations. Need to be updated.");

        if (print_higher_order){

            for (const auto& [mask, val] : three_point_probs.values) {
                std::cout << "(";
                for (size_t i = 0; i < mask.size(); ++i) {
                    std::cout << mask[i];
                    if (i != mask.size() - 1) std::cout << ",";
                }
                std::cout << ") : "  << val << "\n";
            }  
            
            for (const auto& [mask, val] : four_point_probs.values) {
                std::cout << "(";
                for (size_t i = 0; i < mask.size(); ++i) {
                    std::cout << mask[i];
                    if (i != mask.size() - 1) std::cout << ",";
                }
                std::cout << ") : "  << val << "\n";
            }  


        }

    }
    
    ProbDict p_space_Xdets;
    ProbDict p_space_Zdets;
    ProbDict p_time_Xdets;
    ProbDict p_time_Zdets;
    ProbDict p_diag_Xdets;
    ProbDict p_diag_Zdets;
    ProbDict p_bd_Xdets;
    ProbDict p_bd_Zdets;

    std::tie(p_space_Xdets,p_space_Zdets) = estimate_space_edges(eigen_batch, vi_mean, nsims,  four_point_probs, three_point_probs, X_det_inds_per_rd, Z_det_inds_per_rd);
    std::tie(p_time_Xdets,p_time_Zdets)   = estimate_time_edges(eigen_batch,  vi_mean,  nsims, four_point_probs, three_point_probs, X_det_inds_per_rd, Z_det_inds_per_rd);
    // std::tie(p_diag_Xdets,p_diag_Zdets)   = estimate_diag_edges(eigen_batch, vi_mean, nsims, n_stabs, rds,four_point_probs,three_point_probs);
    std::tie(p_bd_Xdets,p_bd_Zdets)       = estimate_bd_edges(vi_mean,  four_point_probs, three_point_probs, p_space_Xdets, p_space_Zdets, p_time_Xdets, p_time_Zdets,
                                                              p_diag_Xdets, p_diag_Zdets, X_det_inds_per_rd,  Z_det_inds_per_rd);
    

    ProbDictXZ p_space;
    p_space["X"] = p_space_Xdets;
    p_space["Z"] = p_space_Zdets;
    ProbDictXZ p_time;        
    p_time["X"] = p_time_Xdets;
    p_time["Z"] = p_time_Zdets;
    ProbDictXZ p_bd;   
    p_bd["X"] = p_bd_Xdets;
    p_bd["Z"] = p_bd_Zdets;
    ProbDictXZ p_diag;   
    p_diag["X"] = p_diag_Xdets;
    p_diag["Z"] = p_diag_Zdets;
                
    return std::make_tuple(p_space,p_time,p_bd,p_diag);

}











