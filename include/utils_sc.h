#pragma once
#include <vector>
#include <tuple>

inline std::tuple<std::vector<int>,std::vector<int>,std::vector<std::vector<int>>,std::vector<std::vector<int>>> build_X_and_Z_detector_indices(int n_anc, int rds_eff){
    //Build the X and Z detector indices.

    // Inputs:
    // n_anc: total number of ancilla qubits (both X and Z)
    // rds_eff: total # of QEC rounds (including last stabilizer reconstruction round)


    //Types of dets:
    //We start with X, we have another X layer, and then we alternate between X and Z, ending with X
    //For rd=1 we have [X X X X, X X X X] (rds_eff = 2)
    //For rd=2 we have [X X X X, X X X X, Z Z Z Z, X X X X] (rds_eff = 3)
    //For rd=3 we have [X X X X, X X X X, Z Z Z Z, X X X X, Z Z Z Z, X X X X] (rds_eff = 4)
    //
    //The Z-type layer appears rds_eff-2 times. The X-type layer appears rds_eff times.
    
    std::vector<int> X_det_inds; 
    X_det_inds.reserve(rds_eff*n_anc/2); //# of X dets is (rds_eff)*n_anc/2
    std::vector<int> Z_det_inds; 
    Z_det_inds.reserve((rds_eff-2)*n_anc/2); //# of Z dets is (rds_eff-2)*n_anc/2

    std::vector<std::vector<int>> X_det_inds_per_rd; 
    X_det_inds_per_rd.reserve(rds_eff);
    std::vector<std::vector<int>> Z_det_inds_per_rd; 
    Z_det_inds_per_rd.reserve(rds_eff-2);
    
    for (int anc=0; anc<n_anc/2; ++anc){
        X_det_inds.push_back(anc);
    }

    X_det_inds_per_rd.push_back(X_det_inds);

    int shift = X_det_inds.size();
    
    //Start from when X and Z alternate (rd=1)
    for (int rd=1; rd<rds_eff-1; ++rd){

        std::vector<int> tempX;
        for (int anc=0; anc<n_anc/2; ++anc){
            X_det_inds.push_back(shift+anc+n_anc*(rd-1));
            tempX.push_back(shift+anc+n_anc*(rd-1));
        }

        X_det_inds_per_rd.push_back(tempX);


        std::vector<int> tempZ;
        for (int anc=n_anc/2; anc<n_anc; ++anc){
            Z_det_inds.push_back(shift + anc+n_anc*(rd-1));
            tempZ.push_back(shift + anc+n_anc*(rd-1));
        }

        Z_det_inds_per_rd.push_back(tempZ);

    }

    std::vector<int> tempX;
    for (int anc=0; anc<n_anc/2; ++anc){
        X_det_inds.push_back(shift + anc + n_anc*(rds_eff-2));
        tempX.push_back(shift + anc +n_anc*(rds_eff-2));
    }

    X_det_inds_per_rd.push_back(tempX);


    return std::make_tuple(X_det_inds,Z_det_inds,X_det_inds_per_rd,Z_det_inds_per_rd);

}

struct ProbDictXZ {
    ProbDict X;
    ProbDict Z;

    ProbDict& operator[](const std::string& type) {
        if (type == "X") return X;
        if (type == "Z") return Z;
        throw std::invalid_argument("Invalid type, must be 'X' or 'Z'");
    }

    const ProbDict& operator[](const std::string& type) const {
        if (type == "X") return X;
        if (type == "Z") return Z;
        throw std::invalid_argument("Invalid type, must be 'X' or 'Z'");
    }
};