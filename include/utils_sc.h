#pragma once
#include <vector>
#include <tuple>
#include <unordered_map>  
#include <string>         
#include <stdexcept>      
#include <cstddef>        

#include <map>
#include <cmath>
#include <cstdlib>
#include <iostream> //for std::cout


// inline std::tuple<std::vector<int>,std::vector<int>,std::vector<std::vector<int>>,std::vector<std::vector<int>>> build_X_and_Z_detector_indices(int n_anc, int rds_eff){
//     //Build the X and Z detector indices.

//     // Inputs:
//     // n_anc: total number of ancilla qubits (both X and Z)
//     // rds_eff: total # of QEC rounds (including last stabilizer reconstruction round)


//     //Types of dets:
//     //We start with X, we have another X layer, and then we alternate between X and Z, ending with X
//     //For rd=1 we have [X X X X, X X X X] (rds_eff = 2)
//     //For rd=2 we have [X X X X, X X X X, Z Z Z Z, X X X X] (rds_eff = 3)
//     //For rd=3 we have [X X X X, X X X X, Z Z Z Z, X X X X, Z Z Z Z, X X X X] (rds_eff = 4)
//     //
//     //The Z-type layer appears rds_eff-2 times. The X-type layer appears rds_eff times.
    
//     std::vector<int> X_det_inds; 
//     X_det_inds.reserve(rds_eff*n_anc/2); //# of X dets is (rds_eff)*n_anc/2
//     std::vector<int> Z_det_inds; 
//     Z_det_inds.reserve((rds_eff-2)*n_anc/2); //# of Z dets is (rds_eff-2)*n_anc/2

//     std::vector<std::vector<int>> X_det_inds_per_rd; 
//     X_det_inds_per_rd.reserve(rds_eff);
//     std::vector<std::vector<int>> Z_det_inds_per_rd; 
//     Z_det_inds_per_rd.reserve(rds_eff-2);
    
//     for (int anc=0; anc<n_anc/2; ++anc){
//         X_det_inds.push_back(anc);
//     }

//     X_det_inds_per_rd.push_back(X_det_inds);

//     int shift = X_det_inds.size();
    
//     //Start from when X and Z alternate (rd=1)
//     for (int rd=1; rd<rds_eff-1; ++rd){

//         std::vector<int> tempX;
//         for (int anc=0; anc<n_anc/2; ++anc){
//             X_det_inds.push_back(shift+anc+n_anc*(rd-1));
//             tempX.push_back(shift+anc+n_anc*(rd-1));
//         }

//         X_det_inds_per_rd.push_back(tempX);


//         std::vector<int> tempZ;
//         for (int anc=n_anc/2; anc<n_anc; ++anc){
//             Z_det_inds.push_back(shift + anc+n_anc*(rd-1));
//             tempZ.push_back(shift + anc+n_anc*(rd-1));
//         }

//         Z_det_inds_per_rd.push_back(tempZ);

//     }

//     std::vector<int> tempX;
//     for (int anc=0; anc<n_anc/2; ++anc){
//         X_det_inds.push_back(shift + anc + n_anc*(rds_eff-2));
//         tempX.push_back(shift + anc +n_anc*(rds_eff-2));
//     }

//     X_det_inds_per_rd.push_back(tempX);


//     return std::make_tuple(X_det_inds,Z_det_inds,X_det_inds_per_rd,Z_det_inds_per_rd);

// }

inline std::tuple<std::vector<int>,std::vector<int>,std::vector<std::vector<int>>,std::vector<std::vector<int>>> build_X_and_Z_detector_indices(int n_anc, int rds_eff){
    //Build the X and Z detector indices.

    // Inputs:
    // n_anc: total number of ancilla qubits (both X and Z)
    // rds_eff: total # of QEC rounds (including last stabilizer reconstruction round)

    //Types of dets:
    //We start with X, we have another X layer, and then we alternate between X and Z, ending with X
    //For rd=1 we have [X X X X, Z Z Z Z, X X X X] (rds_eff = 2)
    //For rd=2 we have [X X X X, Z Z Z Z, X X X X, Z Z Z Z, X X X X] (rds_eff = 3)
    //For rd=3 we have [X X X X, Z Z Z Z, X X X X, Z Z Z Z, X X X X, Z Z Z Z, X X X X] (rds_eff = 4)
    //
    //The Z-type layer appears rds_eff-1 times. The X-type layer appears rds_eff times.
    
    std::vector<int> X_det_inds; 
    X_det_inds.reserve(rds_eff*n_anc/2); //# of X dets is (rds_eff)*n_anc/2
    std::vector<int> Z_det_inds; 
    Z_det_inds.reserve((rds_eff-1)*n_anc/2); //# of Z dets is (rds_eff-1)*n_anc/2

    std::vector<std::vector<int>> X_det_inds_per_rd; 
    X_det_inds_per_rd.reserve(rds_eff);
    std::vector<std::vector<int>> Z_det_inds_per_rd; 
    Z_det_inds_per_rd.reserve(rds_eff-1);
    

    //Alternating X and Z dets
    for (int rd=0; rd<rds_eff-1; ++rd){

        std::vector<int> tempX;
        for (int anc=0; anc<n_anc/2; ++anc){
            X_det_inds.push_back(anc+n_anc*(rd));
            tempX.push_back(anc+n_anc*(rd));
        }

        X_det_inds_per_rd.push_back(tempX);


        std::vector<int> tempZ;
        for (int anc=n_anc/2; anc<n_anc; ++anc){
            Z_det_inds.push_back( anc+n_anc*(rd));
            tempZ.push_back( anc+n_anc*(rd));
        }

        Z_det_inds_per_rd.push_back(tempZ);

    }

    std::vector<int> tempX;
    for (int anc=0; anc<n_anc/2; ++anc){
        X_det_inds.push_back( anc + n_anc*(rds_eff-1));
        tempX.push_back( anc +n_anc*(rds_eff-1));
    }

    X_det_inds_per_rd.push_back(tempX);

    return std::make_tuple(X_det_inds,Z_det_inds,X_det_inds_per_rd,Z_det_inds_per_rd);

}

inline std::vector<std::vector<std::vector<int>>> cnot_schedule_for_rsc(int d){
    /*
    Construct the qubit coordinates for a distance d rotated surface code,
    and get a CNOT schedule. Default here is NW, NE, SW, SE for all checks.

    Input:
        d: distance of rotated surface code
    Output:
        a vector of size 4 (containing the 4 CNOT layers), and each one of them contains pairs of qubits [ctrl,trgt]
    */

    int n_data = d*d;
    int n_anc  = (d*d-1);

    //Start with coordingates of data qubits
    int x0 = 0;
    int y0 = 0;
    int shift = 2;

    std::vector<std::vector<int>> data_coords; //All qubits are placed @ integer positions

    for (int row=0 ; row<d ;++row){

        for (int col=0; col<d ; ++col){

            int x = x0 + row * shift;
            int y = y0 - col * shift;
            
            
            data_coords.push_back({x,y});

        }

    }

    //Now get the X-check coordinates
    int shift_v = 4;
    int max_col = 0;
    std::vector<std::vector<int>> X_check_coords;

    for (int k=0; k<d; ++k){

        if (  k %2 == 0 ){
            max_col+=1;
        }

    }

    x0 = 1;

    for (int row=0; row<d-1; ++row){

        int y0;
        if ( row % 2 ==0){
            y0=1;
        }
        else{
            y0=-1;
        }

        for (int col=0; col<max_col; ++col){
            int x = x0 + row * shift;
            int y = y0 - col * shift_v;
            X_check_coords.push_back({x,y});
        }

    }

    //Now get the Z-check coordinates
    std::vector<std::vector<int>> Z_check_coords;
    max_col = 0;

    for (int k=0; k<d; ++k){
        if ( k % 2 ==1){
            max_col+=1;
        }
    }

    x0 = -1;
    for (int row=0 ; row<d+1; ++row){
        int y0;
        if (row %2 ==0){
            y0=-3;
        }
        else{
            y0=-1;
        }

        for (int col=0; col<max_col; ++col){
            int x = x0 + row * shift;
            int y = y0 - col * shift_v;
            Z_check_coords.push_back({x,y});
        }
    }

    //Now get the qubits that are checked by X-checks
    std::map<int, std::vector<int>> X_check_q;
    int cnt = 0;

    for (size_t k=0; k<X_check_coords.size(); ++k){

        std::vector<int> coords_check = X_check_coords[k];
        int x_X = coords_check[0];
        int y_X = coords_check[1];

        int cnt_d = 0;
        std::vector<int> data_qubits;

        for (size_t l=0; l<data_coords.size(); ++l){

            std::vector<int> coords_data = data_coords[l];
            int x_d = coords_data[0];
            int y_d = coords_data[1];

            if (std::abs(static_cast<int>(x_d-x_X)) + std::abs(static_cast<int>(y_d-y_X)) <=2){
                data_qubits.push_back(cnt_d);
            }
            cnt_d+=1;

        }
        X_check_q[cnt] = data_qubits;
        cnt+=1;        

    }

    //Now get the qubits that are checked by Z-checks
    std::map<int, std::vector<int>> Z_check_q;
    cnt = 0;
    for (size_t k=0; k<Z_check_coords.size(); ++k){

        std::vector<int> coords_check = Z_check_coords[k];
        int x_Z = coords_check[0];
        int y_Z = coords_check[1];

        int cnt_d = 0;
        std::vector<int> data_qubits;

        for (size_t l=0; l<data_coords.size(); ++l){

            std::vector<int> coords_data = data_coords[l];
            int x_d = coords_data[0];
            int y_d = coords_data[1];

            if (std::abs(static_cast<int>(x_d-x_Z)) + std::abs(static_cast<int>(y_d-y_Z)) <=2){
                data_qubits.push_back(cnt_d);
            }
            cnt_d+=1;


        }
        Z_check_q[cnt] = data_qubits;
        cnt+=1;

    }

    //Now, do the NW, NE, SW, SE CNOT pattern
    std::vector<std::vector<int>> CNOT_NW; //Contains pairs [ctrl,trgt]
    std::vector<std::vector<int>> CNOT_NE;
    std::vector<std::vector<int>> CNOT_SW;
    std::vector<std::vector<int>> CNOT_SE;
    
    //Start with X-checks (check ctrl, data trgt)
    for (const auto& [anc, all_data] : X_check_q){

        const auto& x_coords = X_check_coords[anc];
        int x_a = x_coords[0];
        int y_a = x_coords[1];

        for (int data : all_data){

            const auto& d_coords = data_coords[data];
            int x_d = d_coords[0];
            int y_d = d_coords[1];

            std::vector<int> pair{anc+n_data,data};
            
            if (x_d < x_a and y_d > y_a){

                CNOT_NW.push_back(pair);
            }

            if (x_d>x_a and y_d>y_a){
                CNOT_NE.push_back(pair);
            }

            if (x_d > x_a and y_d < y_a){

                CNOT_SE.push_back(pair);
            }

            if (x_d < x_a and y_d < y_a){
                CNOT_SW.push_back(pair);
            }


        }

        

    }

    //Do the same for the Z-checks (data ctrl, check trgt)
    for (const auto& [anc, all_data] : Z_check_q){

        const auto& x_coords = Z_check_coords[anc];
        int x_a = x_coords[0];
        int y_a = x_coords[1];

        for (int data : all_data){

            const auto& d_coords = data_coords[data];
            int x_d = d_coords[0];
            int y_d = d_coords[1];

            std::vector<int> pair{data, anc + n_data + n_anc/2 };
            
            if (x_d < x_a and y_d > y_a){

                CNOT_NW.push_back(pair);
            }

            if (x_d>x_a and y_d>y_a){
                CNOT_NE.push_back(pair);
            }

            if (x_d > x_a and y_d < y_a){

                CNOT_SE.push_back(pair);
            }

            if (x_d < x_a and y_d < y_a){
                CNOT_SW.push_back(pair);
            }


        }

        

    }

    std::vector<std::vector<std::vector<int>>> CNOT_layers{CNOT_NW,CNOT_NE,CNOT_SW,CNOT_SE};

    return CNOT_layers;
}


struct VectorHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t h = 0;
        for (int x : v) {
            h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct ProbDict {
    std::unordered_map<std::vector<int>, double, VectorHash> values;
};

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