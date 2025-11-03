// #include "call_to_pymatching.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "PrecisionOfTypes.h"
#include <cmath>  // for std::log
namespace py = pybind11;

//Decode w/ the default all =1 weights
std::vector<std::vector<int>> decode_batch_with_pymatching(const std::vector<std::vector<int>>& H,
                                                               const std::vector<std::vector<uint8_t>>& batch,
                                                               int repetitions){
   
    // Ensure Python is initialized
    // if (!Py_IsInitialized()) {
    //     Py_Initialize();
    // }

    py::gil_scoped_acquire gil;


    // Static Matching object (imported once)
    static py::object Matching;
    static bool matching_initialized = false;

    if (!matching_initialized) {
        try {
            py::module_ pymatching = py::module_::import("pymatching");
            Matching = pymatching.attr("Matching");
            // matching_initialized = true;
        } catch (py::error_already_set& e) {
            std::cerr << "Failed to import pymatching: " << e.what() << std::endl;
            return {};
        }
    }

    // Create decoder
    py::object decoder;
    try {
        decoder = Matching(py::cast(H), py::arg("repetitions") = repetitions);
    } catch (py::error_already_set& e) {
        std::cerr << "Error constructing Matching: " << e.what() << std::endl;
        return {};
    }

    // Convert batch and decode
    py::array_t<int> np_batch = py::cast(batch);


    py::object corrections;
    try {
        corrections = decoder.attr("decode_batch")(np_batch);
    } catch (py::error_already_set& e) {
        std::cerr << "Error during decode_batch: " << e.what() << std::endl;
        return {};
    }

    // Cast and return result
    try {
        return corrections.cast<std::vector<std::vector<int>>>();
    } catch (const std::exception& e) {
        std::cerr << "Error converting result: " << e.what() << std::endl;
        return {};
    }
}








//Decode by creating our own decoding graph based on the parity check matrix, the input space-like and time-like weights
//(Note: if there are diagonal edges we need to add those too, and add a new weights argument).
std::vector<std::vector<int>> decode_with_pymatching_create_graph(const std::vector<std::vector<int>>& H, 
                                                                  const std::vector<Real>& space_prob, 
                                                                  const std::vector<Real>& time_prob,
                                                                  const std::vector<std::vector<uint8_t>>& batch,int repetitions){
   
    // Ensure Python is initialized
    // if (!Py_IsInitialized()) {
    //     Py_Initialize();
    // }

    // Static Matching object (imported once)
    static py::object Matching;
    static bool matching_initialized = false;

    if (!matching_initialized) {
        try {
            py::module_ pymatching = py::module_::import("pymatching");
            Matching = pymatching.attr("Matching");
            matching_initialized = true;
        } catch (py::error_already_set& e) {
            std::cerr << "Failed to import pymatching: " << e.what() << std::endl;
            return {};
        }
    }

    
    
    py::object decoder = Matching();
    //Now add the weights
    size_t n_anc = H.size();                  // number of anc  qubits
    size_t d     = H[0].size();               // number of data qubits
    int virtual_node = repetitions*(d-1); // We can use just one virtual node for all rounds
    

    //If we have k repetitions the # of edges for repetition code is k*d + (d-1)*(k-1).
    //The number of nodes is k*(d-1)
    //Loop through columns. If we see just one non-zero location it is a boundary anc node.
    
    if (space_prob.size() != repetitions * d) {
        std::cerr << "space_prob has size " << space_prob.size()
                << ", but expected " << (repetitions * d) << std::endl;
        return {};
    }
    if (time_prob.size() != n_anc * (repetitions-1)) {
        std::cerr << "time_prob has size " << time_prob.size()
                << ", but expected " << n_anc * (repetitions-1) << std::endl;
        return {};
    }

    
    
    int cnt=0;
    
    //Spacelike weights across rounds
    for (int rd=0; rd<repetitions; ++rd){


        //Which ancillas check a particular data qubit? For given col, find the indices of nnz rows
        int fault_cnt=0;
        for (int col=0; col<d ; ++col){

            //Find how many non-zeros and their locations
            std::vector<int> anc_checking_data;
            for (int row=0; row<n_anc; ++row){

                if (H[row][col]==1){
                    anc_checking_data.push_back(row + rd*n_anc);
                }
            }

            Real p = space_prob[cnt];

            if (std::abs(p)<1e-20){
                p = 1e-20; //If p is 0, then we need to put a very high weight value.
            }

            //fault_cnt counts the qubit in the particular column. i think we do not need to reset it per round.
            if (anc_checking_data.size()>1){


                //Two nodes
                
                decoder.attr("add_edge")(py::arg("node1")=anc_checking_data[0],
                                py::arg("node2")=anc_checking_data[1],
                                py::arg("qubit_id")=fault_cnt, 
                                py::arg("weight")=std::log((1.0 - p) / p));
            }
            else {
                decoder.attr("add_edge")(py::arg("node1")=anc_checking_data[0],
                                py::arg("node2")=virtual_node,
                                py::arg("qubit_id")=fault_cnt,
                                py::arg("weight")=std::log((1.0 - p) / p));


            }
            
            cnt      +=1;
            fault_cnt+=1;

        }

    }


    //Timelike weights across rounds
    cnt=0;
    for (int rd1=0; rd1<(repetitions-1); ++rd1){

        int rd2=rd1+1;
        for (int anc=0; anc<n_anc; anc++){

            int idx_anc1 = anc + rd1 * n_anc;
            int idx_anc2 = anc + rd2 * n_anc;

            Real q = time_prob[cnt];

            if (std::abs(q)<1e-20){
                q = 1e-20; //If q is 0, then we need to put a very high weight value.
            }

            decoder.attr("add_edge")(py::arg("node1")=idx_anc1,
                            py::arg("node2")=idx_anc2,
                            py::arg("weight")=std::log((1.0 - q) / q));
            
            cnt+=1;
            
        }
    }

    //Diagonal edges (t,anc) - (t+1,anc-1)
    // for (int rd1=0; rd1<(repetitions-2); ++rd1){
    //     int rd2 = rd1+1;
    //     for (int anc1=1; anc1<n_anc; ++anc1){

    //         int anc2 = anc1-1;
    //         int idx_anc1 = anc1 + rd1 * n_anc;
    //         int idx_anc2 = anc2 + rd2 * n_anc;

    //         double p = space_prob[0]; //Just assume for now that it is the space prob

    //         //We don't need a fault_id (that was only for the physical qubits)
    //         if (std::abs(p)<1e-20){
    //             p = 1e-20; //If p is 0, then we need to put a very high weight value.
    //         }            

    //         decoder.attr("add_edge")(py::arg("node1")=idx_anc1,
    //                         py::arg("node2")=idx_anc2,
    //                         py::arg("weight")=std::log((1.0 - p) / p));



    //     }

    // }


    py::set boundary_nodes;
    boundary_nodes.add(virtual_node);
    decoder.attr("set_boundary_nodes")(py::arg("nodes") = boundary_nodes); 

    // Convert batch and decode
    py::array_t<int> np_batch = py::cast(batch);
    py::object corrections;
    try {
        corrections = decoder.attr("decode_batch")(np_batch);
    } catch (py::error_already_set& e) {
        std::cerr << "Error during decode_batch: " << e.what() << std::endl;
        return {};
    }

    // Cast and return result
    try {
        return corrections.cast<std::vector<std::vector<int>>>();
    } catch (const std::exception& e) {
        std::cerr << "Error converting result: " << e.what() << std::endl;
        return {};
    }
}


//Here we use the estimated probabilities.
//rds is the regular QEC rounds, excluding potentially the final round.
std::vector<std::vector<int>> decode_with_pymatching_create_graph_V2(const std::vector<std::vector<int>>& H, 
                                                                  const std::vector<Real>& space_prob, 
                                                                  const std::vector<Real>& time_prob,
                                                                  const std::vector<Real>& diag_prob,
                                                                  const std::vector<std::vector<uint8_t>>& batch,int rds, 
                                                                  int include_stab_reconstruction){
   
    if (include_stab_reconstruction==1){
        rds+=1;
    }                                           

    // Ensure Python is initialized
    // if (!Py_IsInitialized()) {
    //     Py_Initialize();
    // }

    // Static Matching object (imported once)
    static py::object Matching;
    static bool matching_initialized = false;

    if (!matching_initialized) {
        try {
            py::module_ pymatching = py::module_::import("pymatching");
            Matching = pymatching.attr("Matching");
            matching_initialized = true;
        } catch (py::error_already_set& e) {
            std::cerr << "Failed to import pymatching: " << e.what() << std::endl;
            return {};
        }
    }

    
    py::object decoder = Matching();
    //Now add the weights
    size_t n_anc = H.size();                  // number of anc  qubits
    size_t d     = H[0].size();               // number of data qubits
    int virtual_node = rds*(d-1);            // We can use just one virtual node for all rounds
    

    //Add the time edges: they are ordered again per round

    // std::cout << "ADDING TIME EDGES TO DECODING GRAPH \n";
    int cnt=0;
    for (int rd1=0; rd1<rds-1; ++rd1){

        int rd2 = rd1+1;

        for (int anc1=0; anc1<n_anc; ++anc1){
            
            int anc2 = anc1;

            int indx1 = anc1 + n_anc * (rd1);
            int indx2 = anc2 + n_anc * (rd2);
            
            Real p =  time_prob[cnt];
            // p=0.1;
            if (p<1e-20){ p = 1e-20; //If p is 0, then we need to put a very high weight value.
            }

            // std::cout<< "INDX1: " << indx1 << ", INDX2: " << indx2 << ", p:" << p << "\n";

            
            decoder.attr("add_edge")(py::arg("node1")=indx1,
                            py::arg("node2")=indx2,
                            py::arg("weight")=std::log((1.0 - p) / p));

            cnt+=1;
        }
    }    

    
    //Add the diagonal edges (t,anc) - (t+1,anc-1):
    // std::cout << "ADDING DIAGONAL EDGES TO DECODING GRAPH \n";
    cnt=0;
    for (int rd1=0; rd1<rds-1; ++rd1){

        int rd2 = rd1+1;

        for (int anc1=1; anc1<n_anc; ++anc1){
            
            int anc2 = anc1-1;

            int indx1 = anc1 + n_anc * (rd1);
            int indx2 = anc2 + n_anc * (rd2);

            //If we have d=3, then anc 0 and anc1
            //The fault id should be 1.

            //If we have d=5, then anc 0 , 1,2,3
            //First fault id should be 1, next should be 2

            int fault_cnt = anc1; 

            

            Real p = diag_prob[cnt];

            // p=0.1;
            
            if (p<1e-20){ p = 1e-20; //If p is 0, then we need to put a very high weight value.
            }

            // std::cout<< "INDX1: " << indx1 << ", INDX2: " << indx2 << ", p:" << p << "\n";

            
            decoder.attr("add_edge")(py::arg("node1")=indx1,
                            py::arg("node2")=indx2,
                            py::arg("qubit_id")=py::set(py::make_tuple(fault_cnt)),
                            py::arg("weight")=std::log((1.0 - p) / p));

            cnt+=1;
            
            
        }
    }    

    

    //Now add the boundaries
    // std::cout << "ADDING BOUNDARIES TO DECODING GRAPH \n";


    int fault_cnt = 0;
    int anc1      = 0;
    for (int rd1=0; rd1<rds; ++rd1){
        
        int indx1 = anc1 + n_anc * (rd1);
        
        Real p = space_prob[rd1*(d)]; //This is correct.
        
        // p=0.1;
        if (p<1e-20){ p = 1e-20; //If p is 0, then we need to put a very high weight value.
        }

        // std::cout<< "INDX1: " << indx1 << ", INDX2: " << virtual_node << ", p:" << p << "\n";

        
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("qubit_id")=py::set(py::make_tuple(fault_cnt)), 
                        py::arg("weight")=std::log((1.0 - p) / p));



    }
    
    // std::cout << "ADDING BOUNDARIES TO DECODING GRAPH \n";

    fault_cnt = d-1;
    anc1      = n_anc-1;
    for (int rd1=0; rd1<rds; ++rd1){
        
        int indx1 = anc1 + n_anc * (rd1);
        
        Real p = space_prob[anc1+1+rd1*(d)]; //This is correct.
        // p=0.1;
        if (p<1e-20){ 
            p = 1e-20; //If p is 0, then we need to put a very high weight value.
        }

        // std::cout<< "INDX1: " << indx1 << ", INDX2: " << virtual_node << ", p:" << p << "\n";

        
        
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("qubit_id")=py::set(py::make_tuple(fault_cnt)), 
                        py::arg("weight")=std::log((1.0 - p) / p));



    }
    
    
    
    // std::cout << "ADDING BULK EDGES TO DECODING GRAPH \n";
    //Add the bulk edges:
    
    for (int rd1=0; rd1<rds; ++rd1){
        int rd2=rd1;
        

        for (int anc1=0; anc1<n_anc-1; ++anc1){

            int fault_cnt = anc1 + 1; //goes from 1 ... d-2
            
            int anc2  = anc1+1;
            int indx1 = anc1 + n_anc * (rd1);
            int indx2 = anc2 + n_anc * (rd2);
            

            Real p = space_prob[anc1+1 + rd1*d];
            // p=0.1;
            if (p<1e-20){
                 p = 1e-20; //If p is 0, then we need to put a very high weight value.
            }

            // std::cout<< "INDX1: " << indx1 << ", INDX2: " << indx2 << ", p:" << p << "\n";

            decoder.attr("add_edge")(py::arg("node1")=indx1,
                            py::arg("node2")=indx2,
                            py::arg("qubit_id")=py::set(py::make_tuple(fault_cnt)), 
                            py::arg("weight")=std::log((1.0 - p) / p));

            
        }
        
        

    }

    


    py::set boundary_nodes;
    boundary_nodes.add(virtual_node);
    decoder.attr("set_boundary_nodes")(py::arg("nodes") = boundary_nodes); 

    // Convert batch and decode

    //COULD WE HAVE A PROBLEM HERE W/ UINT8 THAT WE CAST TO INT FOR PYTHON?
    
    py::array_t<int> np_batch = py::cast(batch);


    py::object corrections;
    try {
        corrections = decoder.attr("decode_batch")(np_batch);
    } catch (py::error_already_set& e) {
        std::cerr << "Error during decode_batch: " << e.what() << std::endl;
        return {};
    }

    // Cast and return result
    try {
        return corrections.cast<std::vector<std::vector<int>>>();
    } catch (const std::exception& e) {
        std::cerr << "Error converting result: " << e.what() << std::endl;
        return {};
    }
}



