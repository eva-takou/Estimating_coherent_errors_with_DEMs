#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "PrecisionOfTypes.h"
#include <cmath> 
namespace py = pybind11;


std::vector<std::vector<int>> decode_batch_with_pymatching(const std::vector<std::vector<int>>& H, const std::vector<std::vector<uint8_t>>& batch, int repetitions){

                                                                
    /*
    Decode with default weights =1 (up to phenomenological) given the parity check matrix H.

    Input:
    H: parity check matrix
    batch: the batch of detection events
    repetitions: number of rounds 

    Output:
    the corrections per qubit.
    */                                                              


    // py::gil_scoped_acquire gil;


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




std::vector<std::vector<int>> decode_with_pymatching_create_graph(const std::vector<std::vector<int>>& H, 
                                                                  const std::vector<Real>& space_prob, 
                                                                  const std::vector<Real>& time_prob,
                                                                  const std::vector<Real>& diag_prob,
                                                                  const std::vector<std::vector<uint8_t>>& batch,int rds, 
                                                                  int include_stab_reconstruction){
   
    /*
    Create a DEM based on the estimated weights. This has been used for the repetition code DEM.

    Inputs:
    H: Parity check matrix of repetition code
    space_prob: array of probabilities of space bulk (data) edges, ordered per round
    time_prob: array of probabilities of time-like edges, ordered per round
    diag_prob: array of probabilities of diagonal edges, ordered per round
    batch: the syndromes we want to decode 
    rds: the # of QEC rounds
    include_stab_reconstruction: whether to include the last measurement round of data qubits
    
    Output:
    The qubit corrections
    */

    if (include_stab_reconstruction==1){
        rds+=1;
    }                                           


    //Import Matching
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
    size_t n_anc     = H.size();               // number of anc  qubits
    size_t d         = H[0].size();           // number of data qubits
    int virtual_node = rds*n_anc;            // We can use just one virtual node 
    

    //Add the time edges: they are ordered per round (no fault id for time edges)

    int cnt=0;
    for (int rd1=0; rd1<rds-1; ++rd1){

        int rd2 = rd1+1;

        for (int anc1=0; anc1<n_anc; ++anc1){
            
            int anc2 = anc1;

            int indx1 = anc1 + n_anc * (rd1);
            int indx2 = anc2 + n_anc * (rd2);
            
            Real p =  time_prob[cnt];
            
            if (p<1e-20){ p = 1e-20; //If p is 0, then we need to put a very high weight value.
            }

            
            decoder.attr("add_edge")(py::arg("node1")=indx1,
                            py::arg("node2")=indx2,
                            py::arg("weight")=std::log((1.0 - p) / p));

            cnt+=1;
        }
    }    

    
    //Add the diagonal edges (t,anc) - (t+1,anc-1) (need to put a fault id)
    
    cnt=0;
    for (int rd1=0; rd1<rds-1; ++rd1){

        int rd2 = rd1+1;

        for (int anc1=1; anc1<n_anc; ++anc1){
            
            int anc2 = anc1-1;

            int indx1 = anc1 + n_anc * (rd1);
            int indx2 = anc2 + n_anc * (rd2);

            int fault_cnt = anc1; 


            Real p = diag_prob[cnt];

            
            if (p<1e-20){ p = 1e-20; //If p is 0, then we need to put a very high weight value.
            }

            
            decoder.attr("add_edge")(py::arg("node1")=indx1,
                            py::arg("node2")=indx2,
                            py::arg("qubit_id")=py::set(py::make_tuple(fault_cnt)),
                            py::arg("weight")=std::log((1.0 - p) / p));

            cnt+=1;
            
        }
    }    

    

    //Add the boundary edges 


    int fault_cnt = 0;
    int anc1      = 0;
    for (int rd1=0; rd1<rds; ++rd1){
        
        int indx1 = anc1 + n_anc * (rd1);
        
        Real p = space_prob[rd1*(d)]; //This is correct.
        
        if (p<1e-20){ p = 1e-20; //If p is 0, then we need to put a very high weight value.
        }

        
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("qubit_id")=py::set(py::make_tuple(fault_cnt)), 
                        py::arg("weight")=std::log((1.0 - p) / p));


    }
    
    

    fault_cnt = d-1;
    anc1      = n_anc-1;
    for (int rd1=0; rd1<rds; ++rd1){
        
        int indx1 = anc1 + n_anc * (rd1);
        
        Real p = space_prob[anc1+1+rd1*(d)]; //This is correct.
        
        if (p<1e-20){ 
            p = 1e-20; //If p is 0, then we need to put a very high weight value.
        }

        
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("qubit_id")=py::set(py::make_tuple(fault_cnt)), 
                        py::arg("weight")=std::log((1.0 - p) / p));



    }
    
    
    
    //Add the bulk edges
    
    for (int rd1=0; rd1<rds; ++rd1){
        int rd2=rd1;
        

        for (int anc1=0; anc1<n_anc-1; ++anc1){

            int fault_cnt = anc1 + 1; //goes from 1 ... d-2
            
            int anc2  = anc1+1;
            int indx1 = anc1 + n_anc * (rd1);
            int indx2 = anc2 + n_anc * (rd2);
            

            Real p = space_prob[anc1+1 + rd1*d];
            
            if (p<1e-20){
                 p = 1e-20; //If p is 0, then we need to put a very high weight value.
            }


            decoder.attr("add_edge")(py::arg("node1")=indx1,
                            py::arg("node2")=indx2,
                            py::arg("qubit_id")=py::set(py::make_tuple(fault_cnt)), 
                            py::arg("weight")=std::log((1.0 - p) / p));
         
        }
        

    }


    py::set boundary_nodes;
    boundary_nodes.add(virtual_node);
    decoder.attr("set_boundary_nodes")(py::arg("nodes") = boundary_nodes); 

    // Convert from uint8 to int batch and decode
    
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



