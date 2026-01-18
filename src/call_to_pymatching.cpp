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


std::tuple<std::vector<int>,std::vector<int>,std::vector<std::vector<int>>,std::vector<std::vector<int>>> build_X_and_Z_detector_indices(int n_anc, int rds_eff){
    //

    // Inputs:
    // n_anc: total number of ancilla qubits (X and Z)
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

    //Now, we want to add the space edges. Need to inspect each column of H and see if 2 checks, check the same qubit
    //TODO: Will be useful to pass a transposed pcm so that we can pick the rows 

    //Add bulk space edges manually for now
    //TODO: Should probably be usefull to store all probabilities in the form of a dictionary.

    for (int k=0; k<X_det_inds.size(); ++k){
        std::cout << "X_INDS:" << X_det_inds[k] << "\n";

    }

    for (int k=0; k<Z_det_inds.size(); ++k){
        std::cout << "Z_INDS:" << Z_det_inds[k] << "\n";

    }    

    for (int k=0; k<X_det_inds_per_rd.size(); ++k){

        std::vector<int> temp = X_det_inds_per_rd[k];
        
        for (int l=0; l<temp.size(); ++l){
            std::cout << "INDS:" << temp[l] << "\n";
        }

    }

    return std::make_tuple(X_det_inds,Z_det_inds,X_det_inds_per_rd,Z_det_inds_per_rd);

}


//Break into X and Z DEM independent DEMs.
//Configured -- hardcoded graph connectivity only for d=3 matching graph.
//TODO: Think how to order probabilities, and if we will break them independently.
//Might be good to build "dictionaries" to be able to access the elements based on detector indices.
//TODO: Need to add diagonal edges besides the space-edges.
std::vector<std::vector<int>> decode_with_pymatching_create_graph_for_sc_XZ(const std::vector<std::vector<int>>& H, 
                                                                  const std::vector<Real>& space_prob, 
                                                                  const std::vector<Real>& time_prob,
                                                                  const std::vector<Real>& diag_prob,
                                                                  const std::vector<std::vector<uint8_t>>& batch,int rds, 
                                                                  int include_stab_reconstruction){
   
    /*
    Create a DEM based on the estimated weights for a rotated surface code. This is for XZ decoding
    i.e., we separate into independent X and Z checks. (We do not exploit Y correlations)

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

    int rds_eff = rds;                                                                    
    if (include_stab_reconstruction==1){
        rds_eff+=1; }                                           

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

    //Note we always pass the full H matrix, even for rds=2, but we never build a Z-type matching graph.

    py::object decoder = Matching();
    size_t n_anc     = H.size();      // number of anc  qubits
    size_t n_data    = H[0].size();   // number of data qubits
    int virtual_node = virtual_node = (rds_eff*n_anc/2 + (rds_eff-2)*n_anc/2);    //If rds_eff = 2, then 2*4 = 8, so virtual node is correct.

    //Get the detector indices:
    std::vector<int> X_det_inds;
    std::vector<int> Z_det_inds;
    std::vector<std::vector<int>> X_det_inds_per_rd;
    std::vector<std::vector<int>> Z_det_inds_per_rd;
    std::tie(X_det_inds,Z_det_inds,X_det_inds_per_rd,Z_det_inds_per_rd) =  build_X_and_Z_detector_indices(n_anc, rds_eff);
    
    //X-type time edges
    int cnt=0;
    for (int k=0; k<X_det_inds_per_rd.size()-1; ++k){
        
        std::vector<int> dets_this_rd = X_det_inds_per_rd[k];
        std::vector<int> dets_next_rd = X_det_inds_per_rd[k+1];
        
        for (int m=0; m<dets_next_rd.size(); ++m){

            int indx1 = dets_this_rd[m];
            int indx2 = dets_next_rd[m];

            std::cout << "X Time edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

            Real p =  time_prob[cnt];
            if (p<1e-20){ p = 1e-20; }

            decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));     

            cnt+=1;

        }
    }
    std::cout << "Added all X time edges w/o a problem" << "\n";
    std::cout << "Size of Z_det_inds_per_rd:" << Z_det_inds_per_rd.size() << "\n";
    //Z-type time edges
    if (rds_eff>2){
        cnt=0;
        for (int k=0; k<Z_det_inds_per_rd.size()-1; ++k){
            
            std::vector<int> dets_this_rd = Z_det_inds_per_rd[k];
            std::vector<int> dets_next_rd = Z_det_inds_per_rd[k+1];
            
            for (int m=0; m<dets_next_rd.size(); ++m){

                int indx1 = dets_this_rd[m];
                int indx2 = dets_next_rd[m];

                std::cout << "Z Time edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

                Real p =  time_prob[cnt];
                if (p<1e-20){ p = 1e-20; }

                decoder.attr("add_edge")(py::arg("node1")=indx1,
                                        py::arg("node2")=indx2,
                                        py::arg("weight")=std::log((1.0 - p) / p));     

                cnt+=1;

            }    
        }
    }

    std::cout << "Added all time edges w/o a problem" << "\n";
    
    //Xtype edges within the same time layer
    //Putting the fault ID as a vertical for the middle qubits (3,4,5) for the X_L operator
    for (int k=0; k<X_det_inds_per_rd.size(); ++k){

        std::vector<int> temp = X_det_inds_per_rd[k];

        //Apply the connectivity according to bulk edges for each layer
        int indx1 = temp[0];
        int indx2 = temp[2];
        Real p    = space_prob[0]; 

        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("qubit_id")=py::set(py::make_tuple(0)), //FAULT-ID
                                    py::arg("weight")=std::log((1.0 - p) / p));

        indx1 = temp[1];
        indx2 = temp[2];

        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";
        

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("qubit_id")=py::set(py::make_tuple(1)), //FAULT-ID
                                    py::arg("weight")=std::log((1.0 - p) / p));                                    

        indx1 = temp[1];
        indx2 = temp[3];

        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";
        

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("qubit_id")=py::set(py::make_tuple(2)), //FAULT-ID
                                    py::arg("weight")=std::log((1.0 - p) / p));        

        //Boundary qubit checked by weight 2
        indx1 = temp[0];                                                

        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << virtual_node << "\n";
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("weight")=std::log((1.0 - p) / p));                                    

        //Boundary qubit checked by weight 2                       
        indx1 = temp[3];                                
        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << virtual_node << "\n";                
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("weight")=std::log((1.0 - p) / p));       

        //2 Boundary qubits checked by weight 4    
        indx1 = temp[1];                                                                     
        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << virtual_node << "\n";
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("weight")=std::log((1.0 - p) / p));     
        
        //2 Boundary qubits checked by weight 4                                                       
        indx1 = temp[2];                                
        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << virtual_node << "\n";                
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("weight")=std::log((1.0 - p) / p));       


    }

    //Ztype edges within the same time layer
    for (int k=0; k<Z_det_inds_per_rd.size(); ++k){

        std::vector<int> temp = Z_det_inds_per_rd[k];

        //Apply the connectivity according to bulk edges for each layer
        int indx1 = temp[0];
        int indx2 = temp[1];
        Real p    = space_prob[0]; 

        std::cout << "Z Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));

        indx1 = temp[1];
        indx2 = temp[2];

        std::cout << "Z Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";
        

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));                                    

        indx1 = temp[2];
        indx2 = temp[3];

        std::cout << "Z Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";
        

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));        

        //Boundary qubit checked by weight 2
        indx1 = temp[0];                                                

        std::cout << "Z Space edges, indx1: " << indx1 << " indx2: " << virtual_node << "\n";
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("weight")=std::log((1.0 - p) / p));                                    

        //Boundary qubit checked by weight 2                       
        indx1 = temp[3];                                
        std::cout << "Z Space edges, indx1: " << indx1 << " indx2: " << virtual_node << "\n";                
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("weight")=std::log((1.0 - p) / p));       

        //2 Boundary qubits checked by weight 4    
        indx1 = temp[1];                                                                     
        std::cout << "Z Space edges, indx1: " << indx1 << " indx2: " << virtual_node << "\n";
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("weight")=std::log((1.0 - p) / p));     
        
        //2 Boundary qubits checked by weight 4                                                       
        indx1 = temp[2];                                
        std::cout << "Z Space edges, indx1: " << indx1 << " indx2: " << virtual_node << "\n";                
        decoder.attr("add_edge")(py::arg("node1")=indx1,
                        py::arg("node2")=virtual_node,
                        py::arg("weight")=std::log((1.0 - p) / p));       


    }
    
    std::cout << "Added all space edges w/o a problem" << "\n";


    //Add diagonal edges (space-time edges) for X-DEM only
    //Hard-coded for now, only for d=3. Assume errors on both X and Z checks, but check only what is flipped in the X-DEM (aka, comment out Z detectors in the stim's DEM)
    //Found from stim circuit, when we have 2-qubit error after each CNOT
    for (int k=0; k<X_det_inds_per_rd.size()-1; ++k){
        
        std::vector<int> temp         = X_det_inds_per_rd[k];
        std::vector<int> temp_next_rd = X_det_inds_per_rd[k+1];

        //Apply the connectivity according to bulk edges for each layer
        int indx1 = temp[0];
        int indx2 = temp_next_rd[1]; //D0-D5
        Real p    = space_prob[0]; 

        std::cout << "X Diag Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));

        indx1 = temp[0];
        indx2 = temp_next_rd[2];  //D0-D6
        

        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));        

        indx1 = temp[1];
        indx2 = temp_next_rd[3];  //D1-D7
        

        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));     
                                    
                                    
        indx1 = temp[2];
        indx2 = temp_next_rd[1];  //D2-D5
        

        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));        

        indx1 = temp[2];
        indx2 = temp_next_rd[3];  //D2-D7
        

        std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

        decoder.attr("add_edge")(py::arg("node1")=indx1,
                                    py::arg("node2")=indx2,
                                    py::arg("weight")=std::log((1.0 - p) / p));         

                            
    }

    //Add diagonal edges (space-time edges) for Z-DEM only
    //The question is should we put the space-time edges assuming the Z-DEM w/o errors on the X-part (so restricting to what fires in the 
    //Z-type detectors when there is noise everywhere).
    //The following is hard-coded for now, so it's applicable for d=3 only.
    if (rds_eff>2){
        for (int k=0; k<Z_det_inds_per_rd.size()-1; ++k){
            
            std::vector<int> temp         = Z_det_inds_per_rd[k];
            std::vector<int> temp_next_rd = Z_det_inds_per_rd[k+1];

            //Apply the connectivity according to bulk edges for each layer
            int indx1 = temp[0];
            int indx2 = temp_next_rd[1]; //D0-D5
            Real p    = space_prob[0]; 

            std::cout << "Z Diag Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

            decoder.attr("add_edge")(py::arg("node1")=indx1,
                                        py::arg("node2")=indx2,
                                        py::arg("weight")=std::log((1.0 - p) / p));

            indx1 = temp[0];
            indx2 = temp_next_rd[2];  //D0-D6
            

            std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

            decoder.attr("add_edge")(py::arg("node1")=indx1,
                                        py::arg("node2")=indx2,
                                        py::arg("weight")=std::log((1.0 - p) / p));        

            indx1 = temp[1];
            indx2 = temp_next_rd[2];  //D1-D6
            

            std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

            decoder.attr("add_edge")(py::arg("node1")=indx1,
                                        py::arg("node2")=indx2,
                                        py::arg("weight")=std::log((1.0 - p) / p));     
                                        
                                        
            indx1 = temp[1];
            indx2 = temp_next_rd[3];  //D1-D7
            

            std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

            decoder.attr("add_edge")(py::arg("node1")=indx1,
                                        py::arg("node2")=indx2,
                                        py::arg("weight")=std::log((1.0 - p) / p));        

            indx1 = temp[2];
            indx2 = temp_next_rd[3];  //D2-D7
            

            std::cout << "Space edges, indx1: " << indx1 << " indx2: " << indx2 << "\n";

            decoder.attr("add_edge")(py::arg("node1")=indx1,
                                        py::arg("node2")=indx2,
                                        py::arg("weight")=std::log((1.0 - p) / p));         

                                
        }
    }


    py::set boundary_nodes;
    boundary_nodes.add(virtual_node);
    decoder.attr("set_boundary_nodes")(py::arg("nodes") = boundary_nodes); 

    // Convert from uint8 to int and decode
    py::array_t<int> np_batch = py::cast(batch);

    //Here we might need to apply corrections to matched_det_edges?
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