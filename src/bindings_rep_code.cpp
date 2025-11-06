#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Monte_Carlo_Sim_rep_code.h"  


namespace py = pybind11;

PYBIND11_MODULE(sample_outcomes_rep_code, m) {
    m.doc() = "Wrapper for simulating repetition code";

//     m.def("sample_outcomes",
//           &sample_outcomes_MC_Final,
//           py::arg("d"),
//           py::arg("rds"),
//           py::arg("Nsamples"),
//           py::arg("Nsamples_data"),
//           py::arg("nsims"),
//           py::arg("theta_data"),
//           py::arg("theta_anc"),
//           py::arg("theta_G"),
//           py::arg("q_readout"),
//           py::arg("Reset_ancilla"),
//           py::arg("include_stab_reconstruction"),
//           "Run quantum error correction simulation and return results");

//     m.def("sample_outcomes_LER_via_thetaL",
//           &sample_outcomes_MC_Final_LER_based_on_thetaL,
//           py::arg("d"),
//           py::arg("rds"),
//           py::arg("Nsamples"),
//           py::arg("Nsamples_data"),
//           py::arg("nsims"),
//           py::arg("theta_data"),
//           py::arg("theta_anc"),
//           py::arg("theta_G"),
//           py::arg("q_readout"),
//           py::arg("Reset_ancilla"),
//           py::arg("include_stab_reconstruction"),
//           "Run quantum error correction simulation and return results");        


    m.def("get_LER_from_estimated_DEM",
          &get_LER_from_estimated_DEM,
          py::arg("d"),
          py::arg("rds"),
          py::arg("ITERS"),
          py::arg("theta_data"),
          py::arg("theta_anc"),
          py::arg("theta_G"),
          py::arg("q_readout"),
          py::arg("Reset_ancilla"),
          py::arg("include_higher_order"),
          py::arg("print_higher_order"),
          "Run memory QEC sim and return LER from estimated DEM");      
          
    m.def("get_LER_from_uniform_DEM",
          &get_LER_from_uniform_DEM,
          py::arg("d"),
          py::arg("rds"),
          py::arg("ITERS"),
          py::arg("theta_data"),
          py::arg("theta_anc"),
          py::arg("theta_G"),
          py::arg("q_readout"),
          py::arg("Reset_ancilla"),
          "Run memory QEC sim and return LER from uniform DEM");           
         

    m.def("test_return_vectors", []() {
        std::vector<std::vector<int>> result(3);
        result[0] = {1, 2, 3};
        result[1] = {4, 5, 6};
        result[2] = {7, 8, 9};
        return result;
    });

}




