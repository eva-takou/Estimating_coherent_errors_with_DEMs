#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Monte_Carlo_Sim_rep_code.h"  


namespace py = pybind11;

PYBIND11_MODULE(sample_repetition_code, m) {
    m.doc() = "Wrapper for simulating repetition code";     


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
          "Run circuit-level memory QEC sim and return LER from estimated DEM");      
          
    m.def("get_LER_from_uniform_DEM_circuit_level",
          &get_LER_from_uniform_DEM_circuit_level,
          py::arg("d"),
          py::arg("rds"),
          py::arg("ITERS"),
          py::arg("theta_data"),
          py::arg("theta_anc"),
          py::arg("theta_G"),
          py::arg("q_readout"),
          py::arg("Reset_ancilla"),
          "Run circuit-level memory QEC sim and return LER from uniform DEM");           

    m.def("get_LER_from_uniform_DEM_phenom_level",
          &get_LER_from_uniform_DEM_phenom_level,
          py::arg("d"),
          py::arg("rds"),
          py::arg("ITERS"),
          py::arg("theta_data"),
          py::arg("theta_anc"),
          py::arg("q_readout"),
          py::arg("Reset_ancilla"),
          "Run phenom memory QEC sim and return LER from uniform DEM");         

    m.def("get_logical_infidelity",
          &get_logical_infidelity,
          py::arg("d"),
          py::arg("rds"),
          py::arg("ITERS"),
          py::arg("theta_data"),
          py::arg("q_readout"),
          py::arg("Reset_ancilla"),
          "Run code-capacity + classical readout errors memory QEC sim and return LER based on logical angle");                   
          


}




