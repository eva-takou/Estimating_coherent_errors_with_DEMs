#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Monte_Carlo_Sim_surf_code.h"  

namespace py = pybind11;

PYBIND11_MODULE(sample_surface_code, m) {
    m.doc() = "Wrapper for simulating surface code";     


    m.def("get_LER_from_uniform_DEM_code_capacity_level",
          &get_LER_from_uniform_DEM_code_capacity_level,
          py::arg("d"),
          py::arg("rds"),
          py::arg("ITERS"),
          py::arg("theta_data"),
          py::arg("q_readout"),
          py::arg("Reset_ancilla"),
          "Run phenom memory QEC sim and return LER from uniform DEM");         


}




