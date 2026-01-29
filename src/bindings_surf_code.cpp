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
          py::arg("theta_anc"),
          py::arg("theta_G"),
          py::arg("q_readout"),
          "Run phenom memory QEC sim and return LER from uniform DEM");   
          
          
    m.def("sample_detection_events",
          &sample_detection_events,
          py::arg("rds"),
          py::arg("ITERS"),
          py::arg("theta_data"),
          py::arg("theta_anc"),
          py::arg("theta_G"),
          py::arg("q_readout"),
          "Return the detection events & observable flips after e^{-i\theta Z} and e^{-i\theta ZZ} errors.");             

}




