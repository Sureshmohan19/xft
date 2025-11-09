#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ============================================================================
// XFT Core Module - Python Bindings
// This module will expose Array and its operations to Python users.
// ============================================================================

PYBIND11_MODULE(xft_core, m) {
    m.doc() = "XFT - Deep Learning Framework (Core Module)";
    
    // 1. Version Info
    m.attr("__version__") = "0.0.1";
    m.attr("__author__") = "Suresh Neethimohan";

    // Internal methods that are not exposed to Python users.
    // 1. Scalar class (cpp/xft/scalar.h and cpp/xft/scalar_types.h) will be kept internal - users never interact with it directly.
}