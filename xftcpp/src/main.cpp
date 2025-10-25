#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "Version.h"

namespace nb = nanobind;

// the module definition is now clean and declarative.
// it describes *what* the module contains, not *how* it is implemented.
NB_MODULE(xftcpp, m) {
    m.doc() = "The C++ backend for the xft framework.";
    
    // bind the functions from our other files.
    m.def("get_version", &xftcpp::get_version, 
          "Returns the version of the xftcpp library.");
}