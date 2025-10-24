// src/main.cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>

namespace nb = nanobind;

std::string get_version() {
    return "xftcpp version 0.0.1";
}

NB_MODULE(xftcpp, m) {
    m.doc() = "The C++ backend for the xft framework.";
    m.def("get_version", &get_version, "Returns the version of the xftcpp library.");
}