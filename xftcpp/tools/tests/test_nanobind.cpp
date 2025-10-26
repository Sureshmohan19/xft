#include <nanobind/nanobind.h>
namespace nb = nanobind;

NB_MODULE(_test_nanobind, m) {
    m.def("ping", []() { return "pong"; });
}
