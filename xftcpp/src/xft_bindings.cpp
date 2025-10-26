#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/status_casters.h"

namespace nb = nanobind;

NB_MODULE(_xft_core, m) {
  m.doc() = "XFT core Python bindings";

  // Status casters are now automatically registered when including status_casters.h
  m.def("hello", []() { return "Hello from XFT!"; });
  
  m.attr("__version__") = "0.1.0";
}
