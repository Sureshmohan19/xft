#include "pjrt_plugin.h"

#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include "xla/pjrt/c/pjrt_c_api.h"

namespace xftcpp {

std::string test_pjrt_plugin_load() {
    // --- THE GROUND TRUTH, FORGED IN C++ ---
    // We hardcode the path to the engine you built.
    // This is the keystone of our entire structure.
    const std::string path = "/Users/aakritisuresh/Desktop/xla/bazel-out/darwin_arm64-opt/bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
    // --- END ---
    
    try {
        void* handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error("Failed to load PJRT plugin library at '" + path + "': " + std::string(dlerror()));
        }

        typedef const PJRT_Api* (*GetPjrtApiFunc)();
        GetPjrtApiFunc get_pjrt_api = (GetPjrtApiFunc)dlsym(handle, "GetPjrtApi");

        if (!get_pjrt_api) {
            throw std::runtime_error("Failed to find 'GetPjrtApi' symbol in PJRT plugin.");
        }

        const PJRT_Api* api = get_pjrt_api();

        if (api) {
            return "C++ SUCCESS: Loaded PJRT plugin from your Bazel build! API version: " +
                   std::to_string(api->pjrt_api_version.major_version) +
                   "." + std::to_string(api->pjrt_api_version.minor_version);
        } else {
            return "C++ ERROR: Plugin loaded but returned a null API pointer.";
        }
    } catch (const std::exception& e) {
        return "C++ ERROR: " + std::string(e.what());
    }
}

int main() {
    std::cout << "=== Testing PJRT plugin load from C++ ===" << std::endl;
    std::cout << xftcpp::test_pjrt_plugin_load() << std::endl;
    return 0;
}

} // namespace xftcpp