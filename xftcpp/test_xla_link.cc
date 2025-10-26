#include <iostream>
#include "absl/status/status.h"
#include "xla/util.h"

int main() {
    std::cout << "XLA linking test successful!" << std::endl;
    auto status = absl::OkStatus();
    std::cout << "Status: " << status.message() << std::endl;
    return 0;
}
