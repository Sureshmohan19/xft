# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(xftcpp)

# Export compile commands for IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python first (required by nanobind)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Automatically find nanobind via Python
execute_process(
    COMMAND python -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find nanobind, which was installed by pip
find_package(nanobind REQUIRED)

# This command creates our Python module
# It will be named xftcpp.so (or .pyd on Windows)
nanobind_add_module(xftcpp src/main.cpp)