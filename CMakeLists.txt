# XFT - CMake Build Configuration

# ----------------------------------------------------------------------------
# Target Platform: x86_64 Linux (AMD Zen2+)
# Compiler: GCC/Clang with C++17

cmake_minimum_required(VERSION 3.15)

project(xft LANGUAGES CXX)

# ----------------------------------------------------------------------------
# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------------------------------------------------------
# Build Type
# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------------------------------------------------------
# Compiler Flags
# x86_64 optimizations - portable across AMD/Intel
# -march=native will optimize for your specific CPU (Zen2, Zen3, etc.)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall          # All warnings
        -Wextra        # Extra warnings
        -Wpedantic     # Pedantic warnings
        -Werror        # Treat warnings as errors (optional, comment out if too strict)
    )
    
    # Optimization flags for Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(
            -O3                # Maximum optimization
            -march=native      # Optimize for this CPU (uses AVX2 on Zen2)
            -DNDEBUG          # Disable assertions
        )
    endif()
    
    # Debug flags
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(
            -g             # Debug symbols
            -O0            # No optimization
            -fno-omit-frame-pointer  # Keep stack traces readable
        )
    endif()
endif()

# ----------------------------------------------------------------------------
# Include Directories
# All our headers are in cpp/
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp)

# ----------------------------------------------------------------------------
# Python Module: xft_core
# This builds the Python extension module for testing

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create the Python module
pybind11_add_module(xft_core MODULE cpp/bindings/bindings.cpp)

# Link against our headers (header-only for now)
target_include_directories(xft_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

# Output the module to the project root for easy Python import
set_target_properties(xft_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# ----------------------------------------------------------------------------
# C++ Test Executable: test_scalar_cpp
# This tests our C++ code directly without Python

add_executable(test_scalar_cpp tests/test_scalar.cpp)

# Link against our headers
target_include_directories(test_scalar_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

# Output to build directory
set_target_properties(test_scalar_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ----------------------------------------------------------------------------
# Custom Targets

# Target: make test
# Builds and runs C++ tests
add_custom_target(test
    COMMAND test_scalar_cpp
    DEPENDS test_scalar_cpp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running C++ tests..."
)

# ----------------------------------------------------------------------------
# Installation (Optional)
# For now, we just use the module from the source directory
# Uncomment if you want to install to site-packages:

# install(TARGETS xft_core
#     LIBRARY DESTINATION ${Python_SITEARCH}
# )

# ----------------------------------------------------------------------------
# Platform Verification
# Verify we're on the expected platform
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    message(WARNING "This project is optimized for x86_64, but detected: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "This project is designed for Linux, but detected: ${CMAKE_SYSTEM_NAME}")
endif()