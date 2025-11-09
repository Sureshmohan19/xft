#!/usr/bin/env python3
"""
XFT - Setup Script
==================
Builds C++ extension using CMake and integrates with Python packaging.
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Custom Extension class that uses CMake to build."""
    
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command that invokes CMake."""
    
    def run(self):
        """Run CMake build for all extensions."""
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions)
            )
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        """Build a single extension using CMake."""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]
        
        # Build type
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        
        # Platform-specific arguments
        if sys.platform.startswith('linux'):
            cmake_args += [
                '-DCMAKE_CXX_FLAGS=-fPIC',
            ]
        
        # Parallel build
        if hasattr(self, 'parallel') and self.parallel:
            build_args += [f'-j{self.parallel}']
        else:
            build_args += ['-j4']
        
        # Build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Run CMake configuration
        print(f"Running CMake in {self.build_temp}")
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        
        # Run CMake build
        print(f"Building extension {ext.name}")
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', 'xft_core'] + build_args,
            cwd=self.build_temp
        )
        
        print(f"Extension {ext.name} built successfully")


# Read version from pyproject.toml (or define here)
VERSION = "0.1.0"

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

setup(
    name="xft",
    version=VERSION,
    author="Aakriti Suresh",
    description="XFT - Deep Learning Framework optimized for x86_64",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension('xft.xft_core')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.8",
)

