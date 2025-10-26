# Copyright 2025 XFT Authors.
#
# Simplified nanobind-based pywrap rule inspired by JAX.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

def pywrap_extension(
        name,
        srcs,
        deps = [],
        copts = [],
        linkopts = [],
        visibility = None):
    """Defines a nanobind-based Python extension module."""

    native.cc_binary(
        name = name + ".so",
        srcs = srcs,
        deps = deps + [
            "@nanobind//:nanobind",
            "//xftcpp/tools/python_runtime:headers",
        ],
        linkshared = True,
        copts = copts + [
            "-std=c++17",
        ],
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:darwin": [
                "-undefined",
                "dynamic_lookup",
            ],
            "//conditions:default": [],
        }),
        visibility = visibility,
    )
