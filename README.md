# xft

simple deep-learning framework ❤️

Minimal deep learning framework built to understand how everything works under the hood.

A minimal array library built from scratch using **Pure Python + C++ + CUDA**. No magic. No hidden abstractions.

## Overview

This project is an educational exploration into how deep learning frameworks operate internally. It focuses on building the core components from scratch rather than relying on mature libraries.

At the moment, only the **array creation** layer has been implemented in C++. Python bindings are planned but not yet completed.

## Current Status

Very early stage.

* Array creation implemented in C++.
* CUDA groundwork being prepared.
* Python bindings not yet added.
* No tensor operations, autograd, or neural network modules implemented.

## Goals

* Build a minimal array and tensor system.
* Learn internal mechanics of frameworks like PyTorch.
* Implement autograd from scratch.
* Write a small CUDA backend to support GPU arrays.

## Planned Roadmap

* Python bindings for the C++ array layer.
* Basic tensor ops (add, mul, matmul).
* Autograd engine.
* CUDA kernels for core operations.
* Minimal neural network layers.

## Why This Exists

To deeply understand how deep learning frameworks work end to end: memory, kernels, tensors, autograd, and execution flows.

## License

MIT License.
