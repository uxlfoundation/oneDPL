# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

oneDPL (oneAPI Data Parallel Library) is a header-only C++17 library implementing the [oneAPI specification](https://github.com/uxlfoundation/oneAPI-spec/tree/main/source/elements/oneDPL) for parallel algorithms. It provides C++ standard library-like parallel algorithms that work across heterogeneous devices (CPUs, GPUs) using different parallel backends (TBB, DPCPP/SYCL, OpenMP, serial).

**Key Resources:**
- [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md) - architecture, roadmap and development patterns
- [cmake/README.md](cmake/README.md) - Complete build system documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Testing requirements and contribution guide
- [Official Documentation](https://uxlfoundation.github.io/oneDPL)

## Quick Start: Building and Testing

### Configure Build
```bash
cmake -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_CXX_STANDARD=17 \
      -DONEDPL_BACKEND=dpcpp \
      -DCMAKE_BUILD_TYPE=Release \
      -B build
```

**Common CMake Variables:**
- `ONEDPL_BACKEND`: tbb (default), dpcpp, dpcpp_only, omp, serial
- `CMAKE_BUILD_TYPE`: Debug, Release, RelWithDebInfo, RelWithAsserts

**Note:** `RelWithAsserts` is Release without `-DNDEBUG`, useful for testing assert-heavy code.
**Device selection:** For SYCL/DPC++ device selection, use `ONEAPI_DEVICE_SELECTOR` as documented in the [Device Selection](#device-selection) section below.

### Build and Run Tests

```bash
# Build specific test
cmake --build build --target sort.pass

# Build all tests
cmake --build build --target build-onedpl-tests

# Build by category
cmake --build build --target build-onedpl-algorithm-tests

# Run specific test
cd build && ctest -R ^sort.pass$

# Run tests by category
ctest -L algorithm
ctest -L numeric

# Run test executable directly
./build/test/sort.pass
```

**See `cmake/README.md` for complete testing options.**

### Build Documentation

```bash
# Setup documentation
cd documentation
python3 -m venv docdpl
source docdpl/bin/activate
pip install -r _auxiliary/requirements.txt

# Generate HTML in build/html
make html
```

## Architecture Quick Reference

oneDPL uses a three-tier architecture (see `IMPLEMENTATION_DETAILS.md` for details):

1. **Public API Layer** (`include/oneapi/dpl/`) - Standard-like algorithm headers
2. **Pattern Layer** (`include/oneapi/dpl/pstl/`, `include/oneapi/dpl/internal/`) - `__pattern_*()` implementations and extensions
3. **Backend Layer** (`include/oneapi/dpl/pstl/`) - Parallel execution primitives

**Key Pattern:** Algorithm → Pattern Function → Backend Primitive

Example: `std::any_of()` → `__pattern_any_of()` → `__parallel_or()`

## Test Organization

- `test/parallel_api/` - Parallel algorithm tests (algorithm, numeric, memory, ranges)
- `test/xpu_api/` - Device-side API tests
- `test/general/` - General functionality
- `test/kt/` - Kernel templates (experimental)

Tests use `.pass.cpp` suffix.

# Code styling
- Code should be self-documenting.
- Only use comments when necessary to explain something that is unclear.
- Do not create short 1-2 line functions only used once unless it improves understandability.

## Code Formatting

**clang-format is required** for all code except tests:
```bash
clang-format -i <file>
```
Contributors can override clang-format suggestions for readability in exceptional cases.

When touching existing code, migrate any `::std::` usages to `std::` within the functions touched.

## Device Selection

SYCL/DPC++ device selection via environment variables:

```bash
# Intel LLVM Compiler >= 2023.1
# Uncomment exactly one of the following selectors:
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# export ONEAPI_DEVICE_SELECTOR=opencl:gpu
# export ONEAPI_DEVICE_SELECTOR=*:cpu
```

## Code Review Guidelines

- **Formatting-only changes are frowned upon.** PRs should not include changes that are purely cosmetic (whitespace, brace style, etc.) without substantive functional changes. If a file is being modified for functional reasons, incidental formatting fixes in the same area are acceptable, but reformatting unrelated to the PR's purpose should be flagged in reviews and requested to be removed.

## Important Notes

- **Header-only library** - no binary artifacts
- **C++17 minimum** required; parallel ranges API requires **C++20**
- **Backend selection is compile-time only** - zero runtime overhead
- **Two-phase headers** - `*_defs.h` (declarations), `*_impl.h` (implementations)

