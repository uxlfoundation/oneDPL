# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

oneDPL (oneAPI Data Parallel Library) is a header-only C++17 library (parallel ranges API requires C++20) implementing the [oneAPI specification](https://github.com/uxlfoundation/oneAPI-spec/tree/main/source/elements/oneDPL) for parallel algorithms across CPUs and GPUs using TBB, OpenMP, serial, or SYCL/DPC++ backends.

**Key Resources:**
- [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md) - architecture, code layout, and test organization
- [cmake/README.md](cmake/README.md) - complete build and test configuration options
- [CONTRIBUTING.md](CONTRIBUTING.md) - contribution and testing requirements
- `documentation/library_guide/` - public API semantics and oneDPL extensions

## Building and Testing

```bash
cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_STANDARD=17 -DONEDPL_BACKEND=dpcpp \
      -DCMAKE_BUILD_TYPE=Release -B build

cmake --build build --target sort.pass                  # single test
cmake --build build --target run-sort.pass              # build and run single test
cmake --build build --target build-onedpl-tests         # all tests
cmake --build build --target build-onedpl-algorithm-tests  # by label

cd build && ctest -R ^sort.pass$
ctest -L ^algorithm$
```

- `ONEDPL_BACKEND`: tbb, dpcpp, dpcpp_only, omp, serial. Defaults to dpcpp if the compiler supports SYCL, otherwise tbb.
- `CMAKE_BUILD_TYPE`: Debug, Release, RelWithDebInfo, RelWithAsserts (Release without `-DNDEBUG`).
- Test labels come from the test's directory path components (e.g. `algorithm`, `ranges`, `xpu_api`).
- For host-only changes, `g++` with the `serial` or `tbb` backend compiles much faster than `icpx` with `dpcpp`.
- Tests should compile with both C++17 and C++20; ranges tests are only active under C++20.

### Device Selection

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu   # or opencl:gpu, opencl:cpu
```

`SYCL_HOST_UNIFIED_MEMORY=0` on a CPU device reproduces buffer bugs that otherwise only appear on discrete GPUs.

### Documentation

```bash
cd documentation && python3 -m venv docdpl && source docdpl/bin/activate
pip install -r _auxiliary/requirements.txt && make html   # output in build/html
```

## Writing Tests

Tests live in `test/` and use the `.pass.cpp` suffix (except `test/kt/`). Reuse the helpers in `test/support/`:
- `utils.h` - `EXPECT_TRUE`, `EXPECT_EQ`, `EXPECT_EQ_N`, `EXPECT_EQ_RANGES`
- `utils_invoke.h` - `invoke_on_all_policies` runs a test body across host and device policies

## Code Style

- **clang-format is required** for all code except tests: `clang-format -i <file>`. Overrides are acceptable in exceptional cases for readability.
- Code should be self-documenting; only comment what is unclear, keep comments brief.
- Do not create short 1-2 line functions used only once unless it improves understandability.
- Internal identifiers use reserved names: `__name` for functions and variables, `_Name` for template parameters. Internal code lives in namespaces such as `oneapi::dpl::__internal`.
- When touching existing code, migrate `::std::` usages to `std::` within the functions touched.

## Code Review Guidelines

- **Formatting-only changes are frowned upon.** PRs should not include changes that are purely cosmetic (whitespace, brace style, etc.) without substantive functional changes. If a file is being modified for functional reasons, incidental formatting fixes in the same area are acceptable, but reformatting unrelated to the PR's purpose should be flagged in reviews and requested to be removed.
