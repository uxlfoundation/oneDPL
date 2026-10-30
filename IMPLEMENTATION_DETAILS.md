# oneDPL Implementation Details

This document describes the high-level architecture and development patterns in oneDPL.

## Three-Tier Design

Algorithm → Pattern Function → Backend Primitive

Example: `oneapi::dpl::any_of()` → `__internal::__pattern_any_of()` → `__internal::__parallel_or()`

### 1. Public API (Glue) Layer
**Location:** `include/oneapi/dpl/` (public headers) and `include/oneapi/dpl/pstl/glue_*.h`

- Thin wrappers accepting execution policies
- Call `__select_backend()` to obtain a dispatch tag, then call the pattern function with it
- Headers are split into `*_defs.h` (declarations) and `*_impl.h` (implementations); `*_fwd.h` holds pattern forward declarations

### 2. Pattern Layer
Core algorithm logic in `__pattern_*()` functions, overloaded on the dispatch tag and iterator category:

| Location | Contents |
|----------|----------|
| `pstl/algorithm_impl.h`, `numeric_impl.h`, `memory_impl.h` | Host patterns (serial, TBB, OpenMP) |
| `pstl/hetero/*_impl_hetero.h` | Device (SYCL) patterns |
| `pstl/algorithm_ranges_impl.h`, `numeric_ranges_impl.h`, `memory_ranges_impl.h` | C++20 `std::ranges` patterns (host) |
| `pstl/hetero/*_ranges_impl_hetero.h` | `oneapi::dpl::__ranges` patterns (device), used by both C++20 ranges and oneDPL ranges |
| `internal/` | Extensions: by-segment algorithms, binary search, async, dynamic selection, random |

`__brick_*()` functions are elementary serial/SIMD implementations used inside work units by patterns, e.g. `__brick_any_of()` serves both serial and parallel `__pattern_any_of()`.

### 3. Backend Layer
**Host backends** (`include/oneapi/dpl/pstl/`):
- `parallel_backend_tbb.h`, `parallel_backend_omp.h` (+ `omp/`), `parallel_backend_serial.h`
- Each implements `__parallel_for()`, `__parallel_reduce()`, `__parallel_strict_scan()`, etc.
- `parallel_backend.h` aliases `__par_backend` to the active host backend

**SYCL backend** (`include/oneapi/dpl/pstl/hetero/dpcpp/`):
- `parallel_backend_sycl.h` is the entry point and contains scan, copy_if, and related primitives
- Per-algorithm kernels are in `parallel_backend_sycl_<algo>.h`
- `execution_sycl_defs.h` defines device policies; `sycl_iterator.h` and `utils_ranges_sycl.h` adapt buffers and iterators for kernels

## Execution Policies and Dispatch Tags

| Policy | Dispatch tag |
|--------|--------------|
| `seq`, `unseq` | `__serial_tag<_IsVector>` |
| `par`, `par_unseq` | `__parallel_tag<_IsVector>` (random-access iterators) or `__parallel_forward_tag` (forward iterators) |
| `device_policy` | `__hetero_tag<__device_backend_tag>` |
| `fpga_policy` | `__hetero_tag<__fpga_backend_tag>` |

## Backend Selection

Backend selection is compile-time only. The host backend is chosen by the `ONEDPL_USE_TBB_BACKEND` / `ONEDPL_USE_OPENMP_BACKEND` macros (serial when both are `0`); the SYCL backend is enabled when the compiler supports SYCL unless `ONEDPL_USE_DPCPP_BACKEND=0`. oneDPL's CMake `ONEDPL_BACKEND` option sets these macros; users who include the headers directly set them (or rely on defaults) themselves. See `internal/version_impl.h` and `pstl/onedpl_config.h`.

## Testing Architecture

Tests are organized in `test/` by area. Under the dpcpp backend, most tests run with both host and device policies.

| Directory | Contents |
|-----------|----------|
| `parallel_api/` | Parallel algorithms: `algorithm/`, `numeric/`, `memory/`, `ranges/`, `iterator/`, `dynamic_selection/`, `device_array/`, `experimental/` |
| `xpu_api/` | C++ standard APIs callable from SYCL kernels |
| `general/` | Policies, SYCL iterators, and other general functionality |
| `kt/` | Kernel templates (experimental, require specific device capabilities; `*.cpp` suffix) |
| `support/` | Shared test utilities |
