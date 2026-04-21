# oneDPL Implementation Details

This document describes the high-level architecture and development patterns in oneDPL.

oneDPL is a header-only library, and has a C++17 minimum requirement.

## Three-Tier Design Pattern

oneDPL uses a layered architecture to provide a single API across multiple backends:

### 1. Glue/Public API Layer
**Location:** `include/oneapi/dpl/`

Public headers matching C++ standard library naming:
- `algorithm`, `execution`, `numeric`, `memory`, `ranges`, `iterator`

Implementation in `glue_*.h` files:
- Thin wrappers accepting execution policies
- Dispatch to pattern layer using `__select_backend()`
- Files split: `*_defs.h` (declarations), `*_impl.h` (implementations)

### 2. Pattern Implementation Layer
**Location:** `include/oneapi/dpl/pstl/` (e.g., `algorithm_impl.h`, `numeric_impl.h`)

Core algorithm logic in `__pattern_*()` functions:
- Polymorphic overloads based on:
  - Iterator type (forward, random-access)
  - Execution policy (serial, parallel, hetero)
  - Vectorization capability (`_IsVector` template parameter)
- Enable single algorithm to serve multiple execution contexts

Note: `include/oneapi/dpl/internal/` contains some extension implementations (scan-by-segment, binary search, dynamic selection, etc.).

Example flow: `std::any_of()` → `__internal::__pattern_any_of()` → backend primitives

### 3. Backend Implementation Layer
**Location:** `include/oneapi/dpl/pstl/`

Backend-specific parallelism primitives:

**Host Backends:**
- `parallel_backend_tbb.h` - Intel Threading Building Blocks
- `parallel_backend_omp.h` - OpenMP pragmas
- `parallel_backend_serial.h` - Sequential fallback

**Heterogeneous Backend:**
- `hetero/dpcpp/parallel_backend_sycl*.h` - SYCL/DPC++ for GPUs/accelerators

Each backend implements common interface:
- `__parallel_for()`, `__parallel_reduce()`, `__parallel_scan()`, etc.
- Compile-time selection via namespace aliasing (`__par_backend`)

## Execution Policies

Policies control algorithm execution strategy:

| Policy | Execution | Use Case |
|--------|-----------|----------|
| `sequenced_policy` (seq) | Sequential | Single-threaded CPU |
| `parallel_policy` (par) | Parallel threads | Multi-core CPU with TBB/OpenMP |
| `parallel_unsequenced_policy` (par_unseq) | Parallel + SIMD | Multi-core CPU with vectorization |
| `unsequenced_policy` (unseq) | SIMD only | Single-threaded with vectorization |
| Device policies | GPU/accelerator | SYCL/DPC++ execution |

Policies map to execution tags:
- `__serial_tag` - Sequential execution
- `__parallel_tag<_IsVector>` - Parallel execution with optional vectorization
- `__hetero_tag<_BackendTag>` - Device execution

## Backend Architecture

### Backend Selection

Backends selected at **compile-time only** via CMake `ONEDPL_BACKEND`:

```cmake
-DONEDPL_BACKEND=tbb        # TBB (default for most platforms)
-DONEDPL_BACKEND=dpcpp      # SYCL/DPC++ with TBB for host
-DONEDPL_BACKEND=dpcpp_only # SYCL/DPC++ without TBB
-DONEDPL_BACKEND=omp        # OpenMP
-DONEDPL_BACKEND=serial     # Sequential
```

Selection logic in `CMakeLists.txt`:
- When `ONEDPL_BACKEND` is not set, auto-detects SYCL support: defaults to `dpcpp` if SYCL is available, otherwise `tbb`
- When `ONEDPL_BACKEND` is explicitly set, auto-detection is skipped; the specified backend's dependencies must be available
- Namespace `__par_backend` points to active backend

### SYCL/DPC++ Backend

**Specialized Kernel Implementations:**

| File | Purpose |
|------|---------|
| `parallel_backend_sycl_for.h` | Kernel launching primitives |
| `parallel_backend_sycl_reduce.h` | Reduction operations |
| `parallel_backend_sycl_*scan*.h` | Scan/prefix-sum operations |
| `parallel_backend_sycl_merge*.h` | Merge algorithms |
| `parallel_backend_sycl_radix_sort*.h` | Optimized radix sort |
| `parallel_backend_sycl_reduce_by_segment.h` | Segmented operations |

## Development Patterns

### Pattern-Based Implementation

Standard pattern for implementing algorithms:

1. **Public API** (`glue_algorithm_impl.h`)
   - Accepts execution policy + algorithm parameters
   - Dispatches to pattern function

2. **Pattern Function** (`algorithm_impl.h`)
   - `__pattern_*()` with overloads for different contexts
   - Selects appropriate brick/backend based on:
     - Iterator category
     - Execution policy
     - Vectorization support

3. **Backend Primitive**
   - Actual parallel execution via backend
   - Example: `__par_backend::__parallel_or()`

### Brick Functions

`__brick_*()` functions provide elementary implementations:
- Used within work units by patterns
- Enable code reuse between serial/parallel variants
- Example: `__brick_any_of()` used by both serial and parallel `__pattern_any_of()`


## Testing Architecture

Tests organized by execution model:

**`test/parallel_api/`** - Tests using host execution policies
- `algorithm/` - Parallel algorithm tests
- `numeric/` - Reduction, scan operations
- `memory/` - Memory algorithms
- `ranges/` - Range-based API

**`test/xpu_api/`** - Tests for device-side API
- C++ standard APIs callable from SYCL kernels
- Validates device-side iterator support

**`test/general/`** - General functionality
- SYCL iterator tests
- Policy behavior validation

**`test/kt/`** - Kernel templates (experimental)
- Hardware-specific optimizations
- Requires specific device capabilities

Test naming: `*.pass.cpp` suffix for tests expected to pass.
