# oneDPL Development Guide

**oneAPI DPC++ Library (oneDPL)** - A high-productivity C++ library providing parallel algorithms and utilities for heterogeneous computing, particularly targeting SYCL/DPC++ devices.

## Repository Overview

oneDPL is a **header-only library** implementing the oneAPI specification. It provides:
- Parallel STL algorithms (like `std::for_each`, `std::transform`, etc.)
- Numeric algorithms (`std::transform_reduce`, `std::inclusive_scan`, etc.)
- Custom iterators (zip, permutation, counting, etc.)
- SYCL-specific features (buffer iterators, USM support, kernel templates)
- Multiple backend support: SYCL/DPC++, TBB, OpenMP, serial

**License**: Apache 2.0 with LLVM exceptions
**Homepage**: https://uxlfoundation.github.io/oneDPL

---

## Directory Structure

### Core Implementation (`include/`)
```
include/oneapi/dpl/
├── algorithm              # Public API: parallel algorithms
├── numeric                # Public API: numeric algorithms
├── execution              # Execution policies (seq, par, unseq, device policies)
├── iterator               # Custom iterators (zip, permutation, transform, etc.)
├── async                  # Asynchronous operations
├── random                 # Random number generation
├── pstl/                  # Parallel STL implementation layer
│   ├── algorithm_impl.h           # Host-side algorithm patterns
│   ├── numeric_impl.h             # Host-side numeric patterns
│   ├── glue_algorithm_impl.h      # Dispatch layer connecting API → patterns
│   ├── parallel_backend.h         # Backend abstraction
│   ├── parallel_backend_tbb.h     # TBB backend
│   ├── parallel_backend_omp.h     # OpenMP backend
│   └── hetero/                    # Device (heterogeneous) implementations
│       ├── algorithm_impl_hetero.h     # Device algorithm patterns (2133 lines)
│       ├── numeric_impl_hetero.h       # Device numeric patterns
│       └── dpcpp/                      # SYCL/DPC++ backend specifics
│           ├── parallel_backend_sycl.h          # Core SYCL backend (128KB!)
│           ├── utils_ranges_sycl.h              # SYCL buffer management
│           ├── parallel_backend_sycl_utils.h    # SYCL utilities
│           ├── execution_sycl_defs.h            # SYCL execution policies
│           ├── sycl_iterator.h                  # SYCL iterator adaptors
│           └── parallel_backend_sycl_*.h        # Specialized algorithms
│               ├── _reduce.h           # Reductions
│               ├── _merge_sort.h       # Merge sort
│               ├── _radix_sort*.h      # Radix sort (including one workgroup)
│               ├── _histogram.h        # Histogram
│               ├── _scan_by_segment.h  # Segmented scan
│               └── ...
└── internal/              # Internal utilities, dynamic selection
```

### Testing (`test/`)
```
test/
├── parallel_api/          # Main test suite (834 .pass.cpp files)
│   ├── algorithm/         # Algorithm tests
│   ├── numeric/           # Numeric algorithm tests
│   ├── iterator/          # Iterator tests
│   ├── ranges/            # Range algorithm tests
│   ├── memory/            # Memory operation tests
│   ├── dynamic_selection/ # Dynamic selection API tests
│   └── experimental/      # Experimental feature tests
├── xpu_api/               # Cross-platform API tests
├── general/               # General tests (headers, interface checks)
├── kt/                    # Kernel template tests
└── support/               # Test utilities and helpers
```

### Build System (`cmake/`)
- Root `CMakeLists.txt` (422 lines) - Main build configuration
- `cmake/README.md` - Comprehensive CMake usage guide
- `cmake/FindTBB.cmake` - TBB discovery
- `cmake/oneDPLWindowsIntelLLVMConfig.cmake` - Windows workarounds

### Documentation (`documentation/`)
- `library_guide/` - Complete API documentation
- `release_notes.rst` - Version history
- Examples in `examples/` directory

---

## Code Architecture & Dispatch Flow

### Algorithm Implementation Pattern

oneDPL uses a **layered dispatch architecture** to route algorithm calls to appropriate backend implementations:

```
User Code
    ↓
1. Public API (include/oneapi/dpl/algorithm)
    └─ oneapi::dpl::for_each(policy, first, last, func)
    ↓
2. Glue Layer (pstl/glue_algorithm_impl.h)
    └─ Dispatches based on execution policy type
    ↓
3. Generic Pattern Layer (pstl/algorithm_impl.h OR hetero/algorithm_impl_hetero.h)
    └─ __pattern_walk1(__hetero_tag, exec, first, last, func)
    └─ Creates SYCL buffers, sets up access modes
    ↓
4. Backend Layer (pstl/hetero/dpcpp/parallel_backend_sycl.h)
    └─ __parallel_for(tag, exec, kernel_func, n, buffers...)
    └─ Submits SYCL kernel to queue
    ↓
5. Kernel Execution (unseq_backend_sycl.h)
    └─ walk_n_vectors_or_scalars::operator()
    └─ Executes user function on device
```

### Algorithm Pattern Classes

oneDPL organizes algorithms into **10 main pattern classes**. Each has specialized SYCL backend implementations.

| Pattern | Backend File | Backend Primitive | Use Case |
|---------|-------------|------------------|----------|
| **for** | `parallel_backend_sycl_for.h` | `__parallel_for` | Element-wise operations: `for_each`, `transform`, `fill`, `generate` |
| **find_or** | `parallel_backend_sycl.h` | `__parallel_find_or` | Search with early-exit: `find_if`, `any_of`, `search`, `adjacent_find` |
| **reduce** | `parallel_backend_sycl_reduce.h` | `__parallel_transform_reduce` | Aggregations: `reduce`, `count`, `min_element`, `inner_product` |
| **scan** | `parallel_backend_sycl_reduce_then_scan.h` | `__parallel_transform_scan` | Prefix sums: `inclusive_scan`, `exclusive_scan` |
| **merge** | `parallel_backend_sycl_merge.h` | `__parallel_merge` | Merge sorted sequences using balanced path partitioning |
| **merge_sort** | `parallel_backend_sycl_merge_sort.h` | `__parallel_sort_impl` | Stable sorting via bottom-up merging |
| **radix_sort** | `parallel_backend_sycl_radix_sort.h` | `__parallel_radix_sort` | Fast integer/float sorting (2-5x faster than merge) |
| **reduce_by_segment** | `parallel_backend_sycl_reduce_by_segment.h` | `__parallel_reduce_by_segment` | Reduce within segments defined by keys |
| **scan_by_segment** | `parallel_backend_sycl_scan_by_segment.h` | `__parallel_scan_by_segment` | Prefix sum within segments (resets at boundaries) |
| **histogram** | `parallel_backend_sycl_histogram.h` | `__parallel_histogram` | Frequency counting with atomic operations |

**Walk Pattern Building Blocks** (used by `for` pattern):
- `__pattern_walk1(first, last, func)` - Single-range: `for_each`, `fill`
- `__pattern_walk2(first1, last1, first2, func)` - Two-range: `transform`, `copy`
- `__pattern_walk3(first1, last1, first2, first3, func)` - Three-range: binary `transform`

**Key Implementation Details**:
- **Radix sort** requires `_ONEDPL_USE_RADIX_SORT` (sub-groups + group algorithms)
- **Sort dispatch**: Radix sort for integral types with standard comparators, merge sort otherwise
- **Reduce/scan**: Multi-stage algorithms (local → reduce → scan → propagate) for parallelism
- **Segmented ops**: Use reduce-then-scan to handle segment boundaries efficiently

---

## SYCL Buffer Management

### `__get_sycl_range` - The Core Utility

**Location**: `include/oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h`

**Purpose**: Creates SYCL buffer views from iterators, managing data movement between host and device.

**Template Parameters**:
```cpp
template <access_mode _AccMode = access_mode::read_write,
          typename _Iterator,
          bool _NoInit = false>
auto __get_sycl_range();
```

**Access Modes**:
| Mode | Copy-in Behavior | Copy-back Behavior | Use Case |
|------|------------------|-------------------|----------|
| `read` | Always copies | Never copies back | Input data only |
| `write` | Copies unless `_NoInit=true` | Always copies back | Output data |
| `read_write` | Copies unless `_NoInit=true` | Always copies back | In-place modification |

**The `_NoInit` Parameter**:
- `_NoInit=false` (default): Buffer is initialized with host data (copy-in happens)
- `_NoInit=true`: Buffer is NOT initialized (no copy-in) - **performance optimization**

### Access Mode Selection Guide

**Critical Rule**: Match access mode to actual data usage to avoid unnecessary data transfers!

| Scenario | Access Mode | `_NoInit` | Rationale |
|----------|-------------|-----------|-----------|
| **Pure input** (read only) | `read` | `false` | Need host data, no write-back |
| **Pure output** (overwrite all elements) | `write` | `true` | Don't need host data, will overwrite everything |
| **Partial output** (preserve some elements) | `write` | `false` | Need to preserve untransformed elements (e.g., `copy_if`, `transform_if`) |
| **In-place modification** | `read_write` | `false` | Need to read and write same buffer |
| **Temporary/scratch buffer** | `read_write` | `true` | Intermediate storage, don't care about initial values |

## Build System & CMake Configuration

### Key CMake Variables

**Backend Selection**:
| Variable | Values | Description |
|----------|--------|-------------|
| `ONEDPL_BACKEND` | `tbb`, `dpcpp`, `dpcpp_only`, `serial`, `omp` | Threading backend (default: `dpcpp` for DPC++ compiler, `tbb` for others) |

**Device Configuration (DPC++ only)**:
| Variable | Values | Description |
|----------|--------|-------------|
| `ONEDPL_DEVICE_TYPE` | `GPU`, `CPU`, `FPGA_HW`, `FPGA_EMU` | Target device type (default: `GPU`) |
| `ONEDPL_DEVICE_BACKEND` | `opencl`, `level_zero`, `cuda`, `hip`, `*` | Device backend (default: `*` = best per runtime) |

**Compiler Options**:
| Variable | Values | Description |
|----------|--------|-------------|
| `CMAKE_CXX_COMPILER` | `dpcpp`, `icpx`, `g++`, etc. | C++ compiler |
| `CMAKE_CXX_STANDARD` | `17`, `20`, `23` | C++ standard version |
| `CMAKE_BUILD_TYPE` | `Release`, `Debug`, `RelWithDebInfo` | Build type |
| `ONEDPL_USE_UNNAMED_LAMBDA` | `ON`, `OFF` | Pass `-fsycl-unnamed-lambda` flag |
| `ONEDPL_ENABLE_SIMD` | `ON` (default), `OFF` | Enable OpenMP SIMD vectorization |

**Test Configuration**:
| Variable | Values | Description |
|----------|--------|-------------|
| `ONEDPL_TEST_EXPLICIT_KERNEL_NAMES` | `AUTO` (default), `ALWAYS` | Control kernel naming in tests |

### Common Build Workflows

**1. Configure for SYCL GPU Development**:
```bash
# From repository root
cmake -B build \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_STANDARD=20 \
  -DONEDPL_BACKEND=dpcpp \
  -DONEDPL_DEVICE_TYPE=GPU \
  -DONEDPL_DEVICE_BACKEND=level_zero \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

**2. Build Specific Test**:
```bash
# Build test executable
cmake --build build --target for_each.pass

# Build and run test
cmake --build build --target run-for_each.pass
```

**3. Build All Tests in Subdirectory**:
```bash
# Build all algorithm tests
cmake --build build --target build-onedpl-algorithm-tests

# Build and run all algorithm tests
cmake --build build --target run-onedpl-algorithm-tests
```

**4. Run Tests via CTest**:
```bash
cd build

# Run all tests
ctest --output-on-failure

# Run specific test
ctest -R for_each.pass --output-on-failure

# Run tests with specific label (subdirectory)
ctest -L algorithm --output-on-failure
```

**5. Device Selection at Runtime**:
```bash
# For SYCL devices
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# or
export SYCL_DEVICE_FILTER=level_zero:gpu

# Run test
./for_each.pass
```

### Test Organization

**Test Target Naming**:
- `<test-name>` - Build specific test (e.g., `for_each.pass`)
- `run-<test-name>` - Build and run specific test
- `build-onedpl-<subdir>-tests` - Build all tests in subdirectory
- `run-onedpl-<subdir>-tests` - Build and run all tests in subdirectory
- `build-onedpl-tests` - Build ALL tests
- `run-onedpl-tests` - Build and run ALL tests

**CTest Labels**: Tests are automatically labeled by their directory path. For example:
- `test/parallel_api/algorithm/for_each.pass.cpp` → labels: `parallel_api`, `algorithm`

**Note**: Tests may fail at runtime if no SYCL device is available (expected in non-GPU environments).

---

## Testing Infrastructure

### Test File Structure

Tests follow a consistent pattern:
```cpp
// Standard header
#include "support/test_config.h"

// Include oneDPL headers via macro (supports both <oneapi/dpl/...> and direct paths)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

// Test implementation
int main() {
    // Test logic using TestUtils::EXPECT_TRUE, etc.
    return TestUtils::done();
}
```

**Key Test Support Files**:
- `test/support/test_config.h` - Configuration, compiler workarounds, feature detection
- `test/support/utils.h` - Test utilities (EXPECT_TRUE, done(), invoke_on_all_policies, etc.)

### Debugging Test Failures

**Common Issues**:
1. **No device available**: Tests fail with "No device found" or similar
   - Check `ONEAPI_DEVICE_SELECTOR` or `SYCL_DEVICE_FILTER` environment variables
   - Verify GPU drivers and runtime are properly installed

2. **Timeout**: Long-running tests may timeout
   - Adjust timeout in `test/CMakeLists.txt` via `set(<test-name>_timeout_<debug|release> <seconds>)`

3. **Windows checked iterators**: Debug builds may have issues
   - Tests automatically disable checked iterators on Windows debug builds
   - See `test/CMakeLists.txt:75-85` for the workaround logic

4. **Compiler-specific issues**: See `test/support/test_config.h:26-81` for known compiler bugs and workarounds

---

## Common Development Patterns

### Adding a New Algorithm Implementation

**Step-by-step guide**:

1. **Add public API** (if not already present):
   - `include/oneapi/dpl/pstl/glue_algorithm_defs.h` - Forward declaration
   - `include/oneapi/dpl/pstl/glue_algorithm_impl.h` - Implementation that dispatches to patterns

2. **Implement host-side pattern** (for TBB/OpenMP/serial backends):
   - `include/oneapi/dpl/pstl/algorithm_impl.h`
   - Use existing patterns or implement new `__pattern_*` function

3. **Implement device-side pattern** (for SYCL backend):
   - `include/oneapi/dpl/pstl/hetero/algorithm_impl_hetero.h`
   - Create `__pattern_*` function with `__hetero_tag` tag
   - Use `__get_sycl_range` to create buffer views with appropriate access modes
   - Call `__par_backend_hetero::__parallel_for` (or other backend primitive)
   - Choose correct wait mode for synchronization

4. **Add tests**:
   - `test/parallel_api/algorithm/` - Add `.pass.cpp` test file
   - Test with different execution policies, data types, and edge cases

### Example: Implementing `transform`

**Simplified example** (actual code is more complex):
```cpp
// 1. Public API (glue_algorithm_impl.h)
template <class _ExecutionPolicy, class _ForwardIterator1,
          class _ForwardIterator2, class _UnaryOperation>
_ForwardIterator2
transform(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
          _ForwardIterator2 __result, _UnaryOperation __op)
{
    auto __dispatch_tag = __internal::__select_backend(__exec, __first);
    return __internal::__pattern_transform(__dispatch_tag,
                                          std::forward<_ExecutionPolicy>(__exec),
                                          __first, __last, __result, __op);
}

// 2. Device pattern (hetero/algorithm_impl_hetero.h)
template <typename _BackendTag, typename _ExecutionPolicy,
          typename _ForwardIterator1, typename _ForwardIterator2, typename _Function>
_ForwardIterator2
__pattern_transform(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec,
                    _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                    _ForwardIterator2 __result, _Function __op)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return __result;

    // Input: read-only
    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<access_mode::read>();
    auto __buf1 = __keep1(__first1, __last1);

    // Output: write-only, no init (pure output)
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<access_mode::write,
                                                           /*_NoInit=*/true>();
    auto __buf2 = __keep2(__result, __result + __n);

    // Submit kernel
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::walk_n_vectors_or_scalars<_Function>{__op, __n},
        __n, __buf1.all_view(), __buf2.all_view())
        .__checked_deferrable_wait();

    return __result + __n;
}
```

### Performance Optimization Tips

1. **Use `_NoInit=true` for pure output buffers**:
   - Avoid unnecessary host-to-device copy
   - Significant speedup for large buffers

2. **Choose minimal access mode**:
   - `read` instead of `read_write` when possible
   - Allows better optimization by SYCL runtime

3. **Prefer deferrable wait** (default):
   - Use `__checked_deferrable_wait()` instead of immediate wait
   - Allows pipelining of operations

4. **Consider specialized backends**:
   - For complex algorithms, consider specialized implementations in `parallel_backend_sycl_*.h`
   - Examples: radix sort, merge sort, reductions, scans

5. **Profile with actual workloads**:
   - Use `sycl::event` profiling to measure kernel time
   - Check memory transfer overhead vs. compute time

---

## Backend Dispatch Logic

### How Execution Policies Map to Backends

**Execution Policy Types**:
| Policy | Location | Backend | Description |
|--------|----------|---------|-------------|
| `oneapi::dpl::execution::seq` | `execution_defs.h` | Serial | Sequential execution |
| `oneapi::dpl::execution::unseq` | `execution_defs.h` | Serial + SIMD | Vectorized sequential |
| `oneapi::dpl::execution::par` | `execution_defs.h` | TBB/OpenMP | Parallel on host |
| `oneapi::dpl::execution::par_unseq` | `execution_defs.h` | TBB/OpenMP + SIMD | Parallel + vectorized on host |
| `oneapi::dpl::execution::dpcpp_default` | `execution_sycl_defs.h` | SYCL | Device execution |
| `oneapi::dpl::execution::make_device_policy(queue)` | `execution_sycl_defs.h` | SYCL | Device with specific queue |

### Dispatch Mechanism

**Tag Dispatch Pattern**:
```cpp
// Tags identify backend type
struct __serial_backend_tag {};
struct __tbb_backend_tag {};
template <typename _BackendTag> struct __hetero_tag {};  // For device execution

// Pattern functions are overloaded on tag type
template <typename _BackendTag, typename _ExecutionPolicy, ...>
void __pattern_walk1(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, ...) {
    // SYCL implementation
}

template <typename _ExecutionPolicy, ...>
void __pattern_walk1(__tbb_backend_tag, _ExecutionPolicy&& __exec, ...) {
    // TBB implementation
}
```

**Selection Logic** (simplified):
```cpp
auto __select_backend(_ExecutionPolicy __exec, Iterator __first) {
    if constexpr (is_device_execution_policy<_ExecutionPolicy>) {
        return __hetero_tag</*SYCL backend*/>{};
    } else if constexpr (is_parallel_execution_policy<_ExecutionPolicy>) {
        return __tbb_backend_tag{};  // or __omp_backend_tag
    } else {
        return __serial_backend_tag{};
    }
}
```

---

## Template Metaprogramming Patterns

oneDPL uses extensive template metaprogramming for:
- Backend dispatch
- SFINAE-based API enabling/disabling
- Iterator trait detection
- Type trait utilities

**Common Idioms**:

### 1. SFINAE for Execution Policy Detection
```cpp
// Enable function only for execution policies
template <class _ExecutionPolicy, class... _Args>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ReturnType>
algorithm_name(_ExecutionPolicy&& __exec, _Args&&... __args);
```

### 2. Conditional Backend Selection
```cpp
// Use different implementations based on backend capabilities
if constexpr (__supports_fast_scan<_BackendTag>) {
    return __pattern_scan_fast(...);
} else {
    return __pattern_scan_generic(...);
}
```

### 3. Iterator Category Detection
```cpp
// Different algorithm for random access vs forward iterators
if constexpr (std::is_same_v<
    typename std::iterator_traits<_Iterator>::iterator_category,
    std::random_access_iterator_tag>) {
    // Optimized path
} else {
    // Generic path
}
```

---

## Design Principles & Best Practices

### Architecture Philosophy

1. **Layered abstraction**: Clean separation between API, patterns, and backend
2. **Backend flexibility**: Same algorithm works on CPU (TBB/OpenMP) or GPU (SYCL)
3. **Zero-overhead abstractions**: Template metaprogramming for compile-time dispatch
4. **Standard compliance**: Implements C++ parallel algorithms specification

### Code Conventions

1. **Naming**:
   - Public API: STL names (`std::for_each`, `oneapi::dpl::for_each`)
   - Internal patterns: `__pattern_*` (double underscore prefix)
   - Internal utilities: `__internal::*` namespace
   - Backend functions: `__parallel_*`, `__par_backend_*`

2. **Template parameters**:
   - Execution policy: `_ExecutionPolicy`
   - Iterators: `_ForwardIterator`, `_RandomAccessIterator`, etc.
   - Functions/predicates: `_Function`, `_Predicate`, `_UnaryOp`, `_BinaryOp`
   - Backend tags: `_BackendTag`

3. **File organization**:
   - Forward declarations in `*_fwd.h`
   - Implementations in `*_impl.h`
   - Glue layer in `glue_*.h`
   - Backend-specific in `parallel_backend_*.h`

### Common Gotchas & Pitfalls

1. **Buffer lifetime management**:
   - `__keep` variables MUST stay in scope until kernel completes
   - SYCL buffer destruction triggers host-device synchronization
   - Example:
   ```cpp
   // WRONG - buffer destroyed too early!
   auto __buf = [&]() {
       auto __keep = __get_sycl_range<...>();
       return __keep(__first, __last);
   }();  // __keep destroyed here, buffer may be invalid!

   // CORRECT
   auto __keep = __get_sycl_range<...>();
   auto __buf = __keep(__first, __last);
   // ... use __buf ...
   // __keep stays alive
   ```

2. **Access mode mismatches**:
   - Using `write` with `_NoInit=false` for pure output wastes bandwidth
   - Using `read` when you need `read_write` causes runtime errors
   - Check actual data usage pattern carefully

3. **Wait mode confusion**:
   - `__deferrable_wait()`: Returns immediately, wait happens on next buffer access (default, preferred)
   - `.wait()`: Synchronous wait, blocks until completion
   - `.__no_wait()`: No synchronization (careful! only for pipelined operations)

4. **Template compilation errors**:
   - Long template instantiation chains make errors hard to read
   - Look for "candidate function" messages to find the actual issue
   - Check iterator categories, execution policy types first

5. **Test failures vs. skip**:
   - Exit code 77 means "skip" (e.g., no device available) - not a failure
   - Actual failures exit with non-zero code != 77

6. **Windows-specific issues**:
   - Debug builds use checked iterators by default (performance impact)
   - Tests automatically switch to release runtime in debug mode on Windows
   - Use Ninja generator (`-GNinja`) for best results

---

## Useful File Locations Quick Reference

### Core Algorithm Flow
| What | File |
|------|------|
| Public API declarations | `include/oneapi/dpl/pstl/glue_algorithm_defs.h` |
| Public API dispatch | `include/oneapi/dpl/pstl/glue_algorithm_impl.h` |
| Host patterns (TBB/OpenMP) | `include/oneapi/dpl/pstl/algorithm_impl.h` |
| Device patterns (SYCL) | `include/oneapi/dpl/pstl/hetero/algorithm_impl_hetero.h` |
| SYCL backend implementation | `include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h` |
| SYCL buffer utilities | `include/oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h` |

### Execution Policies
| What | File |
|------|------|
| Host execution policies | `include/oneapi/dpl/pstl/execution_defs.h` |
| Device execution policies | `include/oneapi/dpl/pstl/hetero/dpcpp/execution_sycl_defs.h` |

### Testing
| What | File |
|------|------|
| Test configuration | `test/support/test_config.h` |
| Test utilities | `test/support/utils.h` |
| CMake test setup | `test/CMakeLists.txt` |

### Build System
| What | File |
|------|------|
| Main CMake config | `CMakeLists.txt` |
| CMake usage guide | `cmake/README.md` |
| TBB discovery | `cmake/FindTBB.cmake` |

---

## Example: Complete Development Workflow

**Scenario**: Optimize the `transform` algorithm for a specific use case.

### 1. Investigation Phase
```bash
# Find the transform implementation
grep -r "__pattern_transform" include/oneapi/dpl/pstl/hetero/

# Read the implementation
# → include/oneapi/dpl/pstl/hetero/algorithm_impl_hetero.h
```

### 2. Understand Current Implementation
```cpp
// Check access modes used
// → Input: access_mode::read
// → Output: access_mode::write with _NoInit=true (good!)
// → Uses __pattern_walk2 with default wait mode
```

### 3. Identify Optimization Opportunity
```
# Hypothesis: Can we pipeline multiple transforms?
# Current: Each transform waits before returning
# Optimization: Use __no_wait for all but last transform
```

### 4. Create Test Case
```bash
# Create test file
cat > test/parallel_api/algorithm/transform_pipeline.pass.cpp << 'EOF'
#include "support/test_config.h"
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(execution)
#include "support/utils.h"

int main() {
    const int n = 1000;
    std::vector<int> input(n, 1);
    std::vector<int> temp(n);
    std::vector<int> output(n);

    auto policy = oneapi::dpl::execution::dpcpp_default;

    // Pipeline: two transforms
    oneapi::dpl::transform(policy, input.begin(), input.end(),
                          temp.begin(), [](int x) { return x * 2; });
    oneapi::dpl::transform(policy, temp.begin(), temp.end(),
                          output.begin(), [](int x) { return x + 1; });

    // Verify: 1 * 2 + 1 = 3
    EXPECT_TRUE(std::all_of(output.begin(), output.end(),
                           [](int x) { return x == 3; }),
                "transform pipeline incorrect");

    return TestUtils::done();
}
EOF
```

### 5. Build and Test
```bash
# Configure build
cmake -B build/sycl_cpp20 \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_STANDARD=20 \
  -DONEDPL_BACKEND=dpcpp

# Build test
cmake --build build/sycl_cpp20 --target transform_pipeline.pass

# Run test
cd build/sycl_cpp20
./transform_pipeline.pass

# Or use CTest
ctest -R transform_pipeline.pass --output-on-failure
```

### 6. Profile Performance
```bash
# Use VTune or similar
vtune -collect gpu-hotspots -result-dir vtune_results -- ./transform_pipeline.pass

# Or use SYCL event profiling (add to code)
```

### 7. Iterate on Implementation
```cpp
// Modify algorithm_impl_hetero.h
// Try different wait modes, access patterns, etc.
// Rebuild and retest
```

---

## Additional Resources

- **Documentation**: https://uxlfoundation.github.io/oneDPL
- **Samples**: https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDPL
- **GitHub Issues**: https://github.com/uxlfoundation/oneDPL/issues
- **Discussions**: https://github.com/uxlfoundation/oneDPL/discussions
- **Slack**: https://uxlfoundation.slack.com/channels/onedpl
- **oneAPI Spec**: https://oneapi-spec.uxlfoundation.org

---

## Contributing

See `CONTRIBUTING.md` for contribution guidelines.

**Key Points**:
- Follow existing code style and conventions
- Add tests for new features
- Update documentation as needed
- Run full test suite before submitting PR
- Sign DCO (Developer Certificate of Origin)

---

*This guide is maintained by the oneDPL development team. Last updated: 2025-01-18*
