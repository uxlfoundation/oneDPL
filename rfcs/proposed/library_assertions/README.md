# Assertions

## Introduction

oneDPL extensively uses `assert` for internal checks, including expensive ones.
Asserts are enabled by default, and disabled when `NDEBUG` is defined.
It can be considered a performance pitfall,
which is important to avoid in a performance-oriented library.

### Motivational Example

Sort pattern from the oneTBB backend has these checks:

```cpp
assert(::std::is_sorted(_M_x_beg + _M_xs, _M_x_beg + _M_xs + __kx, _M_comp));
assert(::std::is_sorted(_M_x_beg + _M_ys, _M_x_beg + _M_ys + __ky, _M_comp));
```

They are expensive.
In practice, omitting `-DNDEBUG` led to 2x slowdown of the baseline.

## Existing practice

Below is a summary of how different C++ libraries, which are somewhat related to oneDPL,
handle internal assertions.

### [libstdc++](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_macros.html)

libstdc++ does not use `assert` and relies on the following macros:

| | `_GLIBCXX_ASSERTIONS` | `_GLIBCXX_DEBUG` | `_GLIBCXX_DEBUG_PEDANTIC` |
|---|---|---|---|
| Purpose | Lightweight precondition checks | Extensive debugging checks | Stricter checking within debug mode |
| Examples | Bounds checks, empty-container access, null smart-pointer dereference | Those checks, plus invalid iterators, invalid ranges, algorithm preconditions | Debug-mode checks, plus checks rejecting selected libstdc++ extensions and implementation-specific behavior |
| Cost | Generally low overhead | Additional time and memory; some checks scan entire ranges | Debug-mode overhead plus stricter checks |
| Container layout / ABI | Preserved | Changes affected containers and iterator types | Same as debug mode |
| Dependencies | Enabled automatically by `_GLIBCXX_DEBUG` | Includes `_GLIBCXX_ASSERTIONS` | Requires `_GLIBCXX_DEBUG` |
| Default | On when compiling without optimization, unless `_GLIBCXX_NO_ASSERTIONS` is defined | Off | Off |

### [libc++](https://libcxx.llvm.org/Hardening.html)

libc++ does not use `assert`; It adds hardening modes, which inspired [C++26 Hardening](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html).

`_LIBCPP_HARDENING_MODE` have the following options:

| | `_LIBCPP_HARDENING_MODE_NONE` | `_LIBCPP_HARDENING_MODE_FAST` | `_LIBCPP_HARDENING_MODE_EXTENSIVE` | `_LIBCPP_HARDENING_MODE_DEBUG` |
|---|---|---|---|---|
| Purpose | Disable hardening | Lightweight hardening for production | Broader checking for production | Testing, CI, and development. Not intended for production |
| Checks | No checks | Security-critical preconditions | `FAST` plus additional checks for undefined behavior | All available checks, including **internal invariant checks** and heuristic checks |
| Examples | None | Container bounds, empty-container access, input-range validity | `FAST` checks plus null pointers, overlapping ranges, incompatible allocators | `EXTENSIVE` checks plus internal invariants and heuristic semantic-requirement checks |
| Cost | No checking overhead | Relatively little overhead; constant-time checks | Additional overhead beyond `FAST` | Potentially significant overhead; expensive heuristic checks |
| ABI | Preserved | Preserved | Preserved | Preserved |
| Default | Upstream default; vendors may change it | Opt-in or vendor-selected | Opt-in or vendor-selected | Opt-in or vendor-selected |

`_LIBCPP_ASSERTION_SEMANTIC` (experimental) controls the action on failure:

- `ignore`
- `observe`: log the failure but continue execution.
- `quick-enforce`: terminate the program immediately via a trap. Default for `FAST` and `EXTENSIVE`.
- `enforce`: log the failure and terminate the program. Default for `DEBUG`.

### MSVC STL

- [`_ITERATOR_DEBUG_LEVEL`](https://learn.microsoft.com/en-us/cpp/standard-library/iterator-debug-level): 0 disables checked iterators, 1 enables checked iterators, 2 enables iterator debugging. Defaults to 2 in Debug and 0 in Release; 2 is an error in Release.
- [`_MSVC_STL_HARDENING`](https://github.com/microsoft/STL/wiki/STL-Hardening): lightweight precondition checks intended for production. Off by default. Per-class `_MSVC_STL_HARDENING_<CLASS>` macros default to its value.
- Uses `assert`: no.

### [oneTBB](https://github.com/uxlfoundation/oneTBB/blob/master/doc/main/reference/source/configuration/enabling_debugging_features.rst)

- `TBB_USE_DEBUG`: default for `TBB_USE_ASSERT`. 1 on Windows if `_DEBUG` is defined, 0 otherwise.
- `TBB_USE_ASSERT`: internal assertion checking, which "can significantly slow down performance". Defaults to `TBB_USE_DEBUG`. Some assertions remain in release builds.
- Uses `assert`: no.

### [CCCL](https://github.com/NVIDIA/cccl/blob/main/docs/cccl/config_macros.rst) (Thrust, CUB, libcu++)

- `CCCL_ENABLE_ASSERTIONS`: host and device checks. Implied by compiling in debug mode.
- `CCCL_ENABLE_HOST_ASSERTIONS`: host checks, independent of debug mode. Implied by `CCCL_ENABLE_ASSERTIONS`.
- `CCCL_ENABLE_DEVICE_ASSERTIONS`: device checks, independent of debug mode. Implied by `CCCL_ENABLE_ASSERTIONS`.
- The macros must be defined consistently in the whole project.
- Uses `assert`: yes in Thrust, no in CUB.

### [HPX](https://github.com/STEllAR-GROUP/hpx/blob/master/docs/sphinx/manual/building_hpx.rst)

- CMake `debug` mode, when building HPX, enables "additional" assertions.
- Uses `assert`: no, except for some rare cases.

### Kokkos

- [`NDEBUG`](https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/API/core/utilities/assert.rst): disables `KOKKOS_ASSERT`, unless `KOKKOS_ENABLE_DEBUG` is defined.
- [`KOKKOS_ENABLE_DEBUG`](https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/API/core/utilities/assert.rst): enables `KOKKOS_ASSERT` regardless of `NDEBUG`. On by default for the Debug build type.
- [`Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html): bounds checking, which "will increase runtime". Off by default.
- Uses `assert`: no.

### Summary

- These libraries do not rely on `NDEBUG`: each provides its own macros to control its checks.
  They allow enabling their assertions even if `NDEBUG` is defined.
- Several libraries tie default checks to a debug build, detected by the build type,
  `_DEBUG`, or the absence of optimization. libc++ vendors choose the default hardening mode.
- Expensive checks are opt-in: libstdc++ debug mode, the libc++ debug hardening mode,
  bounds checking in Kokkos.
- Some provide several levels of checking (libstdc++, libc++, MSVC STL, Kokkos).

## oneDPL Assertions

### Internal vs User-Facing Assertions

- `assert(__parent != nullptr)` from task logic in merge operations.
- `assert(std::ranges::size(__r) > 0);` in `dpl::ranges::min`,
  a condition which violation leads to UB by the C++ standard.
- `assert(::std::is_sorted(_M_z_beg + _M_xs, _M_z_beg + _M_xs + __kx, _M_comp));`
  expects user input to be sorted and also depends on the internal state of the algorithm.

### Expensive vs Cheap Assertions

| Cost | Count | Where |
|---|---:|---|
| O(n) | 4 | parallel_backend_tbb.h:878, 879, 883, 884, std::is_sorted in __merge_func::x_less_y() |
| Runtime SYCL query | 2 | get_device().has(sycl::aspect::atomic64) at parallel_backend_sycl.h:1231 and kt/single_pass_scan.h:252. One API call per algorithm invocation; not expensive, but not free. |
| O(1) on a hot path | ~5 | See below |
| O(1), once per call | ~163 | Size checks, __n > 0, merge-path bounds, optional::has_value(), null checks, std::ranges::size on sized ranges, std::distance on random-access iterators |

## Proposal

1. Add a macro `ONEDPL_USE_ASSERT` to control internal assertions.
2. Have a single level of internal assertions.
3. Do not enable assertions automatically based on platform-specific flags.
4. Do not enable assertions automatically for the CMake `Debug` build type.
5. Do not add CMake options to control `ONEDPL_USE_ASSERT`.
6. Set `ONEDPL_USE_ASSERT=1` in the CI, except for the `Release` build.
7. Document the macro for both library headers and the CMake build system.

(1) The naming follows the oneTBB convention, another oneAPI library.
If `ONEDPL_USE_ASSERT` is`0` or undefined, the assertions are disabled, otherwise they are enabled.
There is no dependency on `NDEBUG`,
hence `ONEDPL_USE_ASSERT` should be implemented without `assert`,
but its behaviour should match the `assert`'s behaviour.

(2) is because the use of internal assertions in oneDPL is not uniform,
there are no developer guidelines,
so levels needs to be clearly defined and consistently maintained.

(3) is because enabling assertions automatically
based on platform-specific flags is not portable.
Lack of portability leads to inconsistency, and more complicated rules.

(4) CMake is portable, however a user may want to differentiate
between the debug build type and the enabling of assertions in oneDPL specifically.

(5) CMake option duplicates the macro. The users can set `target_compile_definitions`
or pass the macro definition via `CMAKE_CXX_FLAGS`.
