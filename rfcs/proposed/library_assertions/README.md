# Assertions

## Introduction

oneDPL extensively uses `assert` for internal checks, including expensive ones.
Asserts are enabled by default, and disabled when `NDEBUG` is defined.
It can be considered a performance pitfall,
which is important to avoid in a performance-oriented library.

### Motivational Example

Sort pattern from the oneTBB backend had these checks (removed in [PR#2882](https://github.com/uxlfoundation/oneDPL/pull/2882)):

```cpp
assert(std::is_sorted(_M_x_beg + _M_xs, _M_x_beg + _M_xs + __kx, _M_comp));
assert(std::is_sorted(_M_x_beg + _M_ys, _M_x_beg + _M_ys + __ky, _M_comp));
```

They were expensive.
In practice, omitting `-DNDEBUG` led to 2x slowdown of the baseline.

## Existing practice

Below is a summary of how different C++ libraries, which are somewhat related to oneDPL,
handle internal assertsions and checks in general.

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

libc++ also had `_LIBCPP_DEBUG` and `_LIBCPP_ASSERT` macros, which were removed in libc++17.

### MSVC STL

MSVC STL does not use `assert` and relies on the following macros:

| | [`_ITERATOR_DEBUG_LEVEL`](https://learn.microsoft.com/en-us/cpp/standard-library/iterator-debug-level) | [`_MSVC_STL_HARDENING`](https://github.com/microsoft/STL/wiki/STL-Hardening) |
|---|---|---|
| Purpose | Checked iterators and iterator debugging, such as dereferencing `end()` or moving an iterators beyond the container bounds | Lightweight precondition checks intended for production |
| Values | `0`: disables checked iterators; `1`: enables checked iterators; `2`: enables iterator debugging | `0`: disabled; `1`: enabled |
| ABI | `1` and `2` change container and iterator layouts | Preserved |
| Default | `2` in Debug (`/MDd` or `/MTd`), `0` in Release (`/MD` or `/MT`) | Off |
| Restrictions | `2` is an error in Release | None |
| Per-class controls | None | `_MSVC_STL_HARDENING_<CLASS>` macros default to `_MSVC_STL_HARDENING` |

### [oneTBB](https://github.com/uxlfoundation/oneTBB/blob/master/doc/main/reference/source/configuration/enabling_debugging_features.rst)

oneTBB does not use `assert` and relies on the following macros:

| | `TBB_USE_DEBUG` | `TBB_USE_ASSERT` | `__TBB_ASSERT_RELEASE` (undocumented) |
|---|---|---|---|
| Purpose | Sets defaults for `TBB_USE_ASSERT` and `TBB_USE_PROFILING_TOOLS`; enables additional debug code (besides assertions) | Internal assertion checking | Internal assertion checking |
| Cost | Can significantly slow down performance | Can significantly slow down performance | ? |
| Default | `1` on Windows if `_DEBUG` is defined (`/MDd` or `/MTd`), `0` otherwise | Value of `TBB_USE_DEBUG` | Always enabled, regardless of `TBB_USE_ASSERT`, `TBB_USE_DEBUG`, or `NDEBUG` |

oneTBB also supports a [custom assertion handler](https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2023-2/custom-assertion-handler.html#GUID-E04E99E7-BE0C-46F3-BF58-7D62D7451B4F)
via `set_assertion_handler` to customize the action on failure.
The default handler prints an error message and terminates the application.

### [CCCL](https://github.com/NVIDIA/cccl/blob/main/docs/cccl/config_macros.rst) (Thrust, CUB, libcu++)

| | `CCCL_ENABLE_ASSERTIONS` | `CCCL_ENABLE_HOST_ASSERTIONS` | `CCCL_ENABLE_DEVICE_ASSERTIONS` |
|---|---|---|---|
| Purpose | Host and device checks | Host checks, independent of debug mode | Device checks, independent of debug mode |
| Default | Enabled when compiling in debug mode | Implied by `CCCL_ENABLE_ASSERTIONS` | Implied by `CCCL_ENABLE_ASSERTIONS` |

Thrust uses `assert` in some places; CUB does not.

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

TBD

## Open Questions

1. **Should the macros have "assert" or "debug" or even "hardening" semantics?**

    - **Assert** implies that the checks follow the `assert` rules, e.g., cause an abort.
    - **Debug** implies that the checks are intended for development and not production.
    - **Hardening** implies that the checks are intended to enforce security and robustness,
    potentially in production as well.

    oneTBB's `TBB_USE_DEBUG` and `TBB_USE_ASSERT` look like different levels of checks.
    The same applies to libstdc++'s `_GLIBCXX_DEBUG` and `_GLIBCXX_ASSERTIONS` macros.

   **Tentative answer:** Follow **Hardening** semantics.
   It will impact the name/names of the macros.

2. **Should there be one or multiple layers of checks?**

    Separate layers for cheap and expensive checks could let users choose the acceptable overhead.
    However, multiple layers add maintenance cost: they need to be documented,
    and developers need to classify checks consistently.
    The current use of assertions in oneDPL have differnt costs, serve different purposes,
    and there are no developer guidelines.

   **Tentative answer:** ?

3. **Should internal and user-facing checks be controlled separately?**

    Internal checks help developers detect implementation errors,
    while user-facing checks diagnose violations of algorithm preconditions.
    Some checks serve both purposes.

   **Tentative answer:** ?

4. **Should debug checks be inferred from compiler flags or CMake build configurations?**

    MSVC STL and oneTBB infer debug settings from compiler flags on Windows.
    Such platform-specific defaults are not portable and can lead to inconsistent behavior
    and more complicated configuration rules.
    CMake provides portable build configurations, but users
    may also want to choose debug compilation and oneDPL assertions independently.

    **Tentative answer:** Do not infer the checks and their levels; make them explicit.

5. **Should there be a dedicated CMake option to enable checks?**

    An option could make configuration more discoverable,
    but would duplicate the macro.

    Users can already define the macro with `target_compile_definitions`
    or pass it through `CMAKE_CXX_FLAGS`.

    **Tentative answer:** Do not add a dedicated CMake option.

6. **Should host and device checks be controlled separately?**

    In oneDPL's SYCL backend, users may want to enable checks in host-side setup
    independently of checks in device kernels.
    oneDPL has no device checks, and it is questionable if they are feasible.

    **Tentative answer:** Do not separate host and device checks for now.
