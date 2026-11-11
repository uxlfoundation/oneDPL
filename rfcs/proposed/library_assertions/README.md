# Assertions

## Introduction

oneDPL extensively uses `assert`, which is enabled by default and disabled when `NDEBUG` is defined.
Other major C++ performance-oriented libraries define their own mechanisms decoupled from the standard `assert`.
Should oneDPL reconsider its use of `assert`?

There is also [P3471](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html) proposal, which overlaps with the internal assertion uses,
especially in the context of precondition checks, what oneDPL currently handles using `assert`.
Its goal is to improve safety and security.

Motivations:

1. Performance pitfalls due to the opt-out nature of `assert` and having expensive checks. Example:

    Sort pattern from the oneTBB backend had these checks (removed in [PR#2882](https://github.com/uxlfoundation/oneDPL/pull/2882)):

    ```cpp
    assert(std::is_sorted(_M_x_beg + _M_xs, _M_x_beg + _M_xs + __kx, _M_comp));
    assert(std::is_sorted(_M_x_beg + _M_ys, _M_x_beg + _M_ys + __ky, _M_comp));
    ```

    Omitting `-DNDEBUG` led to 2x slowdown of the sort algorithms.

2. Improve safety and security following the principles outlined in
[P3471](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html).

## oneDPL Assertions

| Cost | Count | Examples | Cost details | Risk of removal |
|---|---:|---|---|---|
| O(1) per work-item, device code | ~5 | Constructors and `size()` of `take_view_simple`, `drop_view_simple` in `utils_ranges.h`; e.g. `drop_view_simple` is constructed inside the global merge-sort kernel | N/A: SYCL does not generally support `assert` in device code - they must be removed | N/A |
| O(1) per TBB task | 5 | `parallel_backend_tbb.h`: `_M_execute_data != nullptr`, `__parent != nullptr`, `__parent->_M_refcount > 0`, `__next == nullptr` | Medium: a predictable branch and a relaxed load from a cache line that `--_M_refcount` touches right after, paid on every task; small next to task allocation and spawning | Low: a broken task tree crashes or hangs anyway; only the diagnostics get worse |
| O(1) per chunk or search step | ~5 | `__pattern_includes` (2 extra `__comp` calls per chunk), merge-path bounds in `std::lower_bound` | Low or Medium | Low |
| O(1) per call, SYCL device query | 2 | `get_device().has(sycl::aspect::atomic64)` | Low | KT algorithms trust the user, it should stay; the core algorithm should be fixed |
| O(1) per call | ~160 | Size checks, `__n > 0`, result bounds of `__parallel_set_op`, `optional::has_value()`, null checks | Negligible | Not evaluated |

Some assertions check documented preconditions, such as valid range sizes (see `__min_fn` `assert(std::ranges::size(__r) > 0);` check) and some check internal invariants (e.g., `assert(__first != __last);` from `__pattern_adjacent_difference`).

## Existing practice

Below is a summary of how different C++ libraries, which are somewhat related to oneDPL,
handle internal assertions and checks in general.

### [libstdc++](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_macros.html)

libstdc++ relies on the following macros:

| | `_GLIBCXX_ASSERTIONS` | `_GLIBCXX_DEBUG` | `_GLIBCXX_DEBUG_PEDANTIC` |
|---|---|---|---|
| Purpose | Lightweight precondition checks | Extensive debugging checks | Stricter checking within debug mode |
| Examples | Bounds checks, empty-container access, null smart-pointer dereference | Those checks, plus invalid iterators, invalid ranges, algorithm preconditions | Debug-mode checks, plus checks rejecting selected libstdc++ extensions and implementation-specific behavior |
| Cost | Generally low overhead | Additional time and memory; some checks scan entire ranges | Debug-mode overhead plus stricter checks |
| Container layout / ABI | Preserved | Changes affected containers and iterator types | Same as debug mode |
| Dependencies | Enabled automatically by `_GLIBCXX_DEBUG` | Includes `_GLIBCXX_ASSERTIONS` | Requires `_GLIBCXX_DEBUG` |
| Default | On when compiling without optimization, unless `_GLIBCXX_NO_ASSERTIONS` is defined | Off | Off |

### [libc++](https://libcxx.llvm.org/Hardening.html)

libc++ relies on hardening modes, which inspired [P3471](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html).

`_LIBCPP_HARDENING_MODE` has the following options:

| | `_LIBCPP_HARDENING_MODE_NONE` | `_LIBCPP_HARDENING_MODE_FAST` | `_LIBCPP_HARDENING_MODE_EXTENSIVE` | `_LIBCPP_HARDENING_MODE_DEBUG` |
|---|---|---|---|---|
| Purpose | Disable hardening | Lightweight hardening for production | Broader checking for production | Testing, CI, and development. Not intended for production |
| Checks | No checks | Security-critical preconditions | `FAST` plus additional checks for undefined behavior | All available checks, including internal invariant checks and heuristic checks |
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

MSVC STL relies on the following macros:

| | [`_ITERATOR_DEBUG_LEVEL`](https://learn.microsoft.com/en-us/cpp/standard-library/iterator-debug-level) | [`_MSVC_STL_HARDENING`](https://github.com/microsoft/STL/wiki/STL-Hardening) |
|---|---|---|
| Purpose | Checked iterators and iterator debugging, such as dereferencing `end()` or moving an iterators beyond the container bounds | Lightweight precondition checks intended for production |
| Values | `0`: disables checked iterators; `1`: enables checked iterators; `2`: enables iterator debugging | `0`: disabled; `1`: enabled |
| ABI | `1` and `2` change container and iterator layouts | Preserved |
| Default | `2` in Debug (`/MDd` or `/MTd`), `0` in Release (`/MD` or `/MT`) | Off |
| Restrictions | `2` is an error in Release | None |
| Per-class controls | None | `_MSVC_STL_HARDENING_<CLASS>` macros default to `_MSVC_STL_HARDENING` |

### [oneTBB](https://github.com/uxlfoundation/oneTBB/blob/master/doc/main/reference/source/configuration/enabling_debugging_features.rst)

oneTBB relies on the following macros:

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

## Proposal

Short-term: do not add assertions with more than O(1) cost per call.
After [PR#2882](https://github.com/uxlfoundation/oneDPL/pull/2882), no host-side assertion has a noticeable cost.
Remove the assertions reachable from device code, since SYCL does not support them.

Long-term:
wait until [P3471](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html) settles, and potentially adopt it.
Look for the new paper that defines the behaviour for algorithms.
