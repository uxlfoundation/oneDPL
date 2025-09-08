# Support the second portion of the oneDPL Range APIs

## Introduction
Based on statistics (observing C++ code within github.com) for the usage of popular algorithms, the following
range-based APIs are suggested to be implemented next in oneDPL.
`fill`, `move`, `replace`, `replace_if`, `remove`, `remove_if`, `mismatch`, `minmax_element`, `minmax`,
`min`, `max`, `find_first_of`, `find_end`, `is_sorted_until`

## Motivations
The feature is proposed as the next step of range-based API support for oneDPL.

### Key Requirements
- The range-based signatures for the mentioned API should correspond to the [proposed specification](https://github.com/uxlfoundation/oneAPI-spec/pull/614)
that is based on the [C++ standardization proposal P3179](https://wg21.link/p3179).
- The proposed implementation should support all oneDPL execution policies: `seq`, `unseq`, `par`, `par_unseq`, and `device_policy`.

### Implementation proposal
- The implementation is supposed to rely on existing range-based or iterator-based algorithm patterns, which are already
implemented in oneDPL.
- Several algorithms described in P3179 have slightly different semantics. To implement these, some existing algorithm patterns
might require modifications or new versions.

### Implementation limitation
- In case of a `device_policy` and `std::vector` with `USM` allocator, `std::vector` cannot be passed into algorithm directly because a `std::vector` is not [SYCL device_copyable](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec::device.copyable). To support  `std::vector` with `USM` allocator `std::vector` should be wrapped into `std::ranges::subrange`.
- In case of a `device_policy` the projections pointer-to-member and pointer-to-function are not supported, for SYCL backend at least.

### Test coverage
- If a range-based algorithm shares its implementation with an iterator-based variant that is sufficiently tested,
  the range of input sizes, element types, and data distributions tested for the range-based version can be reduced.
  However, if the range-based algorithm performs any additional processing,
  such as handling trivial cases before delegating to the shared implementation,
  those scenarios must be tested explicitly.
- Testing should cover a range of input sizes, from hundreds to millions of elements, to ensure parallel execution
  (where applicable, based on the execution policy). Smaller sizes may be used to verify semantics.
- Output data, return type, and value should be checked/compared with the reference result
computed by the corresponding serial `std::ranges` algorithm or by a custom implemented serial version
in case of different semantics.
- The tests should also call the algorithms with following standard range adapters: `std::ranges::subrange`, `std::span`, `std::views::all`,
  `std::views::iota`, `std::views::transform`, `std::views::reverse`, `std::views::take`, `std::views::drop`
- The tests should also call the algorithms with default and custom predicates, comparators and projections.
- In case of a `device_policy` and `std::vector` with USM allocator, the algorithms accept the vector wrapped into `std::ranges::subrange` or `std::span`.

