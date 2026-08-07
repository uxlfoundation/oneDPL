# Parallel Range Algorithms

## Introduction
All parallel range algorithms described in the oneDPL Specification v1.5 have been implemented.

## Motivations
- Better expressiveness and productivity.
- Opportunity to fuse several parallel algorithm invocations into one.

### Key Requirements
- The range-based signatures for the mentioned API correspond to
  [Parallel Range Algorithms Specification](https://uxlfoundation.github.io/oneDPL/specification/parallel_api/parallel_range_api.html).
- The implementation supports all oneDPL execution policies:
  `seq`, `unseq`, `par`, `par_unseq`, and `device_policy`.
- `ONEDPL_HAS_RANGE_ALGORITHMS` macro is added to detect algorithms available in a specific release.

### Implementation
The implementation relies on the existing
range-based patterns (the experimental parallel range algorithms with device execution policies) or
iterator-based patterns (the remaining algorithms) for the majority of algorithms.

These algorithms need new patterns or significantly modifying the existing ones:
`merge`,  `copy_if`, `unique_copy`, `partition_copy`,
`set_union`, `set_difference`, `set_symmetric_difference`, `set_intersection`.
They must stop execution when the output sequence is exhausted and return the last processed points,
and these points cannot be calculated in advance, before the main algorithmic routine.
`merge` already implements it. The remaining algorithms also must support this case.

### Implementation limitation
- In case of a `device_policy` and `std::vector` with `USM` allocator,
  `std::vector` cannot be passed into algorithm directly because a `std::vector` is not
  [SYCL device_copyable](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec::device.copyable).
  To support `std::vector` with `USM` allocator
  `std::vector` should be wrapped into `std::ranges::subrange`.
- In case of a `device_policy`, the projections pointer-to-member and pointer-to-function
  are not supported for the SYCL backend due to limitations of use in SYCL kernels.

### Test coverage
- If a range-based algorithm shares its implementation with an iterator-based variant
  that is sufficiently tested,
  the range of input sizes, element types, and data distributions tested
  for the range-based version can be reduced.
  However, if the range-based algorithm performs any additional processing,
  such as handling trivial cases before delegating to the shared implementation,
  those scenarios must be tested explicitly.
- Testing should cover a range of input sizes,
  from hundreds to millions of elements, to ensure parallel execution
  (where applicable, based on the execution policy). Smaller sizes may be used to verify semantics.
- Output data, return type, and value should be checked/compared with the reference result
  computed by the corresponding serial `std::ranges` algorithm or
  by a custom implemented serial version in case of different semantics.
- The memory algorithms should be tested with `std::ranges::subrange` and `std::span`
  adapters which can be used with manually allocated and managed storage.
- Other algorithms should be tested with following standard range adapters:
  `std::ranges::subrange`, `std::span`, `std::views::all`, `std::views::iota`,
  `std::views::transform`, `std::views::reverse`, `std::views::take`, `std::views::drop`.
- The tests should also call the algorithms with default and custom predicates,
  comparators and projections.
- In case of a `device_policy` and `std::vector` with USM allocator,
  the algorithms accept the vector wrapped into `std::ranges::subrange` or `std::span`.

### Performance
- No performance regression compared to the equivalent iterator-based algorithms.
  If a range-based algorithm implements some additional logic,
  for example supporting a limited output sequence,
  then the acceptable overhead is to be determined on per-algorithm basis.
