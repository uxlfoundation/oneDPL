# Support the third portion of the oneDPL Range APIs

## Introduction
Based on statistics (observing C++ code within github.com) for the usage of popular algorithms,
the following range-based APIs are suggested to be implemented next in oneDPL:
- Set algorithms: `includes`, `set_intersection`, `set_union`, `set_difference`, `set_symmetric_difference`
- In-place mutating algorithms: `reverse`, `swap_ranges`, `unique`.
- Memory algorithms: `uninitialized_copy`, `uninitialized_move`, `uninitialized_fill`,
  `uninitialized_default_construct`, `uninitialized_value_construct`, `destroy`.
- Copying mutating algorithms: `reverse_copy`, `unique_copy`.

## Motivations
The feature is proposed as the next step of range-based API support for oneDPL.

### Key Requirements
- The range-based signatures for the mentioned API should correspond to the proposed specification,
  which itself is based on the [P3179](https://wg21.link/p3179) proposal:
  - [Set algorithms](https://github.com/uxlfoundation/oneAPI-spec/pull/630).
  - [Memory algorithms](https://github.com/uxlfoundation/oneAPI-spec/pull/631).
  - [In-place mutating algorithms](https://github.com/uxlfoundation/oneAPI-spec/pull/634).
  - [Copying mutating algorithms](https://github.com/uxlfoundation/oneAPI-spec/pull/635).
- The signature of `reverse_copy` should align with [P3709](https://wg21.link/p3709),
  which updates the signature defined in [P3179](https://wg21.link/p3179).
- The proposed implementation should support all oneDPL execution policies:
  `seq`, `unseq`, `par`, `par_unseq`, and `device_policy`.
- To add a new value for the feature testing macro
  `ONEDPL_HAS_RANGE_ALGORITHMS` in oneDPL documentation.

### Implementation proposal
The implementation is supposed to rely on existing range-based or iterator-based algorithm patterns,
which are already implemented in oneDPL.

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
