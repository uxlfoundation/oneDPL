# Parallel Range Algorithms

## Introduction
The following range algorithms have been implemented in oneDPL:
- Whole Sequence Operations: `all_of`, `any_of`, `none_of`, `for_each`, `count`, `count_if`.
- Element Search Operations: `find`, `find_if`, `find_if_not`, `find_first_of`, `adjacent_find`.
- Minimum and Maximum: `min`, `max`, `minmax`, `min_element`, `max_element`, `minmax_element`.
- Sequence Search and Comparison: `equal`, `mismatch`, `find_end`, `search`, `search_n`.
- Sorting and Merge: `sort`, `stable_sort`, `is_sorted`, `is_sorted_until`, `merge`.
- Set operations: `includes`, `set_intersection`, `set_union`, `set_difference`,
  `set_symmetric_difference`.
- Copying mutating algorithms: `copy`, `copy_if`, `move`, `reverse_copy`, `transform`,
  `unique_copy`.
- In-place mutating algorithms: `fill`, `replace`, `replace_if`, `remove`, `remove_if`, `reverse`,
  `swap_ranges`, `unique`.
- Uninitialized Memory Algorithms: `uninitialized_copy`, `uninitialized_move`, `uninitialized_fill`,
  `uninitialized_default_construct`, `uninitialized_value_construct`, `destroy`.

The remaining algorithms (as defined in [P3179](https://wg21.link/p3179))
will be implemented in the future releases.

## Motivations
- Better expressiveness and productivity.
- Opportunity to fuse several parallel algorithm invocations into one.

### Key Requirements
- The range-based signatures for the mentioned API correspond to
  [Parallel Range Algorithms Specification](https://github.com/uxlfoundation/oneAPI-spec/blob/main/source/elements/oneDPL/source/parallel_api/parallel_range_api.rst).
- The implementation supports all oneDPL execution policies:
  `seq`, `unseq`, `par`, `par_unseq`, and `device_policy`.
- `ONEDPL_HAS_RANGE_ALGORITHMS` macro is added to detect available algorithms.

### Implementation
The implementation relies on the existing
range-based patterns (the experimental parallel range algorithms with device execution policies) or
iterator-based patterns (the remaining algorithms) for the majority of algorithms.

These algorithms need new patterns or significantly modifying the existing ones:
`merge`,  `copy_if`, `unique_copy`,
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
