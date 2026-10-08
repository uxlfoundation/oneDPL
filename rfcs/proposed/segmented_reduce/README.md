# Offset-Based Segmented Reduction

## Introduction

RFC proposes adding offset-based segmented reduction to oneDPL.

Motivation:

- Migration from `cub::DeviceSegmentedReduce`.
  SYCLomatic already has `dpct::device::segmented_reduce`,
  but it is a helper rather than a fully-supported library API.
- [Real-world use cases](usage_pattern_study.md), including [#2829](https://github.com/oneapi-src/oneDPL/issues/2829) request with its own motivation.

See the appendix sections for the context on the design considerations below.

## Design Considerations

### Kernel Template or Core Algorithm Extension

Should the algorithm be implemented as a core extension algorithm, a kernel template, or both?

`dpl::reduce_by_segment` is already available as a core extension algorithm.
The new algorithm fits naturally as a sibling to `dpl::reduce_by_segment` with slightly different semantics.

The corresponding kernel template algorithms can be developed later.
The implementation experience can guide the design of the tuning parameters for the kernel templates.
Passing the temporary storage is an obvious gap,
but the corresponding interface in kernel templates has not yet been defined.

### Initial Value, its Default Value, and Identity

The [#2829](https://github.com/oneapi-src/oneDPL/issues/2829) request
specified an identity as the last argument,
while the industry practice it is typically if not always an initial value.

What should be provided as an argument in place of the initial value? What should be the default value?

Initial value applies only once, and known identity can be applied multiple times.
The initial value may be equal to the known identity or not. A safe default initial value is the identity.

Initial value is always needed as the starting point for the accumulator.
Identity value is optional. It is needed for the SIMD and the group reductions,
and to get a safe default initial value.

It may be inferred from the most typical operations and data types, for example:

| Operation | Identity |
|---|---|
| add | 0 |
| multiply | 1 |
| logical_and | true |
| logical_or | false |
| logical_xor | false |
| minimum | Maximum representable value (max for ints, infinity for floats) |
| maximum | Minimum representable value (min for ints, negative infinity for floats) |

A custom operation may or may not have an identity,
and may depend on the context such as the data value range.

How oneDPL may use an identity:
- In OpenMP reductions, the identity can be a runtime value (for user defined reductions),
  or a predefined type and an operation combination (built-in reductions).
- In SYCL group algorithms, it must be a predefined combination of a type and an operation,
  although a runtime value may be used for manual optimizations.

The algorithm (especially in a performance-oriented implementation)
should provide a way to specify the identity explicitly. There are many strategies to do this.
The topic is discussed in [p3732r2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3732r2.html),
although there is no consensus yet.

**Strategy**. Keep initial value as an argument with an "initial value" semantics.
Wait until [p3732r2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3732r2.html) settles,
and follow it to provide a way to specify the identity
for better performance with custom types and operations.
Request the initial value to be passed explicitly by the user.
Provide a default value when, again,
[p3732r2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3732r2.html) settles.

### Sum, Min, Max, ArgMin and ArgMax semantics

All semantics are used according to the [real-world use cases](usage_pattern_study.md).
Therefore, they should be supported.

`Sum`, `Min`, and `Max` semantics can be enabled by
specifying the appropriate binary operation and initial value for the reduction.
no need to introduce special overloads for the `reduce_by_segment` algorithm, complicating the API.

`ArgMin` and `ArgMax` is a special case, because they reduce key-value pairs instead of just values.
Algorithmically it does the following:

1. Pack the input into `zip(counting_iterator, input_first)`.
2. Do the regular segmented reduction
  with a custom binary operation which also checks the order of the arguments.
3. When writing the final result, normalize the index: `result.key -= begin_offsets[segment];`

Given that there options on how to handle `ArgMin` and `ArgMax`:

1. Introduce special functors, e.g., `dpl::argmin` and `dpl::argmax`,
   use the same overload, and customize the behaviour accordingly.
2. Introduce a fancy iterator with `enumerate` semantics,
   add `dpl::argmin` and `dpl::argmax` functors that work with it,
   and to trigger the normalization of the indices.
3. Provide special overloads, and name after min and max element functions, e.g.,
   `min_element_by_segment` and `max_element_by_segment`.

Assuming that the customizations "spoil" the interface, option (3) is preferable.

`min_element` and `max_element` return iterators to the input range,
what should their segmented counterparts do?
`min_element` and `max_element` receive iterators to the input range and return iterators.
Given that the segmented versions receive segment offsets as indices,
returning segment-local indices is natural.

Should the binary operation (comparator) and the initial value be allowed to be specified
with `min_element_by_segment` and `max_element_by_segment`?
What should they return in case of an empty segment?
`min_element` and `max_element` allow passing a comparator. The initial value is not applicable there.
`cub::DeviceSegmentedReduce` does not allow passing them directly.
It uses `<` or `>` as the comparator, and for an empty segment returns an identity,
`{1, cuda::std::numeric_limits<T>::max()}` for `min_element_by_segment` and
`{1, cuda::std::numeric_limits<T>::min()}` for `max_element_by_segment`.
It is not clear why `1` is provided as an index for an empty segment.
What should be done:

- Provide an interface without comparator and no initial value.
  The initial value will be:
  `{OffsetT{}, std::numeric_limits<T>::max()}` for `min_element_by_segment` and
  `{OffsetT{}, std::numeric_limits<T>::min()}` for `max_element_by_segment`.
  The comparator will be `std::less{}` or `std::greater{}` for alignment with the
  corresponding C++ functions.
- Provide an interface with a comparator and an initial value.

Returning both, although possible and potentially useful,
but it will have a trade-off with extra memory traffic when writing the final result,
which should be avoided unless explicitly asked for.

**Strategy**:

- Do nothing special for `Sum`, `Min`, and `Max` semantics.
- For `ArgMin` and `ArgMax`, provide `min_element_by_segment` and `max_element_by_segment`.

### Bounding Input

`cub::DeviceSegmentedReduce` (variable-segments) and `rocprim::segmented_reduce`
do not provide the end bound for the input range explicitly.
The bound is implicitly determined by the segment offsets and the number of segments.
Moreover, these segments may have gaps and be unordered.

It has two issues:
1. Safety: there is a risk of accessing out-of-bounds elements.
  Although it is user's responsibility to provide correct offset sizes and sufficient input range,
  it is still a hazard.
2. Performance: the only information guaranteed to be available on the host
  to distribute the workload is the number of segments.
  It either forces the implementation to assign a segment to a work-group,
  which is not efficient with varying segment sizes,
  and limits batching opportunities for better cache utilization,
  or it involves additional checks to understand the actual input data space.

The interfaces must have the end bound to address these issues.

If the end bound is provided, it creates another question about how to
reconcile the input range size provided with the segment offsets,
because both define the reduction range. Options:

1. Handle the minimum between `last - first` and offsets including gaps.
  In this case, we will need to return stop positions in the input, output and offset arrays.
2. Always handle all segments defined by offsets,
  require `num_elements` to be covering all offsets including gaps.

Option (2) is preferred due to its alignment with expectations by the iterator-based interfaces.
The bound checking in (1) can be implemented in the range-based interfaces if needed.

In the fixed-length segment scenario, the output bound should also be provided for safety reasons.

**Strategy**: provide the end bounds for both variable-length and fixed-length segments.

### Fixed-Length Segments

Is this overload necessary given that the variable-length segment overload can handle all cases?

Yes, because this semantics is already used and due to the performance benefits:

- Select the reduction on host before the kernel submission
  (per work-item, sub-group, work-group or multi-work-group with multiple kernels)
- Compute segment boundaries arithmetically, avoiding offset arrays and their memory traffic.
- Derive tile counts and partial-result locations directly,
  avoiding segment identifiers in multi-kernel implementations.

It's much easier to emulate via existing key-based segmented reduction
via counting and discard iterators,
but it is not as efficient due to segment boundary checks.

**Strategy**: provide the fixed-length segment overloads.

### Binary Operator

Associativity is essential for parallelization.
Commutativity is useful for example, when doing sub-group and work-group striding,
a common approach in GPU programming to improve memory coalescing.

There are different associativity and commutativity requirements in different implementations:

- `cub::DeviceSegmentedReduce`, `rocprim::segmented_reduce`, `dpct::device::segmented_reduce`: both.
- `dpl::reduce_by_segment`: associativity only.
- For the reference, `std::reduce` and `std::transform_reduce` and their oneDPL equivalents require both.

Why is not commutativity required for key-based segmented reduction?
Is it an oversight, or is there a specific reason? Does it come at the cost?
Its requirements should not relaxed straight away, because of being a breaking change,
but the new algorithm may require commutativity immediately.

**Tentative Strategy**: require commutativity.

### Floating Point Determinism

Floating point operations are generally non-deterministic when applied in different orders,
that is they are not associative.

`cub::DeviceSegmentedReduce` environment allows specifying these guarantees:

- no determinism
- run-to-run (default)
- cross-device

The determinism requires adding separate, and likely more complex and less efficient algorithms.
Cross-device determinism may not even be achievable in practice.
Run-to-run determinism question is still open.

Extensive research: [P4229R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4229r0.pdf).

**Tentative Strategy**:

- Do not guarantee determinism, even the weakest run-to-run one.
- Do our best to achieve run-to-run determinism where possible,
  and document where it is not present (may be policy-dependent).

### Overload Disambiguation

The algorithm can be positioned as a sibling of `dpl::reduce_by_segment`,
which means that it should have the same naming if possible.
It should be kept in mind that the proposed overloads
may introduce default values for the binary operation and the initial value.

Overload resolution should be carefully considered to avoid ambiguity.

In the proposed overloads,
the distinction is made by constraining the `num_segments` to an integral type.
In the existing overload, it takes the place of the `InputValueIt val_first` argument,
which is semantically required to be an iterator.

**Strategy**. Constrain arguments which require integral types to avoid ambiguity.

### Accumulator Type

The accumulator type determines the representation of intermediate reduction results.

This type can be influenced by:

- initial value type
- input value type
- output value type
- type inferred from the binary operation

Consider these cases:
- The accumulator type may differ from both the input and output value types.
  For example, having `double` accumulator for `float`
  inputs and outputs improves accuracy and reduces memory traffic.
- Input can hold smaller values than the output type,
  for example, input is `float` or `int` and output is `double` or `int64_t` respectively.
  Then the accumulator type must be selected from the output.
- An initial value is lazily passed as `0` (`int`) when the output is `int64_t`.
  The accumulator types must be `int64_t`.
- An initial value is lazily passed as `0` (`int`) when the output is `double`.
  The accumulator types must be `double`.

The safest approach is use the common type from all involved types as the accumulator type.
For the case with `float` and `double`, the common type would be `double`.

The documentation does not provide explicit rules for the accumulator type. Algorithms checked:

- [CUB `DeviceSegmentedReduce::Reduce`](<https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceSegmentedReduce.html>)
- [rocPRIM `segmented_reduce`](<https://rocm.docs.amd.com/projects/rocPRIM/en/latest/device_ops/reduce.html>)
- `dpl::reduce`
- `dpl::reduce_by_segment`
- [`std::reduce`](<https://eel.is/c++draft/reduce>)

**Strategy**: Use the common type from all involved types as the accumulator type.
Document this choice as it may affect performance and accuracy.

## Return Value

If the input span is provided,
should the function also return the last processed input iterator?

**Tentative Strategy**. Do not return it to be aligned with the existing overload
and iterator-based algorithms in general.

## Empty Segments

**Strategy**. Return the initial value as in
`cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce`.

## Negative Offsets

**Strategy**. Explicitly prohibit them,
contrary to the `cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce` for safety.

In the proposed interfaces,
the `first_value` and `last_value` iterators will define the valid range to access.

## Proposal

### Synopsis

```c++
// Included from <oneapi/dpl/numeric>
// Can be included from <oneapi/dpl/algorithm> as the existing overload, but it is discouraged.

// (1) Reduce using variable length segments, by their offsets
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,     // to be constrained to an integral type
          typename OffsetIt,
          typename OutputValueIt,
          typename BinaryOp,
          typename InitT>
OutputValueIt
oneapi::dpl::reduce_by_segment(
    Policy&&      policy,
    InputValueIt  first_value,
    InputValueIt  last_value,
    SegmentNumT   num_segments,
    OffsetIt      begin_offsets,
    OffsetIt      end_offsets,
    OutputValueIt result_value,
    BinaryOp      binary_op,
    InitT  init_value
);

// (2) Reduce using fixed-length segments
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,      // to be constrained to an integral type
          typename SegmentLengthT,   // to be constrained to an integral type
          typename OutputValueIt,
          typename BinaryOp,
          typename InitT>
OutputValueIt
oneapi::dpl::reduce_by_segment(
    Policy&&        policy,
    InputValueIt    first_value,
    InputValueIt    last_value,
    SegmentNumT     num_segments,
    SegmentLengthT  segment_length,
    OutputValueIt   result_value,
    BinaryOp        binary_op,
    InitT           init_value
);
```

```c++
// (1) Minimum element indices using variable-length segments
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,     // to be constrained to an integral type
          typename OffsetIt,
          typename OutputIndexIt>
OutputIndexIt
oneapi::dpl::min_element_by_segment(
    Policy&&      policy,
    InputValueIt  first_value,
    InputValueIt  last_value,
    SegmentNumT   num_segments,
    OffsetIt      begin_offsets,
    OffsetIt      end_offsets,
    OutputIndexIt result_index
);

// (2) Minimum element indices using variable-length segments, with a comparator
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,     // to be constrained to an integral type
          typename OffsetIt,
          typename OutputIndexIt,
          typename Compare,
          typename InitT>
OutputIndexIt
oneapi::dpl::min_element_by_segment(
    Policy&&      policy,
    InputValueIt  first_value,
    InputValueIt  last_value,
    SegmentNumT   num_segments,
    OffsetIt      begin_offsets,
    OffsetIt      end_offsets,
    OutputIndexIt result_index,
    Compare       comp,
    InitT         init_value
);

// (3) Minimum element indices using fixed-length segments, with defaults
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,      // to be constrained to an integral type
          typename SegmentLengthT,   // to be constrained to an integral type
          typename OutputIndexIt>
OutputIndexIt
oneapi::dpl::min_element_by_segment(
    Policy&&       policy,
    InputValueIt   first_value,
    InputValueIt   last_value,
    SegmentNumT    num_segments,
    SegmentLengthT segment_length,
    OutputIndexIt  result_index
);

// (4) Minimum element indices using fixed-length segments, with a comparator
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,      // to be constrained to an integral type
          typename SegmentLengthT,   // to be constrained to an integral type
          typename OutputIndexIt,
          typename Compare,
          typename InitT>
OutputIndexIt
oneapi::dpl::min_element_by_segment(
    Policy&&       policy,
    InputValueIt   first_value,
    InputValueIt   last_value,
    SegmentNumT    num_segments,
    SegmentLengthT segment_length,
    OutputIndexIt  result_index,
    Compare        comp,
    InitT          init_value
);
```

```c++
// (4) Maximum element indices using variable-length segments
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,     // to be constrained to an integral type
          typename OffsetIt,
          typename OutputIndexIt>
OutputIndexIt
oneapi::dpl::max_element_by_segment(
    Policy&&      policy,
    InputValueIt  first_value,
    InputValueIt  last_value,
    SegmentNumT   num_segments,
    OffsetIt      begin_offsets,
    OffsetIt      end_offsets,
    OutputIndexIt result_index
);

// (5) Maximum element indices using variable-length segments, with a comparator
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,     // to be constrained to an integral type
          typename OffsetIt,
          typename OutputIndexIt,
          typename Compare,
          typename InitT>
OutputIndexIt
oneapi::dpl::max_element_by_segment(
    Policy&&      policy,
    InputValueIt  first_value,
    InputValueIt  last_value,
    SegmentNumT   num_segments,
    OffsetIt      begin_offsets,
    OffsetIt      end_offsets,
    OutputIndexIt result_index,
    Compare       comp,
    InitT         init_value
);

// (6) Maximum element indices using fixed-length segments, with defaults
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,      // to be constrained to an integral type
          typename SegmentLengthT,   // to be constrained to an integral type
          typename OutputIndexIt>
OutputIndexIt
oneapi::dpl::max_element_by_segment(
    Policy&&       policy,
    InputValueIt   first_value,
    InputValueIt   last_value,
    SegmentNumT    num_segments,
    SegmentLengthT segment_length,
    OutputIndexIt  result_index
);

// (7) Maximum element indices using fixed-length segments, with a comparator
template <typename Policy,
          typename InputValueIt,
          typename SegmentNumT,      // to be constrained to an integral type
          typename SegmentLengthT,   // to be constrained to an integral type
          typename OutputIndexIt,
          typename Compare,
          typename InitT>
OutputIndexIt
oneapi::dpl::max_element_by_segment(
    Policy&&       policy,
    InputValueIt   first_value,
    InputValueIt   last_value,
    SegmentNumT    num_segments,
    SegmentLengthT segment_length,
    OutputIndexIt  result_index,
    Compare        comp,
    InitT          init_value
);
```

### Evolution

The RFC can be implemented in stages, for example:

1. Implement device-only policy support with variable-segment overloads.
2. Enable other policies with the variable-segment overloads.
3. Implement fixed-segment overloads.
4. Add support for `min_element_by_segment` and `max_element_by_segment`.
5. Enable identities for the custom binary operations and types.

### Feature Macro

The API is set to be evolving, hence a feature macro should be defined for convenience.
For example: `ONEDPL_HAS_REDUCE_BY_SEGMENT 202109L`
for the current state as it already has the key-based segmented reduction,
`ONEDPL_HAS_REDUCE_BY_SEGMENT 202611L` with the next version,
e.g. including device policy support and variable-segment overloads.
`ONEDPL_HAS_REDUCE_BY_SEGMENT YYYYMML` for future versions.

### Examples

The following complete programs illustrate the proposed overloads.

`device` policy, variable-length segments, `sum` operation:

```c++
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <functional>
#include <iostream>

int main()
{
    auto policy = oneapi::dpl::execution::dpcpp_default;
    auto queue = policy.queue();

    constexpr int num_elements = 5;
    constexpr int num_segments = 3;
    const int input[] = {10, 20, 5, 7, 4};
    const int host_begin_offsets[] = {0, 2, 4};
    const int host_end_offsets[] = {2, 4, 5};

    int* values = sycl::malloc_shared<int>(num_elements, queue);
    int* begin_offsets = sycl::malloc_shared<int>(num_segments, queue);
    int* end_offsets = sycl::malloc_shared<int>(num_segments, queue);
    int* result = sycl::malloc_shared<int>(num_segments, queue);

    std::copy_n(input, num_elements, values);
    std::copy_n(host_begin_offsets, num_segments, begin_offsets);
    std::copy_n(host_end_offsets, num_segments, end_offsets);

    int* result_end = oneapi::dpl::reduce_by_segment(
        policy, values, values + num_elements, num_segments,
        begin_offsets, end_offsets, result, std::plus<int>{}, 0);

    for (int* it = result; it != result_end; ++it)
        std::cout << *it << ' ';
    std::cout << '\n'; // 30 12 4

    sycl::free(values, queue);
    sycl::free(begin_offsets, queue);
    sycl::free(end_offsets, queue);
    sycl::free(result, queue);
}
```

`par_unseq` policy, fixed-length segments, `maximum` operation:

```c++
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

int main()
{
    constexpr int num_segments = 3;
    constexpr std::int64_t segment_length = 2;
    std::array<int, 6> values = {-10, -20, -5, -7, -4, -8};
    std::array<int, num_segments> result{};
    auto maximum = [](int lhs, int rhs) { return std::max(lhs, rhs); };

    auto result_end = oneapi::dpl::reduce_by_segment(
        oneapi::dpl::execution::par_unseq, values.begin(), values.end(),
        num_segments, segment_length, result.begin(),
        maximum, std::numeric_limits<int>::lowest());

    for (auto it = result.begin(); it != result_end; ++it)
        std::cout << *it << ' ';
    std::cout << '\n'; // -10 -5 -4
}
```

`dpcpp_default` policy, variable-length segments, maximum element indices:

```c++
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>

int main()
{
    auto policy = oneapi::dpl::execution::dpcpp_default;
    auto queue = policy.queue();

    constexpr int num_elements = 5;
    constexpr int num_segments = 3;
    int input[] = {20, 10, 5, 7, 4};
    int host_begin_offsets[] = {0, 2, 4};
    int host_end_offsets[] = {2, 4, 5};

    int* values = sycl::malloc_shared<int>(num_elements, queue);
    int* begin_offsets = sycl::malloc_shared<int>(num_segments, queue);
    int* end_offsets = sycl::malloc_shared<int>(num_segments, queue);
    int* result_index = sycl::malloc_shared<int>(num_segments, queue);

    std::copy_n(input, num_elements, values);
    std::copy_n(host_begin_offsets, num_segments, begin_offsets);
    std::copy_n(host_end_offsets, num_segments, end_offsets);

    int* result_end = oneapi::dpl::max_element_by_segment(
        policy, values, values + num_elements, num_segments,
        begin_offsets, end_offsets, result_index);

    for (int* it = result_index; it != result_end; ++it)
        std::cout << *it << ' ';
    std::cout << '\n'; // 0 1 0

    sycl::free(values, queue);
    sycl::free(begin_offsets, queue);
    sycl::free(end_offsets, queue);
    sycl::free(result_index, queue);
}
```

## Testing

Use `std::reduce` with a sequential execution policy
as a reference implementation with a loop over segments,
and additionally test this variant against predefined results for small inputs.

The rest should follow the established testing practices for oneDPL algorithms,
and have comprehensive coverage.

## Appendix A: oneDPL/DPCT algorithms

### `dpl::reduce_by_segment`

It defines segments as consecutive runs of equivalent keys.

```c++
template <typename Policy,
          typename InputKeyIt, typename InputValueIt,
          typename OutputKeyIt, typename OutputValueIt>
std::pair<OutputKeyIt, OutputValueIt> // example return: {out_key_first + 3,
dpl::reduce_by_segment(               //                  out_val_first + 3}
    Policy&&      policy,             // host and device policies
    InputKeyIt    key_first,
    InputKeyIt    key_last,           // example input:  {1, 1, 2, 2, 3}
    InputValueIt  val_first,          // example input:  {10, 20, 5, 7, 4}
    OutputKeyIt   out_key_first,      // example output: {1, 2, 3}
    OutputValueIt out_val_first       // example output: {30, 12, 4}
);

template <typename Policy,
          typename InputKeyIt, typename InputValueIt,
          typename OutputKeyIt, typename OutputValueIt,
          typename BinaryPred>
std::pair<OutputKeyIt, OutputValueIt>
dpl::reduce_by_segment(
    Policy&&      policy,
    InputKeyIt    key_first,
    InputKeyIt    key_last,
    InputValueIt  val_first,
    OutputKeyIt   out_key_first,
    OutputValueIt out_val_first,
    BinaryPred    binary_pred
);

template <typename Policy,
          typename InputKeyIt, typename InputValueIt,
          typename OutputKeyIt, typename OutputValueIt,
          typename BinaryPred, typename BinaryOp>
std::pair<OutputKeyIt, OutputValueIt>
dpl::reduce_by_segment(
    Policy&&      policy,
    InputKeyIt    key_first,
    InputKeyIt    key_last,
    InputValueIt  val_first,
    OutputKeyIt   out_key_first,
    OutputValueIt out_val_first,
    BinaryPred    binary_pred,
    BinaryOp      binary_op
);
```

API notes:
- When omitted, the predicate is `std::equal_to<Key>{}` and the operation is `std::plus<Value>{}`,
  where `Key` and `Value` are the value types of `InputKeyIt` and `InputValueIt`, respectively.
- The binary operation must be associative; commutativity is not required.

Implementation (GPU):

| Stage | Tiling and workgroup assignment | Reads | Writes |
|---|---|---|---|
| **Reduce** | Input is split into tiles, each assigned to a workgroup. A tile may cover part of a segment, a whole segment, or multiple segments. | Original keys and values. | Partial reductions per subgroup and counts of new segment starts per workgroup (segment-boundary counts). |
| **Scan** | Same tiles and workgroup assignment as the Reduce stage. | Original keys and values again, partial reductions, and segment-boundary counts. | Final reduced values and compacted output keys. |

Emulating offset-based reduction with keys is possible but neither straightforward nor efficient.

### `dpl::experimental::ranges::reduce_by_segment`

It provides the same key-based segmentation functionality as `dpl::reduce_by_segment`,
but takes ranges as input and output rather than iterators.
It supports only device execution policies.

### `dpct::device::segmented_reduce`

```c++
template <int GROUP_SIZE,
          typename ValueT, typename OffsetT,
          class BinaryOp = std::plus<>>
void
dpct::device::segmented_reduce(
    sycl::queue queue,
    ValueT*     inputs,        // example input:  {10, 20, 5, 7, 4}
    ValueT*     outputs,       // example output: {30, 12, 4}
    size_t      segment_count, // example:        3
    OffsetT*    begin_offsets, // example input:  {0, 2, 4}
    OffsetT*    end_offsets,   // example input:  {2, 4, 5}
    BinaryOp    binary_op,
    ValueT      init
);
```

API notes:
- There are no default function arguments: both `binary_op` and `init` must be supplied.
- The operation must be both associative and commutative, due to the use of SYCL group reductions.

Implementation: assign a work-group to a segment, and do the reduction within one kernel launch.
When the value type and the passed predicate is compatible with the `sycl::joint_reduce`, it uses this group algorithm.
Otherwise, it uses `sycl::ext::oneapi::experimental::joint_reduce` group algorithm, hence only oneAPI is supported
in a general case.

## Appendix B: Other Libraries

### `cub::DeviceSegmentedReduce`

[It](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceSegmentedReduce.html)
defines segments via offsets.

```c++
// Variable-length segments
template <typename InputValueIt, typename OutputValueIt,
          typename BeginOffsetIt, typename EndOffsetIt,
          typename BinaryOp, typename InitValueT>
static inline cudaError_t
cub::DeviceSegmentedReduce::Reduce(
    void*              d_temp_storage,
    size_t&            temp_storage_bytes,
    InputValueIt       d_in,            // example input:  {10, 20, 5, 7, 4}
    OutputValueIt      d_out,           // example output: {30, 12, 4}
    cuda::std::int64_t num_segments,    // example:        3
    BeginOffsetIt      d_begin_offsets, // example input:  {0, 2, 4}
    EndOffsetIt        d_end_offsets,   // example input:  {2, 4, 5}
    BinaryOp           reduction_op,    // example:        cub::Sum{}
    InitValueT         initial_value,   // example:        0
    cudaStream_t       stream = nullptr
);

// Fixed-length segments
template <typename InputValueIt, typename OutputValueIt,
          typename BinaryOp, typename InitValueT>
static inline cudaError_t
cub::DeviceSegmentedReduce::Reduce(
    void*              d_temp_storage,
    size_t&            temp_storage_bytes,
    InputValueIt       d_in,
    OutputValueIt      d_out,
    cuda::std::int64_t num_segments,
    int                segment_size,
    BinaryOp           reduction_op,
    InitValueT         initial_value,
    cudaStream_t       stream = nullptr
);

// Fixed-length segments with an execution environment
template <typename InputValueIt, typename OutputValueIt,
          typename BinaryOp, typename InitValueT,
          typename EnvT = cuda::std::execution::env<>>
static inline cudaError_t
cub::DeviceSegmentedReduce::Reduce(
    InputValueIt       d_in,            // example input:  {10, 20, 5, 7, 4, 8, 6}
    OutputValueIt      d_out,           // example output: {30, 12, 12}
    cuda::std::int64_t num_segments,    // example:        3
    int                segment_size,    // example:        2
    BinaryOp           reduction_op,    // example:        cub::Sum{}
    InitValueT         initial_value,   // example:        0
    const EnvT&        env = {}
);
```

API notes:
- Variable-length segments can be empty. They produce the initial value.
- There can be gaps between the variable-length segments.
- `BinaryOp` requires both associativity and commutativity.
- Negative offsets neither explicitly allowed nor prohibited.
  Implementation allows it, and accesses a subrange before `d_in`.
- The accumulator type is not stated explicitly.
  AI-assisted research shows that it is inferred from `InputValueIt`, `InitValueT`, and `BinaryOp`.
- The overload with the execution environment allows control over floating-point determinism,
  memory allocation, and tuning parameters such as block size and elements per thread.

There are `Sum`, `Min`, `Max`, `ArgMin`, and `ArgMax`
specialized methods that supply their own binary operation and initial value.
`ArgMin` and `ArgMax` return both the index and the value of the first occurrence of the minimum or maximum element within each segment.
All these functions also have 3 overloads as the general `Reduce`.
Below are the examples of some methods.

```c++
// Variable-length segments
template <typename InputValueIt, typename OutputValueIt,
          typename BeginOffsetIt, typename EndOffsetIt>
static inline cudaError_t
cub::DeviceSegmentedReduce::Min(
    void*              d_temp_storage,
    size_t&            temp_storage_bytes,
    InputValueIt       d_in,            // example input:  {10, 20, 5, 7, 4}
    OutputValueIt      d_out,           // example output: {10, 5, 4}
    cuda::std::int64_t num_segments,    // example:        3
    BeginOffsetIt      d_begin_offsets, // example input:  {0, 2, 4}
    EndOffsetIt        d_end_offsets,   // example input:  {2, 4, 5}
    cudaStream_t       stream = nullptr
);

```

```c++
// Variable-length segments
template <typename InputValueIt, typename OutputValueIt,
          typename BeginOffsetIt, typename EndOffsetIt>
static inline cudaError_t
cub::DeviceSegmentedReduce::ArgMin(
    void*              d_temp_storage,
    size_t&            temp_storage_bytes,
    InputValueIt       d_in,            // example input:  {20, 10, 5, 7, 4}
    OutputValueIt      d_out,           // example output: {{1, 10}, {0, 5}, {0, 4}}
    cuda::std::int64_t num_segments,    // example:        3
    BeginOffsetIt      d_begin_offsets, // example input:  {0, 2, 4}
    EndOffsetIt        d_end_offsets,   // example input:  {2, 4, 5}
    cudaStream_t       stream = nullptr
);
```

The documentation states that the initial value is obtained as shown in the table below.
It implies that the custom types are supported, e.g. via specializing these mechanisms.

| Overload | Initial value |
|---|---|
| `Sum` | `OutputT{}` |
| `Min` | `cuda::std::numeric_limits<T>::max()` |
| `Max` | `cuda::std::numeric_limits<T>::lowest()` |
| `ArgMin` | `{1, cuda::std::numeric_limits<T>::max()}` |
| `ArgMax` | `{1, cuda::std::numeric_limits<T>::lowest()}` |

The documentation says nothing about the maximum value for the number of the input elements,
segment count and their maximum size, however there are implications based on the interfaces:

| Quantity | Declared type | Meaning |
|---|---|---|
| Number of segments | `cuda::std::int64_t num_segments` | The number of segments can be up to 2^63 - 1 |
| Variable-length segment offsets | Templated begin/end iterators | Their common type can represent the maximum number of input elements, but it is not guaranteed |
| Fixed segment length | `int segment_size` | The maximum segment size is 2^31 - 1
| Variable-length argmin/argmax result index | `int`, in `cub::KeyValuePair<int, T>` | The maximum index and the number of input elements is 2^31 - 1 |
| Fixed-length argmin/argmax result index | `int`, in `cuda::std::pair<int, T>` | The maximum index and the number of input elements is 2^31 - 1 |

Implementation:

- Variable-length segments.
  Assign a work-group to a segment.
  Do batched reduction if `num_segments` exceeds `INT_MAX`.
  A single kernel launch is used unless batched.
- Fixed-length segments.
  Split the input into fixed-size tiles which can include a part of a segment, the whole segment or multiple segments,
  and assign the tile to a single work-group.
  Do batched reduction when `num_segments * tiles_per_segment > INT_MAX`
  or `num_segments > INT_MAX` if a segment fits in a work-group.
  One kernel launch is used below the tile-size threshold; two otherwise, per batch.

### `rocprim::segmented_reduce`

[rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/device_ops/reduce.html#segmented-reduce)
provides reduction for segments defined by begin and end offsets.

```c++
template <typename Config = rocprim::default_config,
          typename InputValueIt, typename OutputValueIt,
          typename OffsetIt,
          typename BinaryOp = rocprim::plus<typename std::iterator_traits<InputValueIt>::value_type>,
          typename InitValueT = typename std::iterator_traits<InputValueIt>::value_type>
inline hipError_t
rocprim::segmented_reduce(
    void*          temporary_storage,
    size_t&        storage_size,
    InputValueIt   input,              // example input:  {10, 20, 5, 7, 4}
    OutputValueIt  output,             // example output: {30, 12, 4}
    unsigned int   segments,           // example:        3
    OffsetIt       begin_offsets,      // example input:  {0, 2, 4}
    OffsetIt       end_offsets,        // example input:  {2, 4, 5}
    BinaryOp       reduce_op = BinaryOp(),
    InitValueT     initial_value = InitValueT(),
    hipStream_t    stream = 0,
    bool           debug_synchronous = false
);
```

API specifics:

- Begin and end offsets use the same iterator type.
- Both the binary operation and initial value have defaults.
  The default operation is addition; the default initial value is value-initialized.
- Negative offsets are neither explicitly allowed nor prohibited.
  Implementation allows it, and accesses a sub-range before `input`.
- The binary operation must be associative and commutative (in practice, it is a documentation gap).
- `Config` allows customization of the implementation's tuning parameters.

Implementation: assign one work-group to each segment.
Batch when `num_segments > UINT_MAX / work_group_size`.
A single kernel launch unless batched.

`hipcub::DeviceSegmentedReduce` uses `rocprim::segmented_reduce` as a backend for all its flavours.
It supports only variable-length segments.
