# Offset-Based Segmented Reduction

## Introduction

RFC proposes adding offset-based segmented reduction to oneDPL.

Motivation:

- Migration from `cub::DeviceSegmentedReduce`.
  SYCLomatic already has `dpct::device::segmented_reduce`,
  but it is a helper rather than a fully-supported library API.
- [Real-world use cases](usage_pattern_study.md), including [#2829](https://github.com/oneapi-src/oneDPL/issues/2829) request with its own motivation.

Table of contents:

- [Synopsis](#synopsis)
- [Design Considerations](#design-considerations)
  - [Kernel Template or Core Algorithm Extension](#kernel-template-or-core-algorithm-extension)
  - [Empty, Overlapping, and Unordered Segments](#empty-overlapping-and-unordered-segments)
  - [Initial Value, its Default Value, and Identity](#initial-value-its-default-value-and-identity)
  - [Bounding Input](#bounding-input)
  - [Fixed-Length Segments](#fixed-length-segments)
  - [Binary Operator](#binary-operator)
  - [Floating Point Determinism](#floating-point-determinism)
  - [Binary Operation and Init Value Order](#binary-operation-and-init-value-order)
  - [Arguments and Overload Disambiguation](#arguments-and-overload-disambiguation)
  - [Sum, Min, Max semantics](#sum-min-max-semantics)
  - [ArgMin and ArgMax semantics](#argmin-and-argmax-semantics)
  - [Accumulator Type](#accumulator-type)
  - [Return Value](#return-value)
  - [Negative Offsets](#negative-offsets)
- [Evolution](#evolution)
- [Feature Macro](#feature-macro)
- [Examples](#examples)
- [Testing](#testing)
- [Open Questions](#open-questions)

See [reference implementations](reference_implementations.md) and the
[usage pattern study](usage_pattern_study.md) for background.


## Synopsis

All the proposed segmented algorithms are included from `<oneapi/dpl/numeric>`.

Optionally, the algorithms can also be made available from `<oneapi/dpl/algorithm>`,
because the existing `reduce_by_segment` allows it due to historical reasons.

```c++
// (1) Variable-length segments
template <typename Policy,
          typename InputValueIt,
          typename OutputValueIt,
          typename SegmentNumT,     // to be constrained to an integral type
          typename OffsetIt,       // to be constrained to a non-integral type
          typename InitT,
          typename BinaryOp>
InputValueIt
oneapi::dpl::reduce_by_segment(
    Policy&&      policy,
    InputValueIt  first_value,
    InputValueIt  last_value,
    OutputValueIt result_value,
    SegmentNumT   num_segments,
    OffsetIt      offsets,
    InitT         init_value,
    BinaryOp      binary_op
);

// (2) Fixed-length segments
template <typename Policy,
          typename InputValueIt,
          typename OutputValueIt,
          typename SegmentNumT,      // to be constrained to an integral type
          typename SegmentLengthT,   // to be constrained to an integral type
          typename InitT,
          typename BinaryOp>
void
oneapi::dpl::reduce_by_segment(
    Policy&&        policy,
    InputValueIt    first_value,
    InputValueIt    last_value,
    OutputValueIt   result_value,
    SegmentNumT     num_segments,
    SegmentLengthT  segment_length,
    InitT           init_value,
    BinaryOp        binary_op
);
```

See [Arguments and Overload Disambiguation](#arguments-and-overload-disambiguation)
for details on the constraints.
Note, the key-based overloads must constrain the `OutputKeyIt` type to be non-integral.

## Design Considerations

### Kernel Template or Core Algorithm Extension

Should the algorithm be implemented as a core extension algorithm, a kernel template, or both?

`dpl::reduce_by_segment` is already available as a core extension algorithm.
The new algorithm fits naturally as a sibling to `dpl::reduce_by_segment` with slightly different semantics.

The corresponding kernel template algorithms can be developed later.
The implementation experience can guide the design of the tuning parameters for the kernel templates.
Passing the temporary storage is an obvious gap,
but the corresponding interface in kernel templates has not yet been defined.

**Strategy**: implement it as a core extension algorithm first.

### Empty, Overlapping, and Unordered Segments

`cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce`
support overlapping and unordered segments,
which is possible because of providing two ranges with segment boundaries (`begin_offsets` and `end_offsets`).
However, according to the [usage study](usage_pattern_study.md),
overwhelming majority of the cases involve one consecutive sequence with offsets.
Consecutive segments are also the target use-case in [#2829](https://github.com/oneapi-src/oneDPL/issues/2829).
Therefore, it is suggested to provide the offsets as a
single sequence to support consecutive segments directly.
Adding support for overlapping or unordered segments can be considered in the future if there is demand.

Empty segment is supported by `cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce`.
[#2829](https://github.com/oneapi-src/oneDPL/issues/2829) also requests support for empty segments.

**Strategy**:

- Support empty segments,
- Provide only one sequence of offsets (no overlapping or unordered segments).

### Initial Value, its Default Value, and Identity

The [#2829](https://github.com/oneapi-src/oneDPL/issues/2829) request
specified an identity as the last argument,
while the industry practice it is typically if not always an initial value.

What should be provided as an argument in place of the initial value? What should be the default value?

Initial value applies only once, and known identity can be applied multiple times.
The initial value may be equal to the known identity or not. A safe default initial value is the identity.

Initial value is always needed as the starting point for the accumulator
when empty input is possible.
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

1. Handle the minimum between `last - first` and segment offsets.
  In this case, we will need to return stop positions in the input, output and offset arrays.
2. Always handle all segments defined by offsets,
  require the input range to cover all offsets.

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
but it is not as efficient due to the segment boundary checks.

**Strategy**: provide the fixed-length segment overloads.

### Binary Operator

Associativity is essential for parallelization.
Commutativity is useful, for example, when doing sub-group and work-group striding,
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

### Binary Operation and Init Value Order

In `cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce`,
the order is `binary_op, init_value`.
In `dpl::reduce`, it is the opposite.
For API consistency, `init_value, binary_op` is preferred.
If full order compatibility with CUB and rocPRIM equivalents is required,
wrappers can be added to, for example, `compat` namespace.

**Strategy**: Use `init_value, binary_op` order.

### Arguments and Overload Disambiguation

The algorithm can be positioned as a sibling of `dpl::reduce_by_segment`,
which means that it should have the same naming if possible.
It should be kept in mind that the proposed overloads
may introduce default values for the binary operation and the initial value.
Overload resolution should be carefully considered to avoid ambiguity.

Providing an alternative functionality
to `cub::DeviceSegmentedReduce` is the main motivation,
therefore, the new overloads should have compatible interfaces
and as close in argument order as possible.

The table below shows the suggested argument order.
It follows the argument order from `cub::DeviceSegmentedReduce`,
adds an input boundary ([Bounding Input](#bounding-input)),
replaces the separate begin and end offset ranges with a single sequence
([Empty, Overlapping, and Unordered Segments](#empty-overlapping-and-unordered-segments)),
and reorders the last two arguments
([Binary Operation and Init Value Order](#binary-operation-and-init-value-order)).
Lines are listed for comparison between oneDPL algorithsm,
and CUB algorithms are shown for the reference on the arguments used.

| CUB `Reduce` (variable segments) | CUB `Reduce` (fixed segments) | Line | `dpl::reduce_by_segment` (key-based segments) | `dpl::reduce_by_segment` (variable segments) | `dpl::reduce_by_segment` (fixed segments) |
|---|---|---|---|---|---|
| `d_temp_storage` | `d_temp_storage` | \- | — | — | — |
| `temp_storage_bytes` | `temp_storage_bytes` | 1 | `policy` | `policy` | `policy` |
| `d_in` | `d_in` | 2 | `key_first` | `first` | `first` |
| `d_out` | `d_out` | 3 | `key_last` | `last` | `last` |
| `num_segments` | `num_segments` | 4 | `val_first` | `out_first` | `out_first` |
| `d_begin_offsets` | `segment_size` | 5 | `out_key_first` | `num_segments` | `num_segments` |
| `d_end_offsets` | `reduction_op` | 6 | `out_val_first` | `offsets` | `segment_length` |
| `reduction_op` | `initial_value` | 7 | `binary_pred` (opt) | `init` (opt) | `init` (opt) |
| `initial_value` | `stream` (opt) | 8 | `binary_op` (opt) | `binary_op` (opt) | `binary_op` (opt) |
| `stream` (opt) | — | 9 | — | — | — |

`(opt)` means that the argument is optional or can be made optional in the future.
`—` is padding for better mapping of the relevant arguments.

The overloads overlap in the number of arguments (6 to 8),
and only the 5th and 6th arguments differ in kind (iterator vs integer),
which can be constrained using SFINAE:

| Overload | 5th argument | 6th argument |
|---|---|---|
| key-based (existing) | `out_key_first`: not integral | — |
| variable-length | `num_segments`: integral | `offsets`: not integral |
| fixed-length | `num_segments`: integral | `segment_length`: integral |

The existing key-based overloads must get a new constraint: `!std::is_integral_v<OutputKeyIt>`,
It does not break existing code, because an integral type does not satisfy the iterator requirements.
This SFINAE constraint can be folded into the existing policy constraint, for example:
```c++
__internal::__enable_if<oneapi::dpl::is_execution_policy_v<Policy>, !std::is_integral_v<OutputKeyIt>, /*return-type*/>
```

instead of:

```c++
__internal::__enable_if_execution_policy<Policy, /*return-type*/>
```

The same approach can be applied to constrain the new overloads.

**Strategy**:

- Constrain `num_segments` and `segment_length` to integral types,
  and `offsets` to non-integral types.
- Constrain the existing key-based overloads to reject an integral `out_key_first`.

### Sum, Min, Max semantics

All semantics are used according to the [real-world use cases](usage_pattern_study.md).
Therefore, they should be supported.

`Sum`, `Min`, and `Max` semantics can be enabled by
specifying the appropriate binary operation and initial value for the reduction.
Therefore, there is no need to introduce special overloads
for the `reduce_by_segment` algorithm, complicating the API.

**Strategy**: Do nothing special for `Sum`, `Min`, and `Max` semantics.

### ArgMin and ArgMax semantics

This semantics will be explored in a separate RFC.

### Accumulator Type

The accumulator type determines the representation of intermediate reduction results.

This type can be influenced by:

- initial value type
- input value type
- output value type
- result type of the binary operation

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

### Return Value

`dpl::reduce_by_segment` returns a pair of iterators: `{output_keys_end, output_values_end}`
`cub::DeviceSegmentedReduce::Reduce` and `rocprim::segmented_reduce` return runtime status codes.

Obviously, returning the status codes is not an option
due to the inconsistency with the existing oneDPL algorithms.

In the key-based `dpl::reduce_by_segment`,
`output_keys_end` is useful to determine the number of segments, and
`output_values_end` is useful to determine the number of reduced values,
as this information is not directly passed to the algorithm.
In the new overloads, this either not relevant (no keys) or known in advance:
`out_last = out_first + num_segments;`

In the new overloads, the usefulness of the returned value depends on the segment type.
In variable-length segments, an iterator past the last processed may differ from `last`
because the number of elements to process is determined by segment offsets,
and is not known in advance.
In the fixed-segment case, the last output iterator is known: `out_first + num_segments`.

**Tentative Strategy**. Return an iterator past the last processed element for variable-length segments,
and return nothing for the fixed-segment overloads.

### Negative Offsets

`cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce` allow negative segments.
In this case, the algorithms may access the memory regions before the beginning of the input range.
It is dangerous and should be prohibited.

Handling options:

1. Allow passing only unsigned offsets.
2. Document the requirement, which results in undefined behavior if violated, and do nothing at the implementation level.
3. Stop the algorithm if a negative offset is detected.
4. Skip accessing memory before the beginning of the input range or whole segments with negative offsets.

(1) may be not compatible with the existing code bases.
(2) is dangerous, but acceptable. This approach is commonly used in the standard library.
For example, `std::inplace_merge` requires data to be sorted,
and `middle` must be a valid iterator within the range, otherwise, the behavior is undefined.
(3) and (4) can be treated as actions to take for the undefined behavior scenarios
to avoid segmentation faults when contract is violated, which is optional.

**Strategy**: Document the requirement to avoid negative offsets.

## Evolution

The RFC can be implemented in stages, for example:

1. Implement device-only policy support with variable-segment overloads of `reduce_by_segment`
2. Enable other policies with the variable-segment overloads of `reduce_by_segment`.
3. Implement fixed-segment overloads of `reduce_by_segment`.
4. Enable identities for the custom binary operations and types.

## Feature Macro

The API is set to be evolving, hence a feature macro should be defined for convenience.
For example:

- `ONEDPL_HAS_REDUCE_BY_SEGMENT 202109L`
  for the current state as it already has the key-based segmented reduction,
- `ONEDPL_HAS_REDUCE_BY_SEGMENT 202611L` with the next version, e.g.,
   including device policy support and variable-segment overloads of `reduce_by_segment`.
- `ONEDPL_HAS_REDUCE_BY_SEGMENT YYYYMML` for future versions.

## Examples

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
    constexpr int num_segments = 4;
    const int input[] = {10, 20, 5, 7, 4};
    const int host_offsets[] = {0, 2, 2, 4, 5};

    int* values = sycl::malloc_shared<int>(num_elements, queue);
    int* offsets = sycl::malloc_shared<int>(num_segments + 1, queue);
    int* result = sycl::malloc_shared<int>(num_segments, queue);

    std::copy_n(input, num_elements, values);
    std::copy_n(host_offsets, num_segments + 1, offsets);

    int* result_end = oneapi::dpl::reduce_by_segment(
        policy, values, values + num_elements, result,
        num_segments, offsets, 0, std::plus<int>{});

    for (int* it = result; it != result_end; ++it)
        std::cout << *it << ' ';
    std::cout << '\n'; // 30 0 12 4

    sycl::free(values, queue);
    sycl::free(offsets, queue);
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
        result.begin(), num_segments, segment_length,
        std::numeric_limits<int>::lowest(), maximum);

    for (auto it = result.begin(); it != result_end; ++it)
        std::cout << *it << ' ';
    std::cout << '\n'; // -10 -5 -4
}
```

## Testing

Use `std::reduce` with a sequential execution policy
as a reference implementation with a loop over segments,
and additionally test this variant against predefined results for small inputs.

The rest should follow the established testing practices for oneDPL algorithms,
and have comprehensive coverage.

## Open Questions
