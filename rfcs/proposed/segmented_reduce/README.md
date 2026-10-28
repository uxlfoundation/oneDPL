# Offset-Based Segmented Reduction

## Introduction

This RFC proposes adding offset-based segmented reduction to oneDPL.

Motivation:

- Migration from `cub::DeviceSegmentedReduce`.
  SYCLomatic already has `dpct::device::segmented_reduce`,
  but it is a helper rather than a fully-supported library API.
- [Real-world use cases](usage_pattern_study.md), including [#2829](https://github.com/oneapi-src/oneDPL/issues/2829) request with its own motivation.

Table of contents:

- [Synopsis](#synopsis)
- [Design Considerations](#design-considerations)
  - [Kernel Template or Core Algorithm Extension](#kernel-template-or-core-algorithm-extension)
  - [Empty, Overlapping, Negatively Sized, and Unordered Segments](#empty-overlapping-negatively-sized-and-unordered-segments)
  - [`num_segments` and `segment_length`: Types and Negative Values](#num_segments-and-segment_length-types-and-negative-values)
  - [Binary Operation and Init Value Order](#binary-operation-and-init-value-order)
  - [Initial Value or Identity](#initial-value-or-identity)
  - [Default Initial Value and Binary Operation](#default-initial-value-and-binary-operation)
  - [Identity](#identity)
  - [Bounding Input and Negative Offsets](#bounding-input-and-negative-offsets)
  - [Fixed-Length Segments](#fixed-length-segments)
  - [Binary Operation Requirements](#binary-operation-requirements)
  - [Floating Point Determinism](#floating-point-determinism)
  - [Arguments and Overload Disambiguation](#arguments-and-overload-disambiguation)
  - [Sum, Min, Max semantics](#sum-min-max-semantics)
  - [ArgMin and ArgMax semantics](#argmin-and-argmax-semantics)
  - [Accumulator Type](#accumulator-type)
  - [Return Value](#return-value)
  - [Feature Macro](#feature-macro)
- [Next Steps](#next-steps)
- [Examples](#examples)
- [Testing](#testing)
- [Open Questions](#open-questions)
  - [Different `first` and `last` Types](#different-first-and-last-types)
  - [Iterator Requirements](#iterator-requirements)
- [Links](#links)
  - [C++ Standard References](#c-standard-references)
  - [Performance Studies](#performance-studies)

See [reference implementations](reference_implementations.md) and the
[usage pattern study](usage_pattern_study.md) for background.

## Synopsis

The new overloads must be added to the same namespace and headers
as the existing `reduce_by_segment` overloads, as defined in the
[specification](../../../documentation/specification/parallel_api/algorithms.rst).

```c++
// (1) Variable-length segments,
//     std::plus<> as the default binary operation
template <typename Policy,
          typename InputIt,
          typename OutputIt,
          typename SegmentNumT, // to be constrained to an integral type
          typename OffsetIt,    // to be constrained to a non-integral type
          typename InitT>
OutputIt
oneapi::dpl::reduce_by_segment(
    Policy&&      policy,
    InputIt       first,
    InputIt       last,
    OutputIt      out_first,
    SegmentNumT   num_segments,
    OffsetIt      offsets,
    InitT         init_value
);

// (2) Variable-length segments,
//     a custom binary operation
template <typename Policy,
          typename InputIt,
          typename OutputIt,
          typename SegmentNumT, // to be constrained to an integral type
          typename OffsetIt,    // to be constrained to a non-integral type
          typename InitT,
          typename BinaryOp>
OutputIt
oneapi::dpl::reduce_by_segment(
    Policy&&      policy,
    InputIt       first,
    InputIt       last,
    OutputIt      out_first,
    SegmentNumT   num_segments,
    OffsetIt      offsets,
    InitT         init_value,
    BinaryOp      binary_op
);

// (3) Fixed-length segments,
//     std::plus<> as the default binary operation
template <typename Policy,
          typename InputIt,
          typename OutputIt,
          typename SegmentNumT,    // to be constrained to an integral type
          typename SegmentLengthT, // to be constrained to an integral type
          typename InitT>
OutputIt
oneapi::dpl::reduce_by_segment(
    Policy&&        policy,
    InputIt         first,
    InputIt         last,
    OutputIt        out_first,
    SegmentNumT     num_segments,
    SegmentLengthT  segment_length,
    InitT           init_value
);

// (4) Fixed-length segments,
//     a custom binary operation
template <typename Policy,
          typename InputIt,
          typename OutputIt,
          typename SegmentNumT,    // to be constrained to an integral type
          typename SegmentLengthT, // to be constrained to an integral type
          typename InitT,
          typename BinaryOp>
OutputIt
oneapi::dpl::reduce_by_segment(
    Policy&&        policy,
    InputIt         first,
    InputIt         last,
    OutputIt        out_first,
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

Note, full API compatibility with `cub::DeviceSegmentedReduce` is a non-goal for core algorithms,
because it is a more high-level abstraction.

**Strategy**: implement it as a core extension algorithm first.

### Empty, Overlapping, Negatively Sized, and Unordered Segments

`cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce`
support overlapping and unordered segments,
which is possible because of providing two ranges with segment boundaries (`begin_offsets` and `end_offsets`).
However, according to the [usage study](usage_pattern_study.md),
the overwhelming majority of the cases involve one consecutive sequence with offsets,
which are expected to be sorted in ascending order with no overlaps nor gaps.
Consecutive segments are also the target use-case in [#2829](https://github.com/oneapi-src/oneDPL/issues/2829).
Therefore, it is suggested to provide the offsets as a
single sequence to support consecutive segments directly, and require that it is sorted in ascending order.
Adding support for overlapping or unordered segments can be considered in the future if there is demand.

Empty segments are supported by `cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce`.
[#2829](https://github.com/oneapi-src/oneDPL/issues/2829) also requests support for empty segments.

Negatively sized segments (`offsets[i] > offsets[i + 1]`)
should not be allowed and treated as undefined behavior.
It is not clear how to handle surrounding segments,
for example, in `{0, 10, 8, 15, ...}`, what are the boundaries of the third segment,
are they `{10, 15}` or `{8, 15}`?
Supporting them would also rule out using binary search.

**Strategy**:

- Support empty segments.
- Treat negatively sized segments as erroneous behavior:
  results are unspecified, but the implementation must prevent out-of-bounds accesses to the input range.
  Decision on how to avoid out-of-bounds access is left to the implementation.
- Provide only one sequence of offsets (no overlapping or unordered segments), and
  require that it is sorted in ascending order.

### `num_segments` and `segment_length`: Types and Negative Values
Both overloads have a `num_segments` argument; the fixed-length overload also has `segment_length`.

These are single-value arguments always available on the host,
making it easier to check their validity.
Unlike negatively sized segments defined via offsets, which may reside on a device,
they can be checked before any work is submitted.

Possible argument types:

| Type | Alignment | Non-positive value |
|---|---|---|
| template: unsigned integer | None | Zero returns immediately. Negative values are not representable; signed arguments, even literals like `3`, do not compile |
| template: any integer | The count argument of the standard and oneDPL `*_n` algorithms | Representable for signed types; non-positive values return immediately |
| `std::size_t` | dpct, rocPRIM (`unsigned int`) | Zero returns immediately. A negative argument silently wraps to a huge value. |
| `std::uint64_t` | None | Same as `std::size_t` |
| `std::int64_t` | CUB (`num_segments`) | Representable; non-positive values return immediately. Unsigned arguments silently convert |

**Strategy**:

- Accept any integral type for `num_segments` and `segment_length` via template parameters.
- If `num_segments <= 0` or `segment_length < 0`,
  short-circuit the algorithm: the input and the offsets are not read, and the output is not written.
- If `segment_length == 0` (and `num_segments > 0`), treat all segments as empty.
- Document this behavior.

### Binary Operation and Init Value Order

In `cub::DeviceSegmentedReduce` and `rocprim::segmented_reduce`,
the order is `binary_op, init_value`.
In `dpl::reduce` and in almost all C++ algorithms, it is the opposite.
For API consistency, `init_value, binary_op` is preferred.
If full order compatibility with CUB and rocPRIM equivalents is required,
wrappers can be added to, for example, `compat` namespace.

**Strategy**: Use `init_value, binary_op` order.

### Initial Value or Identity

The [#2829](https://github.com/oneapi-src/oneDPL/issues/2829) request
specified an identity as the last argument,
while in industry practice it is typically if not always an initial value.

An initial value applies only once, and an identity can be applied multiple times.
The initial value may be equal to the identity or not.

The motivation of [#2829](https://github.com/oneapi-src/oneDPL/issues/2829)
is to initialize empty segments to a value equal to the identity,
which is achieved by passing the identity as the initial value.

In theory, there can be use cases where the initial value can be different from the identity,
for example, as a floor for the max operation or a cap for the min operation.

**Strategy**: have an initial value as an argument.

### Default Initial Value and Binary Operation

What should be the default arguments if any?

Precedents:

- `dpl::reduce_by_segment` provides a default binary operation (`std::plus<T>`, where `T` is the input value type);
  there is no initial value argument.
- `std::reduce` provides a default initial value (`T{}`) and a default binary operation (`std::plus<>`).
- `std::accumulate` provides only a default binary operation (`operator+`, equivalent to `std::plus<>`);
  the initial value must be provided explicitly.
- `std::ranges::fold_left` requires passing both.
- `cub::DeviceSegmentedReduce` requires passing both with the general `Reduce` interface,
  and does not provide these arguments at all with `Sum`, `Min`, `Max`.
- `rocprim::segmented_reduce` provides a default initial value (`T{}`) and a default binary operation (`rocprim::plus<T>`).

The following motivations are considered, in the order of importance:

1. Predictable behavior. Surprises are minimized
   when the default initial value and the default binary operation form an identity for the input type.
2. Consistency with the existing `dpl::reduce_by_segment` overloads, and the closest standard algorithm: `std::reduce`.

Both arguments cannot be omitted simultaneously to satisfy (1).
For example, with `std::plus<>`, `T{}` is an identity for arithmetic types,
but not necessarily for a user-defined type; assuming `T{}`, like in `std::reduce`, may lead to unexpected results.
The initial value argument precedes the binary operation argument,
so the initial value must be specified and the binary operation can be defaulted.
It is consistent with `dpl::reduce_by_segment` (it does not have an initial value argument at all)
and `std::accumulate` (it requires an explicit initial value).

Should the binary operation be `std::plus<>` like in `std::reduce` or `std::plus<T>` like in `dpl::reduce_by_segment`?
Unlike `dpl::reduce_by_segment`, the new overloads have an initial value besides the input range.
Using `std::plus<>` allows the type to be deduced as `AccT` following the rules from [Accumulator Type](#accumulator-type).

**Strategy**:

- The initial value must be explicitly specified by the user.
- The binary operation can have a default, which should be `std::plus<>`.

### Identity

Initial value is mandatory when the algorithm's contract is to apply it to each segment,
or as a result for empty segments, what is done by CUB and rocPRIM.
Identity value is optional. It is needed for the SIMD and the group reductions,
and it would be a safe default initial value, should one be introduced in the future.

The identity may be inferred from the most typical operations and data types, for example:

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

**Strategy**: wait until [p3732r2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3732r2.html) settles
and follow it to provide a way to specify the identity explicitly.

### Bounding Input and Negative Offsets

`cub::DeviceSegmentedReduce` (variable-length segments) and `rocprim::segmented_reduce`
do not provide the end bound for the input range explicitly.
The bound is implicitly determined by the segment offsets and the number of segments.
They also allow negative offsets,
in which case the algorithms access the memory regions before the beginning of the input range.

It has two issues:

1. Safety: there is a risk of accessing out-of-bounds elements,
   both past the end and before the beginning of the input range.
2. Performance: the only information guaranteed to be available on the host
   to distribute the workload is the number of segments.
   It either forces the implementation to assign a segment to a work-group,
   which is not efficient with varying segment sizes,
   or it involves additional checks and kernel launches to understand the actual input data space.

The interfaces should have the end bound to address these issues,
and negative offsets should not be processed, or should be disallowed altogether.
It creates another question about how to
reconcile the input range size provided with the segment offsets,
because both define the reduction range.
See also [Different `first` and `last` Types](#different-first-and-last-types)
for the case when `last` does not bound the input.

The offsets and output ranges are out of scope here,
because their sizes are specified in a much more direct way via `num_segments`.

The table below outlines the options for handling input bounds.
Its notation:

- `N = last - first`.
- Valid offsets: `0 <= offsets[0]` and `offsets[num_segments] <= N`
  for variable-length segments
  (the offsets are consecutive, see [Empty, Overlapping, Negatively Sized, and Unordered Segments](#empty-overlapping-negatively-sized-and-unordered-segments),
  thus only the first and the last offsets define the bounds),
  or positive `num_segments` and non-negative `segment_length` with `num_segments * segment_length <= N`
  for fixed-length segments.

| Option | Precedents | Enforcement | Communication | Performance |
|---|---|---|---|---|
| (1) No preconditions, partial reduction of the segments crossing the bounds | Binary `std::ranges::transform`, `std::ranges::mismatch`, and `std::ranges::equal`, which process the minimum of two bounds and return the stop positions | Per-segment clamping. | Segments completely beyond the input range receive `init_value` as if the segments are empty, because the intersection of the offsets with the input is empty | Variable-length: a few `min` and `max` operations per segment, negligible; Fixed-length: `min(num_segments, N / segment_length)` (or `num_segments` if `segment_length == 0`) is computed on the host, no performance impact |
| (2) No preconditions, skip segments crossing and beyond the bounds | No close precedent | Per-segment check | Same as (1) | Same as (1) |
| (3) Require valid offsets, stop if violated | Checked APIs such as `std::vector::at`, which is uncommon for the standard algorithms. Hardened preconditions ([C++26 hardening](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3471r4.html), libc++ hardening modes) also check the preconditions, but terminate instead of throwing | Variable-length: set a flag in the kernel and check it afterwards (the output is partially written). Fixed-length: a host-side check before the launch | An exception (e.g., `std::out_of_range`) | Variable-length with device policies: a small device-to-host copy and a synchronization; with host policies: negligible. Fixed-length: none |
| (4) Require valid offsets, unspecified results but no out-of-bounds accesses to the input range if violated | [Erroneous behavior](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2795r5.html): the behavior is well-defined but incorrect, and implementations may diagnose it | Clamping as in (1) or (2), whichever is cheaper for the backend, not specified | Silent, the output values are unspecified. Optionally, an assertion in debug mode | Same as (1) and (2), without the cost of returning iterators |
| (5) Require valid offsets, document only | Standard practice for iterator-based algorithms: for example, `std::inplace_merge` requires `middle` to be a valid iterator within the range, otherwise, the behavior is undefined. It also matches the behavior of CUB and rocPRIM | None, except for optional host-side assertions in debug mode | None: out-of-bounds accesses, crashes or garbage results | None, `last` is used only for the workload distribution |

**Strategy**:

- Have a parameter for the `last` input element in both proposed function overloads.
- Follow option (1): no preconditions, partial reduction of the segments crossing the bounds.
  Document the behaviour.

### Fixed-Length Segments

Is this overload necessary given that the variable-length segment overload can handle all cases?

Yes, because this semantics is already used and due to the performance benefits:

- Select the reduction on host before the kernel submission
  (per work-item, sub-group, work-group or multi-work-group with multiple kernels)
- Compute segment boundaries arithmetically, avoiding offset arrays and their memory traffic.
- Derive tile counts and partial-result locations directly,
  avoiding segment identifiers in multi-kernel implementations.

It can be emulated via the variable-length overload, with offsets `i * segment_length`
produced by a transform iterator over a counting iterator.
A dedicated implementation can take advantage of the known segment length and number on the host,
to select the optimal reduction strategy before the kernel submission.

**Strategy**: provide the fixed-length segment overloads.

### Binary Operation Requirements

Associativity is essential for parallelization.
Commutativity lets the implementation reorder the operands, which is used by:

- Sub-group and work-group striding, a common GPU technique to improve memory coalescing.
- Host vectorization, for example `omp simd reduction`.
- SYCL group reductions, for example `sycl::reduce_over_group` and `sycl::joint_reduce`.

Requirements in the existing implementations:

- `cub::DeviceSegmentedReduce`, `rocprim::segmented_reduce`, `dpct::device::segmented_reduce`:
  associativity and commutativity.
- `dpl::reduce_by_segment`: associativity only.
- For reference, `std::reduce`, `std::transform_reduce`, and their oneDPL equivalents:
  associativity and commutativity.

The key-based `dpl::reduce_by_segment` relies on the scan pattern internally
to get the offset boundaries, which preserves the operand order,
and because of that, it was likely decided not to require commutativity.
With offsets or a fixed segment length, the boundaries are known in advance,
and the algorithm is a batch of independent reductions.
Requiring commutativity keeps the optimization techniques above available.

**Strategy**:

- Require associativity and commutativity.
- Document that, unlike the key-based overloads,
  the new overloads do not support non-commutative operations.

### Floating Point Determinism

Floating point operations are not associative,
so the result depends on how the implementation groups the operands.

[P4229R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4229r0.pdf) (section 2),
proposed for the C++ standard, distinguishes these scopes of determinism, from the weakest:

- *Run-to-run*: the same result for the same program, input, implementation, and environment.
- *Same-platform*: also the same result across thread count, chunk size, or vector width on one platform.
- *Expression-based*: the result is defined by a named expression, for example a fixed tree
  ([P4016R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4016r0.pdf) proposes one for `reduce`).
- *Cross-ISA*: also the same result across architectures and implementations,
  which needs control over contraction, extended precision, and rounding
  ([P3375R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3375r3.html)).

[CCCL](https://github.com/NVIDIA/cccl/blob/main/docs/cccl/determinism.rst) defines these levels in
`cuda::execution::determinism`, passed to algorithms through the execution environment:

- *No determinism* (`not_guaranteed`): the result may differ between runs, even on the same GPU.
- *Run-to-run* (`run_to_run`): the same result on the same GPU, with the same input, build, and tuning.
- *GPU-to-GPU* (`gpu_to_gpu`): also the same result across GPU architectures.

`cub::DeviceSegmentedReduce` supports the first two, with run-to-run as the default.

oneDPL documents no floating point guarantees for `dpl::reduce` or `dpl::reduce_by_segment`.
The table shows what the current implementation does.

| Policy | Backend | `dpl::reduce` | `dpl::reduce_by_segment` |
|---|---|---|---|
| `seq` | - | *Expression-based*: left fold. | *Expression-based*: left fold. |
| `unseq` | - | *No determinism* for `std::plus` on arithmetic types: `omp simd reduction` is used. Otherwise, *run-to-run*: fixed 64-byte lanes. | *No determinism* where `omp simd scan` is supported (icpx 2023.1+, not Windows). Otherwise, *expression-based*: left fold. |
| `par` | TBB | *No determinism.* `tbb::parallel_reduce` split points depend on work stealing. | *No determinism.* `tbb::parallel_scan` split points depend on work stealing. |
| `par` | OpenMP | *No determinism.* `omp taskloop` assigns chunks dynamically, making the combination order unpredictable. | *Expression-based*: left fold (the scan is serial). |
| `par` | serial | Same as `seq`. | Same as `seq`. |
| `par_unseq` | TBB | Same as `par`. | Same as `par`. |
| `par_unseq` | OpenMP | Same as `par`. | Same as `unseq`. |
| `par_unseq` | serial | Same as `unseq`. | Same as `unseq`. |
| device | SYCL | *No determinism* for arithmetic types with standard operators, such as `std::plus<float>`: `sycl::reduce_over_group` is used. Otherwise, *run-to-run*: grouping depends on the work-group size, compute units, and input size. | *Run-to-run* (reduce-then-scan): values are combined only by oneDPL code. Grouping depends on sub-group sizes, work-group size, compute units, and cache size. |

Run-to-run determinism is lost in three ways:

- Dynamic partitioning: TBB `parallel_reduce` and `parallel_scan`, OpenMP `taskloop`.
- OpenMP reductions and scans, including `omp simd` and `inscan` ones.
  The OpenMP specification says that "the order in which values are combined [is] unspecified",
  and "bitwise-identical results are not guaranteed" even between two parallel executions
  (OpenMP 5.2 section 5.5.6, OpenMP 6.0 section 7.6.7).
- SYCL group algorithms applied to floating point values.
  SYCL 2020 (section 4.17.3.5) only calls their result non-deterministic for non-associative operators,
  so they are assumed to provide no determinism.

Potential implementation challenges for run-to-run determinism:

- Load balancing with host execution:
  work stealing balances the load, while it is more difficult to implement with a fixed partition.
- Host vectorization: replacing `omp simd reduction`
  with custom-vectorized code is more difficult and may be less efficient.
- Device group operations: replacing `sycl::reduce_over_group`
  is more difficult and may be less efficient.
- Long segments on a device: partial results must be combined across work-groups
  without atomics or decoupled look-back, which limits optimization opportunities.

**Strategy**:

- Do not guarantee determinism, consistently with `dpl::reduce` and `dpl::reduce_by_segment`.
- Reconsider determinism in the future for specific policies
  if there is demand and
  if the implementation experience suggests that it is feasible.

### Arguments and Overload Disambiguation

The algorithm can be positioned as a sibling of `dpl::reduce_by_segment`,
which means that it should have the same naming if possible.
It should be kept in mind that the proposed overloads
have a default value for the binary operation, which makes the number of arguments vary.
Overload resolution should be carefully considered to avoid ambiguity.

The table below shows the suggested argument order:

| CUB `Reduce` (variable segments) | CUB `Reduce` (fixed segments) | Line | `dpl::reduce_by_segment` (key-based segments) | `dpl::reduce_by_segment` (variable segments) | `dpl::reduce_by_segment` (fixed segments) |
|---|---|---|---|---|---|
| `d_temp_storage` | `d_temp_storage` | \- | — | — | — |
| `temp_storage_bytes` | `temp_storage_bytes` | 1 | `policy` | `policy` | `policy` |
| `d_in` | `d_in` | 2 | `key_first` | `first` | `first` |
| `d_out` | `d_out` | 3 | `key_last` | `last` | `last` |
| `num_segments` | `num_segments` | 4 | `val_first` | `out_first` | `out_first` |
| `d_begin_offsets` | `segment_size` | 5 | `out_key_first` | `num_segments` | `num_segments` |
| `d_end_offsets` | `reduction_op` | 6 | `out_val_first` | `offsets` | `segment_length` |
| `reduction_op` | `initial_value` | 7 | `binary_pred` (opt) | `init_value` | `init_value` |
| `initial_value` | `stream` (opt) | 8 | `binary_op` (opt) | `binary_op` (opt) | `binary_op` (opt) |
| `stream` (opt) | — | 9 | — | — | — |

`(opt)` means that the argument is optional or can be made optional in the future.
`—` is padding for better mapping of the relevant arguments.

The key-based overloads take 6 to 8 arguments, and the new overloads take 7 or 8,
so they overlap in the number of arguments (7 and 8),
and only the 5th and 6th arguments differ in kind (iterator vs integer),
which can be constrained using SFINAE:

| Overload | 5th argument | 6th argument |
|---|---|---|
| key-based (existing) | `out_key_first`: not integral | — |
| variable-length | `num_segments`: integral | `offsets`: not integral |
| fixed-length | `num_segments`: integral | `segment_length`: integral |

The existing key-based overloads must get a new constraint: `!std::is_integral_v<OutputKeyIt>`.
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

### Sum, Min, Max Semantics

All semantics are used according to the [real-world use cases](usage_pattern_study.md).
Therefore, they should be supported.

`Sum`, `Min`, and `Max` semantics can be enabled by
specifying the appropriate binary operation and initial value for the reduction.
Therefore, there is no need to introduce special overloads
for the `reduce_by_segment` algorithm, complicating the API.

**Strategy**: Do nothing special for `Sum`, `Min`, and `Max` semantics.

### ArgMin and ArgMax Semantics

These semantics will be explored in a separate RFC due to their complexity.

Note, `ArgMin` and `ArgMax` can already be expressed via the proposed `reduce_by_segment` and fancy iterators:

1. Pack the input into `zip(counting_iterator, first)` to reduce global index-value pairs.
2. Use a binary operation which compares the values and, for equal values, selects the smaller index.
3. Make the output iterator convert the global index into a segment-local one:
   the output position corresponds to the segment, which gives access to the segment's begin offset.
   Preserve the default index for empty segments.

`ArgMin` example:

```c++
using IndexValueIt = decltype(
  dpl::make_zip_iterator(dpl::counting_iterator<OffsetT>(0), std::declval<T*>())
);
using IndexValueT = typename IndexValueIt::value_type;

constexpr OffsetT identity_index = std::numeric_limits<OffsetT>::max();
constexpr T identity_value = std::numeric_limits<T>::has_infinity? std::numeric_limits<T>::infinity():
                                                                   std::numeric_limits<T>::max();

// min, return the first match
struct argmin
{
    template <typename A, typename B>
    auto
    operator()(const A& a, const B& b) const
    {
        auto ia = std::get<0>(a);
        auto ib = std::get<0>(b);
        auto va = std::get<1>(a);
        auto vb = std::get<1>(b);
        // NaN values are not supported in this comparison
        const bool take_b = vb < va || (!(va < vb) && ib < ia);
        return IndexValueT{take_b ? ib : ia, take_b ? vb : va};
    }
};
// helper to convert global index to local index when writing the result to the output
struct global_to_local
{
    OffsetT* out_offset_position;
    OffsetT segment_begin_idx;
    const global_to_local&
    operator=(OffsetT global_offset) const
    {
        bool is_empty = global_offset == identity_index;
        *out_offset_position = is_empty ? identity_index : global_offset - segment_begin_idx;
        return *this;
    }
};

T* values = sycl::malloc_shared<T>(n, q);
OffsetT* out_offsets = sycl::malloc_shared<OffsetT>(num_segments, q);
T* out_values = sycl::malloc_shared<T>(num_segments, q);

// pack the input into index-value pairs, the indices are global for now
IndexValueIt in{dpl::counting_iterator<OffsetT>(0), values};

// use IndexValueIt::value_type to guarantee device compatibility for the initial value
const IndexValueT init{identity_index, identity_value};

// Variable-length segments
{
    OffsetT* offsets = sycl::malloc_shared<OffsetT>(num_segments + 1, q);

    // the result position in the output corresponds to the segment index,
    // which is used to get the beginning of the segment in the input
    // which in turn is used to convert the global index to a local index within the segment.
    auto out_local_offsets = dpl::make_transform_iterator(
        dpl::counting_iterator<OffsetT>(0), // segment index
        [=](OffsetT segment_idx) {
            return global_to_local{
                out_offsets + segment_idx,  // where to write the result to
                offsets[segment_idx]        // where the segment begins relative to the input
            };
        }
    );
    auto out = dpl::make_zip_iterator(out_local_offsets, out_values); // index-value pairs as output
    // auto out = dpl::make_zip_iterator(out_local_offsets, dpl::discard_iterator()); // only index as output
    dpl::reduce_by_segment(policy, in, in + n, out, num_segments, offsets, init, argmin{});
}

// Fixed-length segments
{
    // Almost the same as above, but with fixed segment dimensions instead of the offset sequence.
    auto out_local_offsets = dpl::make_transform_iterator(
        dpl::counting_iterator<OffsetT>(0),
        [=](OffsetT segment_idx) { return global_to_local{out_offsets + segment_idx,
                                                       segment_idx * segment_length}; }
    );
    auto out = dpl::make_zip_iterator(out_local_offsets, out_values); // index-value pairs as output
    // auto out = dpl::make_zip_iterator(out_local_offsets, dpl::discard_iterator()); // only index as output
    dpl::reduce_by_segment(policy, in, in + n, out, num_segments, segment_length, init, argmin{});
}
```

**Strategy**:

- Postpone the exploration and implementation of dedicated interfaces.
- Document how `ArgMin` and `ArgMax` can be implemented via `reduce_by_segment` and fancy iterators.
  The emulation is complex, thus there also should be a note that
  a dedicated interface is expected in the future.

### Accumulator Type

The accumulator type determines the representation of intermediate reduction results.

It affects:

- Performance. For example, unnecessary wider accumulator may increase memory traffic.
- Accuracy. For example, using a wider accumulator may reduce rounding errors in floating-point computations.
- Correctness. For example, using an accumulator type
  that can represent all possible intermediate results prevents overflow.

Consider how the accumulator is used:

1. It is initialized by some value, which generally might be the initial value,
   an input value, or a binary operation result
   (using both the initial value and an input value, or either one of them).
2. It is passed as an argument to the binary operation,
   together with an input value or another accumulator.
3. It is updated (assigned) with the result of the binary operation.
4. Eventually it is assigned to the output value.

Following the usage pattern, the accumulator type should be
```c++
using AccT = std::decay_t<std::invoke_result_t<BinaryOp&, InitT, InputRefT>>;
```

where `AccT` is the accumulator type,
`BinaryOp` is the binary operation type,
`InitT` is the initial value type,
and `InputRefT` is `std::iterator_traits<InputIt>::reference`.
`OutputIt` does not affect the accumulator type,
because output iterators often do not have a meaningful value type
(for example, `discard_iterator` or proxy iterators).

The same rule is used by [`std::ranges::fold_left`](<https://eel.is/c++draft/alg.fold>).

The suggested rule allows controlling the accumulator type
through the binary operation, which alone should handle all needed combinations.
For example:

| Goal | Input | Output | `init` | `binary_op` | `AccT` |
|---|---|---|---|---|---|
| Performance | `float` | `float` | `0.0` | `std::plus<float>` | `float` |
| Pitfall: performance | `float` | `float` | `0.0` | `std::plus<>` | `double` |
| Accuracy | `float` | `float` | `0.0` | `std::plus<double>` | `double` |
| No overflow | `int` | `int64_t` | `0` | `std::plus<int64_t>` | `int64_t` |
| Pitfall: overflow | `int` | `int64_t` | `0` | `std::plus<>` | `int` |

The pitfalls are the same as in `std::reduce`,
and documenting the accumulator type helps users avoid them.
They also apply when `binary_op` is omitted, because the default is `std::plus<>`
(see [Default Initial Value and Binary Operation](#default-initial-value-and-binary-operation)).

A parallel reduction combines the operands in an unspecified order and grouping:
a partial result may start from an input element rather than from the initial value,
and partial results are combined with each other.
Therefore the operation imposes the following type requirements:

- `binary_op(a, b)` must be valid for any combination of `AccT`, `InitT`, and `InputRefT`.
- The result of `binary_op(a, b)` for any such combination must be convertible to `AccT`
  and assignable to an `AccT` object.
- `AccT` must be writable to the output iterator.
- `InitT` should be convertible to `AccT` (for empty segments)

Comparison with other reductions:

- In [`std::reduce`](<https://eel.is/c++draft/reduce>) and `dpl::reduce`, `AccT` is `InitT`.
- In the key-based `dpl::reduce_by_segment`, `AccT` is not documented,
  and is the input value type in practice.
- In CUB the accumulator type is not documented, and is inferred as proposed above in practice.
- In rocPRIM the accumulator type is not documented,
  and is inferred as `invoke_result_t<BinaryOp, InitT, InitT>` in practice.

**Strategy**:

- Use `std::decay_t<std::invoke_result_t<BinaryOp&, InitT, InputRefT>>` as the accumulator type.
- Document the binary operation and accumulator type requirements.
- Document how the accumulator type is determined.

### Return Value

`dpl::reduce_by_segment` returns a pair of iterators: `{output_keys_end, output_values_end}`.
`cub::DeviceSegmentedReduce::Reduce` and `rocprim::segmented_reduce` return runtime status codes.

Obviously, returning the status codes is not an option
due to the inconsistency with the existing oneDPL algorithms.

In the key-based `dpl::reduce_by_segment`,
`output_keys_end` is useful to determine the number of segments, and
`output_values_end` is useful to determine the number of reduced values,
as this information is not directly passed to the algorithm.

In the new overloads, the usefulness of the returned value depends on the segment type:

| Iterator | Variable-length | Fixed-length | Usefulness |
|---|---|---|---|
| (1) Output end | `out_first + num_segments` if `num_segments > 0`, otherwise `out_first` | `out_first + num_segments` if `num_segments > 0` and `segment_length >= 0`, otherwise `out_first` | Low: known in advance |
| (2) Output: past the last segment within `N` | `out_first + k`, `k` segments with `offsets[s + 1] <= N`; device-to-host copy and sync | `out_first + min(num_segments, N / segment_length)` (`num_segments` if `segment_length == 0`); free | Medium: tells how many segments lie entirely within the input |
| (3) Offsets end | `offsets + num_segments + 1` | — | Low: known in advance |
| (4) Input: past the last processed | `first + min(offsets[num_segments], N)`; device-to-host copy and sync | `first + min(num_segments * segment_length, N)`; free | Medium: tells where the processing stopped |
| (5) Continuation point: `{first + offsets[k], offsets + k, out_first + k}` | `k` from (2); device-to-host copy and sync | `k` from (2), input at `first + k * segment_length`; free | Medium: tells where both input and output processing stopped, most useful for resuming the reduction |

Notation: `N = last - first`

The useful values, (2), (4), and (5),
differ from the known ones only when the input does not cover all the segments,
which is expected to be a corner case.
Even then, the behavior is documented
(see [Bounding Input and Negative Offsets](#bounding-input-and-negative-offsets)),
so users who need this information can derive it from the arguments, paying the cost only then.
For variable-length segments with device policies,
returning these values requires a device-to-host copy and a synchronization.

Option (2) is only meaningful if segments that do not intersect the input
are considered uninitialized rather than empty,
which does not follow the choice from [Bounding Input and Negative Offsets](#bounding-input-and-negative-offsets).

Option (1) has low usefulness but is cheap and aligns with the existing overloads
by returning the output end.

**Strategy**: return the output end, option (1).

### Feature Macro

The API is expected to evolve,
hence a feature macro can be defined for convenience.

For example:

- `ONEDPL_HAS_SEGMENTED_ALGORITHMS 202611L` after implementing the initial version.
- `ONEDPL_HAS_SEGMENTED_ALGORITHMS YYYYMML` for future versions.

`ONEDPL_HAS_SEGMENTED_ALGORITHMS` will also cover
future segmented algorithms beyond `reduce_by_segment`.

**Strategy**: define a feature macro.

## Next Steps

The RFC may be implemented in stages, for example:

1. Implement `reduce_by_segment` with device policies support, and optionally, host policies.
2. Enable host policies, if not already done.
3. Optimize fixed-length overloads of `reduce_by_segment` by adding a dedicated implementation,
   and explore/implement tiling strategy for the variable-length overloads.
4. Support semantics with min/max values and their indices.
5. Enable identities for the custom binary operations and types.

Suggestion for the initial implementation:

- Implement `reduce_by_segment` for device policies.
- Add a dedicated implementation for the variable-length overloads.
  Consider following the approach from `dpct::device::segmented_reduce`, CUB and rocPRIM,
  which assign one segment to a work-group. Consider adapting `__work_group_reduce_kernel` for this purpose.
  The performance must not be worse than the existing alternative, `dpct::device::segmented_reduce`.
  Ideally, the algorithm should be as efficient as CUB and rocPRIM on comparable hardware, or better.
- For host policies, take one of the following approaches:
  - Add an optimized implementation for all host policies.
  - Use a serial implementation for all host policies.
  - Use a serial implementation for `seq` policy, and issue a static assertion error for other host policies.
- Enable the fixed-length overloads via the variable-length implementation,
  with offsets `i * segment_length` produced by a transform iterator over a counting iterator.
- Add tests.
- Update the [specification](../../../documentation/specification/parallel_api/algorithms.rst)
  and the [library guide](../../../documentation/library_guide/parallel_api/additional_algorithms.rst)
  with the new overloads.

## Examples

The following complete programs illustrate the proposed overloads.

`device` policy, variable-length segments, `sum` operation:

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
    constexpr int num_segments = 4;
    const int input[] = {10, 20, 5, 7, 4};
    const int host_offsets[] = {0, 2, 2, 4, 5};

    int* values = sycl::malloc_shared<int>(num_elements, queue);
    int* offsets = sycl::malloc_shared<int>(num_segments + 1, queue);
    int* result = sycl::malloc_shared<int>(num_segments, queue);

    std::copy_n(input, num_elements, values);
    std::copy_n(host_offsets, num_segments + 1, offsets);

    oneapi::dpl::reduce_by_segment(
        policy, values, values + num_elements, result,
        num_segments, offsets, 0);

    for (int i = 0; i < num_segments; ++i)
        std::cout << result[i] << ' ';
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

    oneapi::dpl::reduce_by_segment(
        oneapi::dpl::execution::par_unseq, values.begin(), values.end(),
        result.begin(), num_segments, segment_length,
        std::numeric_limits<int>::lowest(), maximum);

    for (auto it = result.begin(); it != result.end(); ++it)
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

> [!NOTE]
> Besides the questions in this section,
> some topics from the [Design Considerations](#design-considerations) section
> are partially unresolved or have no clear resolution.
> They can be found using the "Tentative Strategy" keyword.

### Different `first` and `last` Types

`last` can have a different type than `first`.
For example, it may be useful to pass `std::unreachable_sentinel` for better compatibility with CUB and rocPRIM,
assuming projects using these APIs have already built their code around and have lost the information about `last`.

Note, in this case `last` does not bound the input,
so the safety and performance benefits described in
[Bounding Input and Negative Offsets](#bounding-input-and-negative-offsets) are lost,
and the behavior falls back to option (5) from that section.

### Iterator Requirements

Should the iterator types be constrained,
or should their requirements be conveyed via the template parameter names, or only documented?

The algorithm needs random access to the input, output and offsets.
The [synopsis](#synopsis) uses generic names (`InputIt`, `OutputIt`, `OffsetIt`),
as the existing `reduce_by_segment` does (`InputIt1`, `OutputIt1`, etc.).

Possible approaches, not mutually exclusive:

- Constrain: reject non-conforming types during overload resolution,
  for example via SFINAE on `std::iterator_traits<It>::iterator_category` in C++17,
  or `std::random_access_iterator` in C++20.
  It gives earlier and clearer diagnostics,
  but has no precedent in the oneDPL iterator-based algorithms,
  and may reject types that otherwise work, such as proxy iterators or iterators with incomplete traits.
- Name: convey the requirements via the template parameter names, for example `RandomAccessIt`,
  following the convention of the C++ standard ([algorithms.requirements]).
- Document only: keep the generic names and list the requirements in the description of the algorithm.

Note, the answer may depend on [Different `first` and `last` Types](#different-first-and-last-types),
because a sentinel `last` would require constraining or documenting a sentinel-iterator relationship as well.

## Links

### C++ Standard References

- [C++ draft: `std::reduce`](https://eel.is/c++draft/reduce)
- [C++ draft: Numeric algorithm definitions](https://eel.is/c++draft/numerics.defns)
- [P3732R2: Numeric range algorithms](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p3732r2.html)
- [P3809R0: Should we make std::linalg reductions deduce return types like fold algorithms?](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3809r0.html)
- [cppreference: Fold algorithm concepts](https://en.cppreference.com/cpp/algorithm/ranges/fold_concepts)
- [cppreference: `std::reduce`](https://cppreference.com/cpp/algorithm/reduce)

### Performance Studies

These papers may be useful when implementing the algorithm efficiently.

- [Optimized Parallel Reduction for Regular and Irregular Segments on GPU (2026)](https://doi.org/10.1002/cpe.70715), [implementation](https://github.com/MichelBC/bestReduce)
- [Advancing Awkward Arrays for High-Performance CPU and GPU Processing (2026)](https://arxiv.org/abs/2607.28145)
- [GPU Acceleration of Awkward Arrays: Using Python cuda.compute (2026)](https://arxiv.org/abs/2610.08238)
- [Bridging the Vendor Gap: Enabling AMD GPU Support for Awkward Array via ROCm/HIP (2026)](https://arxiv.org/abs/2609.24628)
- [AsyncSparse: Accelerating Sparse Matrix-Matrix Multiplication on Asynchronous GPU Architectures (2026)](https://arxiv.org/abs/2604.17834)
- [GPU-Accelerated Presolving for Linear Programming (2026)](https://www.web.stanford.edu/~boyd/papers/cuPSLP.html)
- [Efficient Sparse Matrix Kernels based on Adaptive Workload-Balancing and Parallel-Reduction (2021)](https://arxiv.org/abs/2106.16064)
- [Strategies for Regular Segmented Reductions on GPU (2017)](https://hiperfit.dk/pdf/fhpc17.pdf)
- [Policy-based Tuning for Performance Portability and Library Co-optimization (2012)](https://research.nvidia.com/index.php/publication/2012-05_policy-based-tuning-performance-portability-and-library-co-optimization)
- [Merge-based Parallel Sparse Matrix-vector Multiplication (2016)](https://mgarland.org/papers/)
