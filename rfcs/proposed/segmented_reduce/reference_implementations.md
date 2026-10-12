# Reference Implementations


## oneDPL/DPCT algorithms

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

## Other Libraries

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
- The initial value is applied to each segment, not only empty segments.
- The accumulator type is not stated explicitly.
  AI-assisted research shows that it is inferred from `InputValueIt`, `InitValueT`, and `BinaryOp`.
- The overload with the execution environment allows control over floating-point determinism,
  memory allocation, and tuning parameters such as block size and elements per thread.

There are `Sum`, `Min`, and `Max`
specialized methods that supply their own binary operation and initial value.
In `Min` and `Max` documentation, the initial value is said to apply to each segment.
All these functions also have 3 overloads as the general `Reduce`.
Below is an example of `Min`.

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

The documentation states that the initial value is obtained as shown in the table below.
It implies that the custom types are supported, e.g. via specializing these mechanisms.

| Overload | Initial value |
|---|---|
| `Sum` | `OutputT{}` |
| `Min` | `cuda::std::numeric_limits<T>::max()` |
| `Max` | `cuda::std::numeric_limits<T>::lowest()` |

The documentation says nothing about the maximum value for the number of the input elements,
segment count and their maximum size, however there are implications based on the interfaces:

| Quantity | Declared type | Meaning |
|---|---|---|
| Number of segments | `cuda::std::int64_t num_segments` | The number of segments can be up to 2^63 - 1 |
| Variable-length segment offsets | Templated begin/end iterators | Their common type can represent the maximum number of input elements, but it is not guaranteed |
| Fixed segment length | `int segment_size` | The maximum segment size is 2^31 - 1

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
- The initial value is applied to each segment, not only empty segments.
- The binary operation must be associative and commutative (in practice, it is a documentation gap).
- `Config` allows customization of the implementation's tuning parameters.

Implementation: assign one work-group to each segment.
Batch when `num_segments > UINT_MAX / work_group_size`.
A single kernel launch unless batched.

`hipcub::DeviceSegmentedReduce` uses `rocprim::segmented_reduce` as a backend for all its flavours.
It supports only variable-length segments.
