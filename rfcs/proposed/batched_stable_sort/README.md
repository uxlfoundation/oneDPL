# Batched Stable Sort

## Introduction

oneDPL's sort APIs, including the kernel templates, sort one sequence at a time. There is no API to
sort a batch of segments in a single launch.

This RFC proposes new algorithms for [Kernel Templates][kt].

A few mentioned use cases:
* LLM token sampling (component of sorts prior to top-p). Batches are simultaneous inferences going
  on at once which are selecting tokens, token probabilities are sorted with one segment per
  inference of equal size, and then the top-p probabilities are kept and chosen from.
* Top-k general purpose fallback - batched sort as a component of partial sort. By chunking a large
  sequence into segments, and sorting each, then taking the top k elements of each chunk and
  sorting those, this is an implementation of partial sort / top-k which fits any general size and
  value of k.

Sorting many individual segments separately with separate kernel launches is far slower than
launching many segments as a batch. We have seen demand for fixed segment sized sort, which can
take advantage of shared structure for a fast algorithm.

Existing solutions:
* CUB `DeviceSegmentedRadixSort`: one work-group radix sorts each segment; arbitrary segment
  offsets, key-only and key/value.
* CUB `DeviceSegmentedSort`: stable and unstable variants; buckets segments by size, using merge
  sort for small and medium segments and radix sort for large.
* SYCLomatic `dpct::segmented_sort_[keys/pairs]`: a naive implementation for correctness (serial
  sort per work-item, or a host loop of parallel sorts); not necessarily stable, and blocking.

### Requirements

Required:
* Stable sort - required for top-p and top-k for reproducibility and tie breaking
* runtime specified segment size
* Out-of-place (in-place is also supported, as a secondary priority)
* key/value pair sort and key only sort
* Important total sizes: 32K - ~256M (should support sizes below 2^30, matching `kt::gpu::radix_sort`)
* Important segment sizes: 2K - 128K (should support arbitrary segment size for correctness)

Preferred:
Temporary data preallocated and supplied

Not required:
custom comparator

## Proposal

### API

namespace:
`oneapi::dpl::experimental::kt::gpu`

```c++
template <bool __is_ascending = true, std::uint8_t __radix_bits = 8, typename _KernelParam,
          typename _KeysIterator1, typename _ValsIterator1,
          typename _KeysIterator2, typename _ValsIterator2>
sycl::event
batched_radix_sort_by_key(sycl::queue __q,
                          _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                          _ValsIterator1 __vals_first,
                          _KeysIterator2 __keys_out_first, _ValsIterator2 __vals_out_first,
                          std::size_t __segment_size, _KernelParam __param = {});
```

Replicates for permutations of:
* iterators and ranges
  Replace `__keys_first`, `__keys_last` with a single keys range, replace output begin iterator with
  range, same with value begin iterators.
* In-place (no out keys or ranges)
* key value pairs or just key only
  Key only sorts remove "_by_key" and any value sequence arguments.
* radix sort and (optionally) merge sort
  Replace "radix" with "merge" and remove radix template parameter

We chose the term `batched`, because it helps indicate that we are launching a batch of fixed size
independent sorts. CUB already has segmented radix sort which allows arbitrary and varied sizes in
a single launch. If we decide to provide a similar API, we reserve `segmented` for that. Kernel
templates are meant to be a thin layer around an algorithm / kernel, so we provide an API for each
stable sort approach: radix, and optionally merge.

Compile time parameters:
* ascending / descending - direction of the sort
* radix - the radix of the sort, only applicable to radix sort
* `kernel_param`: `data_per_workitem`, `workgroup_size`

Runtime Parameters:
* queue - sycl queue
* data params: iterator and range parameters for in-place and out-of-place
* `__segment_size`: size of individual segments

### Semantics

* `__segment_size == 0` is rejected with an assertion.
* For `__segment_size > 0`, `n == 0` is a no-op; otherwise, `n % __segment_size != 0` is rejected
  with an assertion.
* `batched_radix_sort*` sorts with one work-group radix sort when
  `__segment_size <= data_per_workitem * workgroup_size`, and with Modified OneSweep otherwise.
  If the `kernel_param` is not supported by Modified OneSweep (currently 8 radix bits with 512 or
  1024 work-items), `__segment_size > data_per_workitem * workgroup_size` is rejected with an
  assertion.
* `batched_merge_sort*` rejects `__segment_size > data_per_workitem * workgroup_size` with an
  assertion.
* One work-group radix sort and `batched_merge_sort*` stage a full work-group tile of keys (and
  values) in local memory, so `data_per_workitem * workgroup_size * (sizeof(key) + sizeof(value))`
  must not exceed the device's local memory (`sycl::info::device::local_mem_size`). Choosing a
  `kernel_param` which satisfies this is the user's responsibility, as for other kernel templates.
* Input and output must not overlap. Full aliasing is served by the in-place overloads; partial
  overlap is not supported.
* Supported key types, data passing mechanisms (USM pointers, `oneapi::dpl::begin` / `end`,
  `sycl::buffer`, `views::all` / `views::subrange`), and the returned `sycl::event` match the
  existing `kt::gpu::radix_sort[_by_key]`.
* Failure to allocate internal global memory throws `std::bad_alloc`, as for other kernel
  templates (subject to the [temporary allocation](#open-questions) question).

### Example

Top-p sampling: sort each of `B` rows of `V` token probabilities in descending order, carrying token
ids.

```c++
namespace kt = oneapi::dpl::experimental::kt;

sycl::queue q{sycl::gpu_selector_v};
constexpr std::size_t B = 1024, V = 4096, n = B * V;
float* probs = sycl::malloc_device<float>(n, q);               // B rows of V probabilities
std::uint32_t* ids = sycl::malloc_device<std::uint32_t>(n, q); // ids[i] = i % V
float* probs_out = sycl::malloc_device<float>(n, q);
std::uint32_t* ids_out = sycl::malloc_device<std::uint32_t>(n, q);
// ... fill probs and ids ...
q.wait(); // The current API has no dependency-event parameter.

// A row fits in one tile of 8 * 512 = 4096 keys + ids (32 KB of SLM), so one work-group radix
// sort is used; a larger V would be sorted by Modified OneSweep with the same call
using param_t = kt::kernel_param<8, 512>;
sycl::event e = kt::gpu::batched_radix_sort_by_key<false, 8>(q, probs, probs + n, ids, probs_out,
                                                             ids_out, V, param_t{});
e.wait();
// each row of probs_out is sorted descending; ties keep the lower token id first
```

### Implementation Details

#### Segment Size Impact on Algorithm
The best stable sort depends on segment size. Candidates considered:

| algorithm             | scope              | notes                                             |
|-----------------------|--------------------|---------------------------------------------------|
| OneSweep per segment  | all                | current workaround; baseline                      |
| Composite OneSweep    | all                | segment id prepended to key; easy worst-case gain |
| One work-group radix  | segment fits in wg | proposed; packs multiple segments per wg          |
| Modified OneSweep     | all                | proposed; best when segments exceed one wg        |
| Work-group merge path | segment fits in wg | proposed; packs multiple segments per wg          |
| Sub-group merge path  | segment fits in sg | deferred; capped near 1024 by SLM                 |
| Bitonic               | small (< ~1024)    | deferred; unstable unless augmented, pads to pow2 |

#### Plan
Implement two kernels, and optionally a third:
1) One work-group radix sort, for segments which fit into a single work-group.
   * All radix stages run in a single kernel, with no global histogram or lookback.
   * Each segment is owned by `ceil(__segment_size / (32 * data_per_workitem))` sub-groups, and
     multiple segments may be packed into one work-group, at sub-group granularity.
   * Per stage: sub-groups rank their keys with ballots into SLM histograms, the histograms of a
     segment are scanned to give its bin offsets, and keys are reordered through SLM.
   * Slots past the end of a segment are padded with the sort identity, which stays in the last bin.
   * Local memory holds a full tile of keys (and values) while they are reordered,
     `data_per_workitem * workgroup_size * (sizeof(key) + sizeof(value))` bytes, whatever the
     segment size. The histograms reuse the same local memory and are never in it at the same time
     as the data. Histograms should safely fit in SLM for the relevant device targets, even in the
     scenarios when they are larger than the data itself, so they do not change whether a
     `kernel_param` fits.
   * It is the user's responsibility to choose a `kernel_param` which fits in local memory.
2) Modified OneSweep radix sort, which handles any segment size, but is best for segments which do
   not fit into a single work-group.
   * Global histogram and bin offset scan per (segment, radix stage).
   * Sweep tiles are aligned to segments, with one decoupled lookback chain per segment; tile 0 of
     each segment seeds from that segment's offsets.
   * The last tile of each segment is partial: pad keys in registers and mask writes at the
     segment end.
   * Shares implementation with the existing `kt::gpu::radix_sort` where possible, which may
     require refactoring its kernels to be segment-aware.
3) (Optional, or secondary priority) Work-group merge path sort, for segments which fit into a
   single work-group (`workgroup_size * data_per_workitem`). Multiple segments may be packed into
   one work-group.
   * Load into registers and stable sort each work-item's `data_per_workitem` keys (leaf sort).
     The last work-item of a segment may hold fewer than `data_per_workitem` keys; its unused
     slots are padded and excluded from the merges and the final write.
   * `ceil(log2(ceil(__segment_size / data_per_workitem)))` merge rounds (zero when the segment
     fits in one work-item): registers → SLM, barrier, each work-item binary-searches its diagonal
     (co-rank) for its merge path start, then merges `data_per_workitem` keys from SLM back into
     registers. When a round has an odd number of runs, the trailing run is carried over unchanged.
   * Measured to be faster than one work-group radix sort for segments smaller than roughly 1K-2K
     elements (see [Expectations](#expectations-from-proof-of-concept-work)).

All kernels are SYCL (not ESIMD) implementations in the `gpu` namespace. Work-group merge path and
one work-group radix sort have no cross-work-group communication. Modified OneSweep relies on
decoupled lookback, so like `kt::gpu::radix_sort` it requires parallel forward progress between
work-groups, and initially carries the same device and runtime restrictions.

#### Dispatch
With one work-group radix sort and Modified OneSweep behind the same API (see
[open questions](#open-questions)), `batched_radix_sort` serves every segment size, and picks its
kernel with a simple rule:
* `__segment_size <= data_per_workitem * workgroup_size`: one work-group radix sort
* Otherwise: Modified OneSweep

The user chooses which kernel runs through the `kernel_param`, and is responsible for a tile of
keys (and values) which fits in local memory whenever one work-group radix sort is chosen. If
`batched_merge_sort` is optionally provided, the documentation will recommend it for segments
smaller than roughly 1K elements which fit in its tile, where it is faster.

The local memory bound depends on key and value size, so larger types reduce the largest segment
size served by one work-group radix sort and `batched_merge_sort`. For example, at 64 KB of local
memory, the tile is limited to 16K `std::uint32_t` keys, 8K `std::uint32_t` key-value pairs, or 4K
`std::uint64_t` key-value pairs.

#### Expectations from Proof of Concept Work

Expected speedup vs individual sequential `kt::gpu::radix_sort` (OneSweep) calls per segment.
64M `std::uint32_t` keys, out-of-place, key only, out-of-order queue. Merge path numbers are
measured from a proof of concept; modified OneSweep numbers are projected estimates.

| segment size | algorithm         | BMG speedup    | PVC speedup    |
|--------------|-------------------|----------------|----------------|
|     256      | wg-merge          |   ~1700x       |   ~6400x       |
|    1024      | wg-merge          |   ~340x        |   ~1200x       |
|    2048      | wg-merge          |   ~160x        |   ~550x        |
|    4096      | wg-merge          |   ~75x         |   ~225x        |
|     16K      | wg-merge          |   ~14x         |   ~38x         |
|     32K      | wg-merge          |   ~7x          |   ~15x         |
|    larger    | modified OneSweep | size dependent | size dependent |

1.5-4x looks possible for segments up to 256K on PVC. BMG is a less clear win. As segments get very
large, it is possible that a separate kernel launch per segment could be faster.

One work-group radix sort versus work-group merge path sort, measured on the prototype
implementation: the geomean over total sizes 2^8-2^28 of the best one work-group radix time divided
by the best merge time, `float` keys and values, 8 radix bits. Below 1, radix sort is faster.

| segment size | BMG keys | PVC keys | BMG by key | PVC by key |
|--------------|----------|----------|------------|------------|
|      32      |   3.66   |   3.09   |    3.11    |    2.76    |
|     128      |   1.71   |   1.66   |    1.54    |    1.51    |
|     256      |   1.31   |   1.35   |    1.22    |    1.25    |
|    1024      |   1.02   |   1.14   |    0.96    |    1.08    |
|    4096      |   0.78   |   0.92   |    0.77    |    0.92    |
|     16K      |   0.63   |   0.75   |     -      |     -      |

Merge path sort is faster at every total size for segments up to 256 elements. At 1024 elements the
two are close, and radix sort wins at large total sizes. From 4096 elements, radix sort wins in
most cases. 2048 element segments were not measured, and 16K segments do not fit a by key tile.

## Testing

Following the [Kernel Templates testing guidance][kt-testing]:
* Verify each segment against a per-segment `std::stable_sort` reference. Use values holding the
  original index to check stability.
* Segment sizes: 1, small non-power-of-two, around `data_per_workitem * workgroup_size` (-1, ==, +1
  for radix), large segments, and a single segment (`__segment_size == n`).
* All supported key types, including floating point edge cases (-0.0, +0.0, infinities), and key
  and value types of different widths.
* Ascending and descending; key only and by key; out-of-place and in-place; iterators and ranges;
  USM and `sycl::buffer`.
* `n == 0` is a no-op; assertions fire for `__segment_size == 0`, `n % __segment_size != 0`, and
  `__segment_size > data_per_workitem * workgroup_size` for merge sort, and for radix sort with a
  `kernel_param` not supported by Modified OneSweep.
* Reuse the existing `kt::gpu::radix_sort` test setup to cover a representative sample of kernel
  parameters.

## Open Questions

* How (if at all) should we allow users to size and provide their own temporary allocation to the
  algorithm? This is a concrete case of the Kernel Templates questions
  [Reporting Global and Local Memory Requirements][kt-mem-req] and
  [External Allocation of Global Memory][kt-ext-alloc].
  * CUB-style two-call: call once with null to get the size, then call again with the buffer
  * SYCL async memory pool, passing a `memory_pool`
    (see [PR #2842](https://github.com/uxlfoundation/oneDPL/pull/2842))
  * An env object holding the queue and pool, like newer CUB (possibly with defaults)
  * A separately named query function, like `batched_stable_sort_alloc_size`

  My recommendation is to use an async memory pool, and if none specified, create one with the
  provided queue.

* Should only radix sort be provided? One work-group radix sort and Modified OneSweep together
  serve every segment size, which would leave a single algorithm and API. The cost is giving up
  merge path sort's advantage for segments below roughly 1K elements, which is up to 3-4x at 32
  element segments.
    * My suggestion would be to make the initial offering radix only, and provide the merge API
      separately as a secondary priority.

* Should one work-group radix sort be offered through the same `batched_radix_sort` API, with the
  clean dispatch rule: one work-group radix sort if at least one segment fits in a tile
  (`__segment_size <= data_per_workitem * workgroup_size`), Modified OneSweep otherwise? The
  alternative is a separately named kernel template, keeping each kernel template a thin layer
  over one kernel. With a shared API, the `kernel_param` also decides which kernel runs, and its
  tile of keys (and values) must fit in local memory whenever the one work-group kernel is chosen.
    * My suggestion would be to provide these in the same API, and dispatch based on the simple
      rule of whether at least one segment fits into `data_per_workitem * workgroup_size`.

* What are the exact type support requirements?
  * `sycl::half`?
  * `sycl::ext::oneapi::bfloat16`?
  * fp8?

* Do we need to accept input `sycl::event`s to order against previous work in an out-of-order
  queue? See the Kernel Templates question
  [Asynchronous Execution and Dependency Chaining][kt-async].

[kt]: ../../experimental/kernel_templates/README.md
[kt-testing]: ../../experimental/kernel_templates/README.md#testing
[kt-mem-req]: ../../experimental/kernel_templates/README.md#reporting-global-and-local-memory-requirements
[kt-ext-alloc]: ../../experimental/kernel_templates/README.md#external-allocation-of-global-memory
[kt-async]: ../../experimental/kernel_templates/README.md#asynchronous-execution-and-dependency-chaining
