# Batched Sort

## Introduction

The sort API in oneDPL's main APIs and in the kernel templates sort one sequence
of data at a time, there is no API to launch a coordinated launch for a batch
of multiple segments at once.

A few mentioned use cases:
* LLM token sampling (component of sorts prior to top-p). Batches are simultaneous
inferences going on at once which are selecting tokens, token probabilities are
sorted with one segment per inference of equal size, and then the top p
probabilities are kept and chosen from
* Top-k general purpose fallback - batched sort as a component of partial sort.
By chunking a large sequence into segments, and sorting each, then taking the
top k elements of each chunk and sorting those, this is an implementation of
partial sort / top-k which fits any general size and value of k.

Sorting many individual segments separately with separate kernel launches is far
slower than launching many segments as a batch. CUB handles this case with
DeviceSegmentedRadixSort, which handles arbitrary segment sizes. We have seen
demand however for fixed segment sized sort, which can take advantage of shared
structure known at compile time for a fast algorithm.

### Requirements

Required:
* Stable sort - required for top-p and top-k for reproducibility and tie breaking
* runtime specified segment size
* Out-of-place
* key/value pair sort and key only sort
* Important total sizes: 32K - ~256M (should support up to size 2^30)
* Important segment sizes: 2k - 128K (should support arbitrary segment size for correctness)

Preferred:
Temporary data preallocated and supplied

Not required:
custom comparator

## Proposal

### API

namespace:
`oneapi::dpl::experimental::kt::gpu`

```
template <bool __is_ascending = true, std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysIterator1,_ValsIterator1, typename _KeysIterator2, typename _ValsIterator2>
sycl::event
batched_radix_sort_by_key(sycl::queue __q,
                          _KeysIterator1 __keys_first, _KeysIterator1 __keys_last
                          _ValsIterator1 __vals_first,
                          _KeysIterator2 __keys_out_first, _ValsIterator2 __vals_out_first,
                          std::size_t segment_size, _KernelParam __param = {} )
```

Replicates for permutations of:
* iterators and ranges 
  Replace __keys_first, __keys_last with a single keys range, replace output
  begin iterator with range, same with value begin iterators.
* In-place (no out keys or ranges)
* key value pairs or just key only
  Key only sorts remove "_by_key" and any value sequence arguments.
* merge sort and radix sort
  Replace "radix" with "merge" and remove radix template parameter

We chose the term `batched`, because it helps indicate that we are launching a
batch of fixed size independent sorts. CUB already has segmented radix sort
which allows arbitrary and varied sizes in a single launch. If we decide to
provide a similar API, we reserve `segmented` for that. Kernel templates are
meant to be a thin layer around an algorithm / kernel, so we provide the
individual APIs for each stable sort approach, merge and radix.

Compile time parameters:
* ascending / decending - direction of the sort
* radix - the radix of the sort, only applicable to radix sort
* KT Params: dpwi, wgsize - data per work item and wgsize

Runtime Parameters:
* queue - sycl queue
* data params: iterator and range parameters for in-place and out-of-place
* segment_size: size of individual segments

### Implementation Details

#### Segment size impact on algorithm
The current best stable sort depends on the size of the sort. For sequences
larger than can fit in one workgroup, onesweep radix sort is the fastest.

For sequences which can fit one or more into a single workgroup, the current
best stable sort is the esimd single workgroup radix sort. Proof of concept work
has shown that merge sort or possibly bitonic sort may be our best option for
small segment sizes.

At small segment sizes bitonic sort is the fastest option. However, it has a few 
downsides.
1) It is not inherently stable.
2) It is a fixed power of two segment size, so segments must be padded.

You can augment bitonic sort to make it stable, but you may lose most or all
performance gains vs merge sort via merge path.

#### Plan
Implement two kernels:
1) workgroup merge path sort which can handle multiple
segments at a time which fit into a single workgroup (wgsize * dpwi)
2) Modified oneSweep radix sort which will handle size segment, but is best for
segments which do not fit into a single workgroup (wgsize * dpwi).

It is the user's responsibility to invoke the correct algorithm. We can provide
a quick and easy dispatch layer by checking if segments are smaller or larger
than the dpwi * wgsize, possibly just via documentation.

There is opportunity to achieve better performance at the smallest segment
sizes in the future via subgroup level merge path sort or bitonic sort, but this
will be deferred to later.

#### Expectations from proof of concept work

Expected BMG speedup @ 4M total elements vs individual sequential calls:

| segment_size | algorithm         | speedup        |
|--------------|-------------------|----------------|
|   smaller    | wg-merge (sg?)    | very large     |
|     256      | wg-merge (sg?)    |   ~1500x       |
|    1024      | wg-merge          |   ~160x        |
|    2048      | wg-merge          |   ~80x         |
|    4096      | wg-merge          |   ~75x         |
|     16K      | wg-merge          |   ~14x         |
|     32k      | wg-merge          |   ~7x          |
|    larger    | modified onesweep | size dependent |

It seems like 1.5-4x could be possible for segments up to 256k on PVC.
BMG seems like a less clear win.  We always have the option to merely launch
individual radix sorts, so we should do no worse than that.

## Open Questions

* How (if at all) should we allow users to size and provide their own temporary allocation to the algorithm?
 * CUB-style two-call: call once with null to get the size, then call again with the buffer
 * SYCL async memory pool, passing a memory_pool
 * An env object holding the queue and pool, like newer CUB (possibly with defaults)
 * A separately named query function, like batched_stable_sort_alloc_size
  My recommendataion is to use an async memory pool, and if none specified, create
   one with the provided queue.

* What are the exact type support requirements?
  * sycl::half?
  * fp16?
  * fp8?