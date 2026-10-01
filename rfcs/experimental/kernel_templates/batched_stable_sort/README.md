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
Stable sort
Important total sizes: 32K - ~256M
Important segment sizes: 2k - 128K
runtime specified segment size

Preferred:
Temporary data preallocated and supplied

Not required:
custom comparator

## Proposal

### API

`oneapi::dpl::experimental::kt::gpu::batched_merge_sort<ascending>(queue, key_in, key_out, segment_size, kt_kernel_param<dpwi, wgsize>)`

and

`oneapi::dpl::experimental::kt::gpu::batched_radix_sort<ascending, radix>(queue, key_in, key_out, segment_size, kt_kernel_param<dpwi, wgsize>)`

We chose the term `batched`, because it helps indicate that we are launching a
batch of fixed size independent sorts. CUB already has segmented radix sort
which allows arbitrary and varied sizes in a single launch. If we decide to
provide a similar API, we reserve `segmented` for that. We chose `stable` sort
rather than `radix` because it describes the important semantics, and we may
prefer to use another sort under the hood depending on batch size.

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

 - How (if at all) should we allow users to size and provide their own temporary allocation to the algorithm?
