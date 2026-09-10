# Segmented Sort and Segmented Reduce Kernel Templates

## Introduction

There is a gap in oneDPL's offering for segmented operations where a group of segments must be processed, each
segment being processed independently from the other segments, either with a sorting or reduction operation.

NVidia provides DeviceSegmentedSort, DeviceSegmentedRadixSort and DeviceSegmentedReduce within CUB, with a variety of
APIs for each group of algorithms.
* DeviceSegmentedSort, All combinations of the following binary choices:
  * Keys + Pairs - sorting just keys or (key, value) pairs
  * Out-of-place vs DoubleBuffer - input and output sequences, or using a "double buffer" with an indicator to specify
    which buffer of the double buffer is the current input data, and which is the output. This can be used to avoid
    extra passes over the data when using radix sort with even number of radix steps.
  * Stable vs Non-stable - Stability of the sort
* DeviceSegmentedRadixSort, same as DeviceSegmentedSort, but always stable.
* DeviceSegmentedReduce, The following APIs - Reduce, Sum, Min, ArgMin, Max, ArgMax

SYCLomatic helper headers provide a naive implementation of some similar APIs which covers functionality within oneAPI
and SYCL, but these are not meant to be performant specialized APIs, especially when used on a GPU device. They operate
by running a sequential loop of parallel segment operations, or a parallel loop of serial segment operations. On the
host, at least one these naive implementations is likely to provide reasonable performance but not on the GPU.

This leaves a gap on the GPU for a device specific algorithm where kernel launch overheads and massive numbers of
threads demand a more sophisticated algorithm to process such use cases. This agrees with NVIDIA's approach to provide
these APIs within CUB which is GPU device backend only, but not within Thrust, which has both host and GPU device
backends. With this in mind, it makes sense to provide these API with kernel templates which are specific to a
GPU device, at least to start. If we find motivation to provide host side implementations of these APIs in the future,
we can revisit that then.

## Proposal

I propose the following Kernel Template APIs:

```
template <typename Iter1, typename Iter2, typename Iter3, typename Comparator>
sycl::event
segmented_sort(Iter1 in_begin, Iter2 out_begin, Iter3 segment_offsets_begin, Iter3 segment_offsets_end, Comparator comp)

template <typename Iter1, typename Iter2, typename Iter3, typename Comparator>
sycl::event
segmented_stable_sort(Iter1 in_begin, Iter2 out_begin, Iter3 segment_offsets_begin, Iter3 segment_offsets_end,
                      Comparator comp)

template <typename Iter1, typename Iter2, typename Iter3, typename Iter4, typename Iter5, typename Comparator>
sycl::event
segmented_sort_by_key(Iter1 keys_in_begin, Iter2 values_in_begin, Iter3 keys_out_begin, Iter4 value_out_begin,
                      Iter5 segment_offsets_begin, Iter5 segment_offsets_end, Comparator comp)

template <typename Iter1, typename Iter2, typename Iter3, typename Comparator>
sycl::event
segmented_stable_sort_by_key(Iter1 keys_in_begin, Iter2 values_in_begin, Iter3 keys_out_begin, Iter4 value_out_begin,
                             Iter5 segment_offsets_begin, Iter5 segment_offsets_end, Comparator comp)

template <typename Iter1, typename Iter2, typename Iter3, typename BinaryOp>
sycl::event
segmented_reduce(Iter1 in_begin, Iter2 out_begin, Iter3 segment_offsets_begin, Iter3 segment_offsets_end,
                 BinaryOp binary_op)
```

These APIs will sort segments into size classes, group together small, medium, and large segments, and handle each size
class differently. Small segments are ones which fit within a subgroup. These segments will be launched together
in a single kernel, where subgroups each sort or reduce a single segment. Medium segments are segments which can fit
within a single workgroup.  Medium segments will be launched by a single kernel, where each workgroup can iterate
through a list of segments, and fully process them with a sort or reduce. Large segments are large enough to require
multiple kernels to handle them, and segments are launced individually using existing sort and reduce infrastructure.

We have lots of the building blocks for these APIs already within oneDPL. The existing oneDPL reduce and sort
functionality should be used for large segments. Existing single-workgroup functionality should be used as well. Some
refactoring is necessary to to be able to take advantage of running multiple segments from a single launch in a
single-workgroup kernel.

### New Required Functionality
The significant new components which will need to be added to accomplish this is a partitioning kernel which sorts
segments into size classes and determines their order, and kernels to implement operations on small segments. Both the
partitioning of segments into classes and the infrastructure to run small segments should be shared between all
segmented algorithms.

#### Partitioning Phase
Partitioning kernel should be able to be handled by using reduce_then_scan components (with some adjustment) over
segment offsets where reduction determines and counts segments in each class, and then the final write operation
updates a corresponding list with the segment id.

#### Small Segments Sorting
We currently dont have a coordinated subgroup sorter within oneDPL. `__subgroup_bubble_sorter` is used in merge path,
but does not coordinate between workitems within a subgroup, but rather just uses a bubble sort on each workitem
individually for a small fixed size data_per_workitem. For non-stable sorts, we can use bitonic sort, as is done in [1].
For stable sort APIs, we will need something else, which is an open question. Perhaps bubble sort per work-item followed
by merging within the subgroup.

While CUB provides multiple flavors of reduce to cover many different APIs, I think we can just provide segmented_reduce
and offer some wrappers within the SYCLomatic helpers to cover the same functionality without cluttering our interface.


## Alternative Options

### Continue to use exisiting naive implementation
We could of course continue to use the naive implementation described above within the SYCLomatic helper headers. This
can have huge overheads and does not seems acceptible from a performance perspective.  We certainly would not include
such a solution within oneDPL.

### Skip small class
We could skip the small class of segments when sorting, and only consider single-workgroup and large class segments.
This will result in significant overheads for cases with large number of small segments, but could be a possible initial
implementation for simplicity. It seems clear that this will be inferior for any use cases which include more than a few
small segments.

## Testing
New APIs require testing with a variety of use cases testing different configurations of segment counts, sizes and
configurations. It should contain some randomly generated segment sizes, as well as specific end cases with unique
combinations of segment sizes to test edge cases.

## Open Questions
### Questions for Experimental Phase:
* Should we provide a way for users to pass in temporary storage for use, or allocate each time based on the use case?
  * If so, how do we calculate the storage required / preallocate or inform the user to allocate
* Does it make sense to provide a direct response to DoubleBuffer and APIs which use it?

### Questions for Proposal Phase:
* Can we utilize sycl joint reduce with good performance, or should we implement our own intrasubgroup reduction?
* For stable sort APIs what sort should we use for small groups? 
* How to deal with small segments which do not fill the subgroup?

## Related Papers

For further reading and related work, you can refer to the following papers and resources:

1. [Efficient Merging and Sorting on GPUs](https://dl.acm.org/doi/pdf/10.1145/3079079.3079105)
2. [Segmented Sort](https://moderngpu.github.io/segsort.html)
3. [An Evaluation of Fast Segmented Sorting Implementations on GPU](https://www.researchgate.net/publication/357947992_An_evaluation_of_fast_segmented_sorting_implementations_on_GPUs) 
