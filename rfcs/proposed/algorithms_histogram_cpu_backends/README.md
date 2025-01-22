# Host Backends Support for the Histogram APIs

## Introduction
The oneDPL library added histogram APIs, currently implemented only for device policies with the DPC++ backend. These
APIs are defined in the oneAPI Specification 1.4. Please see the
[oneAPI Specification](https://github.com/uxlfoundation/oneAPI-spec/blob/main/source/elements/oneDPL/source/parallel_api/algorithms.rst#parallel-algorithms)
for details. The host-side backends (serial, TBB, OpenMP) are not yet supported. This RFC proposes extending histogram
support to these backends.

The pull request for the proposed implementation exists [here](https://github.com/oneapi-src/oneDPL/pull/1974).

## Motivations
There are many cases to use a host-side serial or a host-side implementation of histogram. Another motivation for adding
the support is simply to be spec compliant with the oneAPI specification.

## Design Considerations

### Key Requirements
Provide support for the `histogram` APIs with the following policies and backends:
- Policies: `seq`, `unseq`, `par`, `par_unseq`
- Backends: `serial`, `tbb`, `openmp`

Users have a choice of execution policies when calling oneDPL APIs. They also have a number of options of backends which
they can select from when using oneDPL. It is important that all combinations of these options have support for the
`histogram` APIs.

### Performance
Histogram algorithms typically involve minimal computation and are likely to be memory-bound. So, the implementation prioritizes
reducing memory accesses and minimizing temporary memory traffic.

For CPU backends, we will focus on input sizes ranging from 32K to 4M elements and 32 - 4k histogram bins. Smaller sizes
of input may best be suited for serial histogram implementation, and very large sizes may be better suited for GPU
device targets. Histogram bin counts can vary from use case to use case, but the most common rule of thumb is to size
the number of bins approximately to the cube root of the number of input elements. For our input size ranges this gives
us a range of 32 - 256. In practice, some users find need to increase the number of bins beyond that rough rule.
For this reason, we have aelected our histogram size range to 32 - 4k elements.

### Memory Footprint
There are no guidelines here from the standard library as this is an extension API. Still, we will minimize memory
footprint where possible.

### Code Reuse
We want to minimize adding requirements for parallel backends to implement, and lift as much as possible to the
algorithm implementation level. We should be able to avoid adding a `__parallel_histogram` call in the individual
backends, and instead rely upon `__parallel_for`.

### SIMD/openMP SIMD Implementation
Currently oneDPL relies upon openMP SIMD to provide its vectorization, which is designed to provide vectorization across
loop iterations, oneDPL does not directly use any intrinsics.

There are a few parts of the histogram algorithm to consider. For the calculation to determine which bin to increment
there are two APIs, even and custom range which have significantly different methods to determine the bin to increment.
For the even bin API, the calculations to determine selected bin have some opportunity for vectorization as each input
has the same mathematical operations applied to each. However, for the custom range API, each input element uses a
binary search through a list of bin boundaries to determine the selected bin. This operation will have a different
length and control flow based upon each input element and will be very difficult to vectorize.

Next, let's consider the increment operation itself. This operation increments a data dependent bin location, and may
result in conflicts between elements of the same vector. This increment operation therefore is unvectorizable without
more complex handling. Some hardware does implement SIMD conflict detection via specific intrinsics, but this is not
available via OpenMP SIMD. Alternatively, we can multiply our number of temporary histogram copies by a factor of the
vector width, but it is unclear if it is worth the overhead. OpenMP SIMD provides an `ordered` structured block which
we can use to exempt the increment from SIMD operations as well. However, this often results in vectorization being
refused by the compiler. Initial implementation will avoid vectorization of this main histogram loop.

Last, for our below proposed implementation there is the task of combining temporary histogram data into the global
output histogram. This is directly vectorizable via our existing brick_walk implementation, and will be vectorized when
a vector policy is used.

### Serial Backend
We plan to support a serial backend for histogram APIs in addition to openMP and TBB. This backend will handle all
policies types, but always provide a serial unvectorized implementation. To make this backend compatible with the other
approaches, we will use a single temporary histogram copy, which then is copied to the final global histogram. In our
benchmarking, using a temporary copy performs similarly as compared to initializing and then accumulating directly into
the output global histogram. There seems to be no performance motivated reason to special case the serial algorithm to
use the global histogram directly.

## Existing APIs / Patterns

### count_if
`histogram` is similar to `count_if` in that it conditionally increments a number of counters based upon the data in a
sequence. `count_if` relies upon the `transform_reduce` pattern internally, and returns a scalar-typed value and doesn't
provide any function to modify the variable being incremented. Using `count_if` without significant modification would
require us to loop through the entire sequence for each output bin in the histogram. From a memory bandwidth
perspective, this is untenable. Similarly, using a `histogram` pattern to implement `count_if` is unlikely to provide a
well-performing result in the end, as contention should be far higher, and `transform_reduce` is a very well-matched
pattern performance-wise.

### parallel_for
`parallel_for` is an interesting pattern in that it is very generic and embarrassingly parallel. This is close to what
we need for `histogram`. However, we cannot simply use it without any added infrastructure. If we were to just use
`parallel_for` alone, there would be a race condition between threads when incrementing the values in the output
histogram. We should be able to use `parallel_for` as a building block for our implementation, but it requires some way
to synchronize and accumulate between threads.

## Alternative Approaches

### Atomics
This method uses atomic operations to remove the race conditions during accumulation. With atomic increments of the
output histogram data, we can merely run a `parallel_for` pattern.

To deal with atomics appropriately, we have some limitations. We must either use standard library atomics, atomics
specific to a backend, or custom atomics specific to a compiler. `C++17` provides `std::atomic<T>`, however, this can
only provide atomicity for data which is created with atomics in mind. This means allocating temporary data and then
copying it to the output data. `C++20` provides `std::atomic_ref<T>` which would allow us to wrap user-provided output
data in an atomic wrapper, but we cannot assume `C++20` for all users. OpenMP provides atomic operations, but that is
only available for the OpenMP backend. The working plan was to implement a macro like `_ONEDPL_ATOMIC_INCREMENT(var)`
which uses an `std::atomic_ref` if available, and alternatively uses compiler builtins like `InterlockedAdd` or
`__atomic_fetch_add_n`. In a proof of concept implementation, this seemed to work, but does reach more into details than
compiler / OS specifics than is desired for implementations prior to `C++20`.

After experimenting with a proof of concept implementation of this implementation, it seems that the atomic
implementation has very limited applicability to real cases. We explored a spectrum of number of elements combined with
number of bins with both OpenMP and TBB. There was some subset of cases for which the atomics implementation
outperformed the proposed implementation (below). However, this was generally limited to some specific cases where the
number of bins was very large (~1 Million), and even for this subset significant benefit was only found for cases with a
small number for input elements relative to number of bins. This makes sense because the atomic implementation is able
to avoid the overhead of allocating and initializing temporary histogram copies, which is largest when the number of
bins is large compared to the number of input elements. With many bins, contention on atomics is also limited as
compared to the embarrassingly parallel proposal which does experience this contention.

When we examine the real world utility of these cases, we find that they are uncommon and unlikely to be the important
use cases. Histograms generally are used to categorize large images or arrays into a smaller number of bins to
characterize the result. Cases for which there are similar or more bins than input elements are not very practical in
practice. The maintenance and complexity cost associated with supporting and maintaining a second implementation to
serve this subset of cases does not seem to be justified. Therefore, this implementation has been discarded at this
time.

### Other Unexplored Approaches
* One could consider some sort of locking approach which locks mutexes for subsections of the output histogram prior to
modifying them. It's possible such an approach could provide a similar approach to atomics, but with different
overhead trade-offs. It seems quite likely that this would result in more overhead, but it could be worth exploring.

* Another possible approach could be to do something like the proposed implementation one, but with some sparse
representation of output data. However, I think the general assumptions we can make about the normal case make this
less likely to be beneficial. It is quite likely that `n` is much larger than the output histograms, and that a large
percentage of the output histogram may be occupied, even when considering dividing the input amongst multiple threads.
This could be explored if we find temporary storage is too large for some cases and the atomic approach does not
provide a good fallback.

## Proposal
After exploring the above implementation for `histogram`, the following proposal better represents the use cases which
are important, and provides reasonable performance for most cases.

### Embarrassingly Parallel Via Temporary Histograms
This method uses temporary storage and a pair of calls to backend specific `parallel_for` functions to accomplish the
`histogram`. These calls will use the existing infrastructure to provide properly composable parallelism, without extra
histogram-specific patterns in the implementation of a backend.

This algorithm does however require that each parallel backend will add a
`__enumerable_thread_local_storage<_StoredType>` struct which provides the following:
* constructor which takes a variadic list of args to pass to the constructor of each thread's object
* `get_for_current_thread()` returns reference to the current thread's stored object
* `get_with_id(int i)` returns reference to the stored object for an index
* `size()` returns number of stored objects

In the TBB backend, this will use `enumerable_thread_specific` internally. For OpenMP, we implement our own similar
thread local storage which will allocate and initialize the thread local storage at the first usage for each active
thread, similar to TBB. The serial backend will merely create a single copy of the temporary object for use. The serial
backend does not technically need any thread specific storage, but to avoid special casing for this serial backend, we
use a single copy of histogram. In practice, our benchmarking reports little difference in performance between this
implementation and the original, which directly accumulated to the output histogram.

With this new structure we will use the following algorithm:

1) Run a `parallel_for` pattern which performs a `histogram` on the input sequence where each thread accumulates into
its own temporary histogram returned by `__enumerable_thread_local_storage`. The parallelism is divided on the input
element axis, and we rely upon existing `parallel_for` to implement chunksize and thread composability.
2) Run a second `parallel_for` over the `histogram` output sequence which accumulates all temporary copies of the
histogram created within `__enumerable_thread_local_storage` into the output histogram sequence. The parallelism is
divided on the histogram bin axis, and each chunk loops through all temporary histograms to accumulate into the
output histogram.

With the overhead associated with this algorithm, the implementation of each `parallel_for` may fallback to a serial
implementation. It makes sense to include this as part of a future improvement of `parallel_for`, where a user could
provide extra information in the call to influence details of the backend implementation from the non-background
specific implementation code.  Details which may be included could include grain size or a functor to determine fallback
to serial implementation.

### Temporary Memory Requirements
Both algorithms should have temporary memory complexity of `O(num_bins)`, and specifically will allocate `num_bins`
output histogram typed elements for each thread used. Depending on the number of input elements, all available threads
may not be used.

### Computational Complexity
#### Even Bin API
The proposed algorithm should have `O(N) + O(num_bins)` operations where `N` is the number of input elements, and
`num_bins` is the number of histogram bins.

#### Custom Range Bin API
The proposed algorithm should have `O(N * log(num_bins)) + O(num_bins)` operations where `N` is the number of input
elements, and `num_bins` is the number of histogram bins.
