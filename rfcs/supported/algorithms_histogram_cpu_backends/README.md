# Host Backends Support for the Histogram APIs

## Introduction
The oneDPL library added histogram APIs in the oneAPI Specification 1.4. Please see the
[oneAPI Specification](https://github.com/uxlfoundation/oneAPI-spec/blob/main/source/elements/oneDPL/source/parallel_api/algorithms.rst#parallel-algorithms)
for details. This RFC outlines the now supported host-side backends (serial, TBB, OpenMP).

The commit containing the implementation proposed here can be found
[here](https://github.com/uxlfoundation/oneDPL/commit/780705c3bf3eee50b72a37cbc730de28d524efe9).

### Version Information
Histogram APIs are supported for host execution policies as of version 2022.8.0 of oneDPL, specification version 1.4.
```
ONEDPL_VERSION_MAJOR=2022
ONEDPL_VERSION_MINOR=8
ONEDPL_VERSION_PATCH=0

ONEDPL_SPEC_VERSION=104
```

Histogram APIs are supported for device execution policies as of version 2022.5.0. It was released before
`ONEDPL_SPEC_VERSION` was defined by the implementation.
```
ONEDPL_VERSION_MAJOR=2022
ONEDPL_VERSION_MINOR=5
ONEDPL_VERSION_PATCH=0
```

## Supported Implementation
After exploring several alternatives, we think the following implementation better represents the use cases which are
important, and provides reasonable performance for most cases.

### Embarrassingly Parallel Via Temporary Histograms
This method uses temporary storage and a pair of calls to backend specific `parallel_for` functions to accomplish the
`histogram`. These calls will use the existing infrastructure to provide properly composable parallelism, without extra
histogram-specific patterns in the implementation of a backend.

#### Enumerable Thread Local Storage feature
This algorithm requires that each parallel backend add a "make" function with the following signature:
```
template <typename _ValueType, typename... _Args>  
auto /* or unspecified enumerable thread local storage type */  
oneapi::dpl::__par_backend::__make_enumerable_tls(_Args&&...);  
```
returning an object which represents an enumerable thread local storage where `_ValueType` represents the type of stored
per-thread objects, and `_Args...` are the types of the arguments to send to the constructor of type `_ValueType` upon
each creation. The enumerable thread local storage object must provide the following member functions:

* `__ValueType& get_for_current_thread()`

 Returns reference to the current thread's stored object. This function is meant to be used within a parallel context,
 with each thread exclusively accessing its part of the storage.

* `__ValueType& get_with_id(std::size_t __i)` 

 Returns reference to the stored object at a given index to the list of already created elements, up to `size()`.
 This function must not be called concurrently with `get_for_current_thread()`, as that may create new
 elements in the storage.

* `std::size_t size() const`

 Returns the number of already created elements. Similar to `get_with_id()`, this must not be 
 called concurrently with `get_for_current_thread()`.

A unified implementation of `__enumerable_thread_local_storage_base` is provided with these features in
`include/oneapi/dpl/pstl/parallel_backend_utils.h` for use as a base implementation in a Curiously Recurring Template
Pattern (CRTP) for individual parallel backends to derive from. When using this base struct, derived classes
in specific parallel backends must provide the following methods.

* `static std::size_t get_num_threads()`

 Returns the maximal number of threads possible in the current parallel execution context.

* `static std::size_t get_thread_num()`

 Returns the index of the calling thread in the current parallel execution context.

When these two functions are provided by the derived structure, the base implementation provides the functionality
specified above. An example of this is found in both `include/oneapi/dpl/pstl/omp/util.h` and
`include/oneapi/dpl/pstl/parallel_backend_tbb.h`. The serial backend instead defines its own specific
`__enumerable_thread_local_storage` type in `include/oneapi/dpl/pstl/parallel_backend_serial.h`.

We decided to avoid using the `enumerable_thread_specific` class provided by TBB, because on Windows it includes
"windows.h".  The baggage associated with "windows.h" is not worth the benefit of using the existing type.

#### Algorithm
1) Call existing `parallel_for` pattern which performs a `histogram` on the input sequence, each thread accumulating
into its own temporary histogram provided by `__enumerable_thread_local_storage`. We rely upon `parallel_for` pattern to
implement chunksize and thread composability.
1) Call a second `parallel_for` pattern over the `histogram` output sequence which accumulates all temporary copies of
the histogram created within `__enumerable_thread_local_storage` into the output histogram sequence. Threads divide the
histogram bin axis, where each chunk loops through all temporary histograms to accumulate into the section of output
histogram. In this parallel section, the inner loop for each thread iterates over individual histogram bins within the
same temporary histogram copy, accumulating into the corresponding global output bins. This operates on chunks of
contiguous elements in memory, and is a better access pattern taking advantage of cachelines, vectorization, and memory
prefetching.

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

## Future work
* Improve `parallel_for` pattern to allow generic implementation level to specify chunksize and serial cutoff thresholds.
* Explore options for some partial vectorization of the first `parallel_for` loop
* Explore improvements to `__enumerable_thread_local_storage` API for enumerating through created elements. Consider
 switching to a visitor pattern, and better protecting against improper usage.

## Design Considerations

### Key Requirements
Provide support for the `histogram` APIs with the following policies and backends:
- Policies: `seq`, `unseq`, `par`, `par_unseq`
- Backends: `serial`, `tbb`, `openmp`

Users have a choice of execution policies when calling oneDPL APIs. They also have a number of options of backends which
they can select from when using oneDPL. It is important that all combinations of these options have support for the
`histogram` APIs.

### Performance
Histogram algorithms typically involve minimal computation and are likely to be memory-bound. So, the implementation
prioritizes reducing memory accesses and minimizing temporary memory traffic.

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
