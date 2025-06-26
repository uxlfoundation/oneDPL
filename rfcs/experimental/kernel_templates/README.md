# Kernel Templates

## Introduction

This RFC describes an experimental set of algorithms with easily tunable controls,
designed to achieve optimal performance on specific hardware and workloads.
They prioritize efficiency over generality when compared to
algorithms that conform to the standard C++ interfaces.

These algorithms are intended for use with SYCL.

The terms "kernel template" and "algorithm" are used interchangeably in this RFC.

## Status

This set of algorithms is in an early stage of development,
with many design aspects yet to be addressed.
See the [Open Questions](#open-questions) list.

Algorithms which are already implemented are described in
https://uxlfoundation.github.io/oneDPL/kernel_templates_main.html.

## Proposal

### High-Level Structure

The algorithms are defined in `<oneapi/dpl/experimental/kernel_templates>`,
in `namespace oneapi::dpl::experimental::kt`.
This namespace contains portable algorithms and namespaces with more specialized implementations.

Below is an example of such a structure with possible algorithms:

```c++
namespace oneapi::dpl::experimental::kt
{
// Portable algorithms
/*return*/ reduce(/*args*/);
/*return*/ transform(/*args*/);
/*return*/ inclusive_scan(/*args*/);
/*return*/ radix_sort(/*args*/);
// ...

  namespace gpu
  {
    // Algorithms optimized for GPU architectures
    /*return*/ reduce(/*args*/);
    /*return*/ transform(/*args*/);
    // ...

    namespace esimd
    {
      // Algorithms optimized for Intel GPUs supporting ESIMD technology.
      /*return*/ reduce(/*args*/);
      /*return*/ transform(/*args*/);
      // ...
    }
  } // namespace oneapi::dpl::experimental::kt::gpu

  namespace cpu
  {
    // Algorithms optimized for CPU architectures
    // ...
  }

} // namespace oneapi::dpl::experimental::kt
```

There is no requirement for the sets of algorithms in each namespace to be identical.

### Abstract Algorithm Signature

A kernel template is a C++ function invoked from the host.

Each function takes a `sycl::queue` as its first argument and
an instance of the [kernel_param](kernel_configuration/README.md) template,
which specializes parameters common to all kernel templates, as its last argument.
The arguments in between vary depending on the specific algorithm.
The function may also have algorithm-specific non-deducible template parameters.

Below is an abstract signature of a kernel template:

```c++
// defined in <oneapi/dpl/experimental/kernel_templates>
// in namespace oneapi::dpl::experimental::kt

template <
    typename AlgorithmParameter1,  // manually specified, may have a default
    typename AlgorithmParameter2,  // ...
    typename AlgorithmParameterN,  // manually specified, may have a default

    typename KernelParam,          // deduced, may have a default in future

    typename Arg1,                 // deduced, may have a default
    typename Arg2,                 // ...
    typename ArgN                  // deduced, may have a default
>
sycl::event kernel_template (
    sycl::queue q,
    /* any cvref */ Arg1 arg1,
    /* any cvref */ Arg2 arg2,
    /* any cvref */ ArgN argn,
    KernelParam param              // manually specified, may have a default in future
);
```

`Arg1`, ..., `ArgN` include sequences to process.
Data must be passed using the same mechanisms described in the
[documentation on passing data](https://uxlfoundation.github.io/oneDPL/parallel_api/pass_data_algorithms.html#pass-data-to-algorithms)
for algorithms with device policies.
Specialized algorithms may extend or restrict the supported data passing mechanisms.

If an algorithm allocates global memory and that allocation is unsuccessful,
it must throw `std::bad_alloc`.
That behaviour may change depending on how
[External Allocation of Global Memory](#external-allocation-of-global-memory)
is addressed.

### Example

The example demonstrates the use of a kernel template
and describes how it is tuned for better performance and what makes it less general
than an alternative algorithm with a standard interface.
It uses `oneapi::dpl::experimental::kt::gpu::esimd::radix_sort`,
which can be compared to `oneapi::dpl::stable_sort`.

Performance tuning controls:
1. Algorithm-specific: the number of bits sorted per radix sort pass (`8`).
2. Common: the number of elements processed per work-item (`416`)
  and the number of work-items in a work-group (`64`).

These parameters influence various factors,
including the number of kernels launched, register and local memory usage,
global memory access, and the utilization of hardware computational resources.
They can be easily adjusted for another GPU with different hardware characteristics.

The kernel template relies on ESIMD technology
and certain forward progress guarantees between work-groups.

```c++
// icpx -fsycl radix_sort.cpp -o radix_sort -I /path/to/oneDPL/include && ./radix_sort
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

#include <oneapi/dpl/experimental/kernel_templates>

namespace kt = oneapi::dpl::experimental::kt;

int main()
{
   const std::size_t n = 6;
   sycl::queue q{sycl::gpu_selector_v};
   std::uint32_t* keys = sycl::malloc_shared<std::uint32_t>(n, q);

   keys[0] = 3, keys[1] = 2, keys[2] = 1, keys[3] = 5, keys[4] = 3, keys[5] = 3;

   auto e = kt::gpu::esimd::radix_sort<false, 8>(q, keys, keys + n, kt::kernel_param<416, 64>{});
   e.wait();

   sycl::free(keys, q);
   return 0;
}
```

## Testing

The testing harness should cover:
 - Varying numbers of elements to process, from thousands to millions.
 - Different element types, depending on the algorithm and its implementation details.
   For example, tests for radix should cover various arithmetic types
   due to differing ordered bit transformations.
 - All supported input sequence types.
   For example, if the algorithm supports
   `sycl::buffer`, `oneapi::dpl::begin`/`oneapi::dpl::end`, or USM pointers, each should be tested.
 - Various combinations of other parameters, depending on the signature of the algorithm.
 - Edge cases.
   For example, zero-sized sequences.

Due to the unlimited number of possible kernel parameter combinations, it is recommended to:
 - Always test the most representative configurations
  (most likely to be used and achieve the best performance).
 - Randomly select a subset of additional parameter combinations to broaden coverage.

The testing harness should generate or enable tests with limited portability on demand,
for example through cmake arguments.

## Open Questions

### Name

The name "Kernel Templates" may be misleading because
these entities more act like algorithms rather than SYCL kernels.
Renaming would require changing the top-level namespace.

### Specializations and their Differentiation

Currently, the specializations of the algorithms belong to different namespaces.
Using tags instead of namespaces offers several advantages,
for example easier dispatching between specializations and avoiding deeply nested namespaces.

### Reporting Global and Local Memory Requirements

Currently, there is no interface that returns the amount of memory to be allocated
based on the algorithm, its configuration, and input parameters.

Instead, the documentation provides only an estimation, for example see
https://uxlfoundation.github.io/oneDPL/kernel_templates/single_pass_scan.html#memory-requirements.

The corresponding interfaces would make algorithms safer
by providing an easy way to get memory requirements and apply fallbacks if needed.

### External Allocation of Global Memory

There are scenarios which benefit from passing pre-allocated temporary memory to an algorithm.
For example, a user program may manage a reusable memory pool.
Also, this is especially useful when an algorithm is called in a loop and the allocation overhead
cannot be amortized by memory pools provided by the device or compiler runtime.

It may be preferable to make this the only option and
avoid internal global memory allocations altogether.
This approach eliminates the overhead associated with retaining temporary memory
until an algorithm completes its asynchronous execution.
Additionally, it reduces the number of interfaces that need to be supported.

This should be done in conjunction with the
[Global and Local Memory Requirements](#reporting-global-and-local-memory-requirements) interface.

### Asynchronous Execution and Dependency Chaining

The algorithms return a `sycl::event` but do not accept any input events.
That limits how they can be embedded in asynchronous dependency chains.

It should be investigated whether passing input events is desirable,
and if so, how to implement it correctly.

### Separation of Specializations Based on a Problem Size

Different approaches may be more effective when processing different number of elements.
For example, a small number of elements can be processed by a single work-group.

Separating such cases would reduce compilation time.

### Algorithms to Implement

A list of algorithms to implement should be defined.

### Benchmarking

A benchmark suite can help select the best configuration and parameters
for a given workload and hardware.

### Default Kernel Configuration Values

`KernelParam` template parameter
is placed before all the deduced parameters to simplify
a potential addition of a default value in the future.
However, it is unclear if a default value should be provided,
especially given the focus on performance tuning.

Creation of a default value with optimal settings is impossible right now.
`kernel_param` template argument, which substitutes `KernelParam`,
includes compile-time values, whose optimal selection depends on run-time information,
such is the number of elements to process.

[Runtime Parameters](kernel_configuration/README.md#runtime-parameters) raises a question
of providing them at run-time.
However, doing so would likely require querying device properties, introducing overhead.

The complexity of manual selection of these values may be reduced by the [Benchmarks](#benchmarking),
making default values less relevant.

### Configuring an Algorithm with Multiple Kernels

It needs to be explored how to configure algorithms with multiple performance-critical kernels.
The `kernel_param` targets a single kernel, and currently, only one instance can be passed.

### Compiler Extension and Differentiation of Algorithms

Currently, it is unclear if we should defferentiate algorithms
which rely on compiler extensions.

## Exit Criteria

The proposed set of algorithms should become fully supported if:
- All the questions under [Kernel Templates](#open-questions) and
  [Kernel Configuration](kernel_configuration/README.md#open-questions) sections are either addressed
  or provided with a justification to postpone or ignore.
- A significant portion of the algorithms listed in
  [Algorithms to Implement](#algorithms-to-implement) are implemented.
- Evidence of sufficiently good performance is provided.
- There is positive adoption feedback.

Some individual algorithms may remain experimental
and have their own exit criteria.
