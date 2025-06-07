# Kernel Templates

## Introduction

This RFC describes an experimental set of algorithms with easily tunable controls,
designed to achieve optimal performance on specific hardware and workloads.

They prioritize efficiency over generality when compared to
algorithms that conform to standard C++ interfaces and use device execution policies.

These algorithms are intended for use with SYCL and are expected to evolve into an extension.

This set of algorithms is in an early stage of development, with many questions yet to be addressed.

Algorithms which are already implemented are described in
https://uxlfoundation.github.io/oneDPL/kernel_templates_main.html

The terms "kernel template" and "algorithm" are used interchangeably in this RFC.

## Design

### High-Level Structure

The algorithms are defined in `<oneapi/dpl/experimental/kernel_templates>`, in `namespace oneapi::dpl::experimental::kt`.
This namespace contains portable algorithms and namespaces with more specialized implementations.

Below is an example of such a structure with possible algorithms:

```c++
// Namespace for portable kernels and namespaces with platform-specific implementations
namespace oneapi::dpl::experimental::kt
{

// The nested namespace for kernels optimized for GPU architectures
namespace gpu
{
  // Algorithms optimized for GPU
  /*return*/ reduce(/*args*/);
  /*return*/ transform(/*args*/);
  /*return*/ inclusive_scan(/*args*/);
  /*return*/ radix_sort(/*args*/);
  // ...

  // Algorithms optimized for Intel GPUs supporting ESIMD technology.
  namespace esimd
  {
    /*return*/ inclusive_scan(/*args*/);
    /*return*/ radix_sort(/*args*/);
    // ...
  }
}

// The nested namespace for kernels optimized for CPU architectures
namespace cpu { /*...*/ }

} // oneapi::dpl::experimental::kt
```

### Abstract Signature

A kernel template is a C++ function invoked from the host.

Each function takes a `sycl::queue` as its first argument and
an instance of the [kernel_param](kernel_configuration/README.md) template,
which applies settings common to all kernel templates, as its last argument.
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

    typename KernelParam,          // manually specified, may have a default in future

    typename Arg1,                 // deduced, may have a default
    typename Arg2,                 // ...
    typename ArgN                  // deduced, may have a default
>
sycl::event kernel_template (
    sycl::queue q,
    /* any cvref */ Arg1 arg1,
    /* any cvref */ Arg2 arg2,
    /* any cvref */ ArgN argn,
    KernelParam param
);
```

`Arg1`, ..., `ArgN` include the sequences to be processed.
Data can be passed in the same ways as described in the
[documentation on passing data]((https://uxlfoundation.github.io/oneDPL/parallel_api/pass_data_algorithms.html#pass-data-to-algorithms))
for algorithms with device policies.
Additionally, a plain `sycl::buffer` can also be used.
Specialized algorithms may impose additional restrictions on how data is passed.

If an algorithm allocates global memory, it must throw `std::bad_alloc` if the allocation is unsuccessful.

### Example

The algorithm in the example below is tuned for better performance using:
1. An algorithm-specific parameter: the number of bits sorted per radix sort pass (`8`).
2. A common set of parameters: the number of elements processed per sub-group (`416`)
  and the work-group size (`64`). [kernel_param](kernel_configuration/README.md)
  page has for more detailed description.

These parameters affect how many kernels are launched, how much register and local memory is used,
how much global memory accessed,
how well the hardware computational resources are utilized for a given the number of elements to sort,
and many other performance factors.
The parameters differ for each GPU and can be easily adjusted.
`oneapi::dpl::stable_sort`, an algorithm with a standard interface,
does not provide such tuning capabilities.

Also `oneapi::dpl::experimental::kt::gpu::esimd::radix_sort` kernel template relies on
ESIMD technology and certain forward progress guarantees between work-groups,
which allows using hardware resources more effectively, but it limits portability.

```c++
// icpx -fsycl radix_sort.cpp -o radix_sort -I /path/to/oneDPL/include && ./radix_sort
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

#include <oneapi/dpl/experimental/kernel_templates>

namespace kt = oneapi::dpl::experimental::kt;

int main()
{
   std::size_t n = 6;
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
 - Different element types.
 - Diverse types of input sequences.
 - Representative data distributions.
 - Various combinations of other parameters, depending on the signature of the algorithm.
 - Edge cases (for example, zero-sized sequences).

Due to the unlimited number of possible kernel parameter combinations, it is recommended to:
 - Always test the most representative configurations for a given algorithm from performance standpoint
 - Randomly select a subset of additional parameter combinations to broaden coverage.

The testing harness should generate or enable tests with limited portability on demand,
for example through cmake arguments.

## Open Questions

There are several design aspects which should be addressed to make it a fully-supported extension.

### Name

The name "Kernel Templates" may be misleading because
these entities more act like algorithms rather than SYCL kernels.
Probably, a better name is "Algorithm Templates".
The renaming requires changing the corresponding namespace.

### Specializations and their Differntiation

Currently, the specializations of the algorithms belong to different namespaces.
Using tags instead tags offers several advantages,
for example easier dispatching between specializations and avoiding deeply nested namespaces.

### Reporting Global and Local Memory Requirements

Currently, there is no an interface which returns amount of memory to be allocated
based on the algorithm, its configuration and input parameters.

Instead there is a description in the documentation which gives an estimation, see
https://uxlfoundation.github.io/oneDPL/kernel_templates/single_pass_scan.html#memory-requirements
as an example for `oneapi::dpl::experimental::gpu::inclusive_scan`.

The corresponding interfaces will make usage of the algorithms safer
by providing an easy way to get that information and fallback if necessary.

### External Allocation of Global Memory

It must be beneficial to allow passing externally allocated memory to an algorithm.
For example, a user program may manage a pool of memory, which can be reused.
It is especially useful if an algorithm is called in a loop and the allocation overhead
cannot be ammortized by memory pools provided by device or compiler runtime.

It may also make sense to make it the only option
and do not allocate global memory internally at all.
It must be done in conjuciton with
[Global and Local Memory Requirements](#global-and-local-memory-requirements)

### Asynchronous Execution

The algorithms return a `sycl::event` but do not accept any input events as inputs.
As a result, they are not fully asynchronous.

It should be investigated whether full asynchrony is desirable,
and if so,how to implement it correctly.

### Separation of Specializations Based on a Problem Size

Different approaches may be more effective when processing different number of elements.
For example, a small number of elements can be processed by a single work-group.

Separation of these cases may be beneficial to avoid extra compilation time.

### Algorithms to Implement

A list of algorithms to be implemented must be defined.

### Benchmarking

A benchmark suit can help to select
the best configuration and parameters for a given workload and hardware.
