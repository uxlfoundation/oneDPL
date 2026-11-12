# Kernel Configuration

## Introduction

`kernel_param` is a generic structure for configuring SYCL kernels.
Each kernel template is invoked with a `kernel_param` type or object.
Typically, a kernel template supports multiple values for the configuration parameters.
Optimal values may depend on the invoked kernel, the data size and type(s),
and underlying hardware characteristics.

## Signature

```c++
// defined in <oneapi/dpl/experimental/kernel_templates>
// in namespace oneapi::dpl::experimental::kt

template <std::uint16_t DataPerWorkItem,
          std::uint16_t WorkGroupSize,
          typename KernelName = /*unspecified*/>
struct kernel_param;
```

The library guide describes all the signature details in
https://github.com/uxlfoundation/oneDPL/blob/oneDPL-2022.8.0-release/documentation/library_guide/kernel_templates/kernel_configuration.rst.

## Open Questions

### Runtime Parameters

The `data_per_workitem` and `workgroup_size` parameters are currently provided at compile time,
but their optimal values may depend on the number of elements to process, which is known only at run time.

It should be investigated whether dynamic, run-time configurable parameters can be supported
without compromising efficiency.

### Dynamic Kernel Parameter Selection

A selector function can be provided
to avoid repeating the algorithm call for different kernel configurations.

```c++
namespace kt = oneapi::dpl::experimental::kt;

// Example of launching an algorithm with the current design
auto name = device.get_info<sycl::info::device::name>();
bool is_bmg = (name.find("B580") != std::string::npos);
bool is_nvidia = (name.find("NVIDIA") != std::string::npos);
if (is_bmg && N <= (1 << 18))
    kt::gpu::reduce(..., kt::kernel_param<3, 512>{});
else if (is_bmg && N <= (1 << 22))
    kt::gpu::reduce(..., kt::kernel_param<5, 512>{});
else if (is_nvidia)
    kt::gpu::reduce(..., kt::kernel_param<8, 256>{});
else
    kt::gpu::reduce(..., kt::kernel_param<7, 512>{});

// Example with a proposed dynamic selector
// The selector returns arguments needed to construct a kernel_param instance
auto selector = [](std::size_t N, sycl::device device) -> std::pair<std::uint16_t, std::uint16_t> {
    auto name = device.get_info<sycl::info::device::name>();
    bool is_bmg = (name.find("B580") != std::string::npos);
    bool is_nvidia = (name.find("NVIDIA") != std::string::npos);
    if (is_bmg && N <= (1 << 18))
        return {3, 512};
    else if (is_bmg && N <= (1 << 22))
        return {5, 512};
    else if (is_nvidia)
        return {8, 256};
    return {7, 512};
};
kt::gpu::reduce(..., kt::select_kernel_param(selector));
```

### Sub-group Size as a Parameter

For some devices configuring a sub-group size might be important for performance,
as it impacts factors such as register pressure and
the number of sub-groups to synchronize.
The optimal value depends on the device architecture and
the characteristics of the workload, for example, the type of elements being processed.

The performance benefit should be demonstrated in practice,
given the added complexity to the interface.

### Indexing Type

32-bit indices have been shown to improve the performance of algorithms with standard interfaces
from the binary search and merge families,
when the index range is sufficient to process the given input sequences.

It should be investigated whether this parameter should be exposed in Kernel Templates.
To determine where to introduce it, we need to assess how many algorithms would benefit:
- If the number is small,
  a more specialized kernel configuration structure that includes the parameter may be added.
- If the number is larger,
  the parameter may be added into `kernel_param`, either as an optional or mandatory member.

Another possible solution is to let the user specify the number of elements as a parameter,
and then deduce the index type from the type of this parameter. For example:

```c++
template <typename InputIt, typename SizeType, typename OutputIt, typename KernelParam>
sycl::event copy_n(InputIt input, SizeType n, OutputIt output, KernelParam param);
```

However, this method is not compatible with range-based inputs,
where the number of elements is determined implicitly from the range itself.
