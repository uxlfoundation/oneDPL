# Kernel Configuration

## Introduction

`kernel_param` is a generic structure for configuring SYCL kernels.
Each kernel template is invoked with a `kernel_param` type or object.
Typically, a kernel template supports multiple values for the configuration parameters.
Optimal values may depend on the invoked kernel, the data size and type(s),
as well as on the used device.

## Signature

```c++
// defined in <oneapi/dpl/experimental/kernel_templates>

namespace oneapi::dpl::experimental::kt {

template <std::uint16_t DataPerWorkItem,
          std::uint16_t WorkGroupSize,
          typename KernelName = /*unspecified*/>
struct kernel_param;

}
```

The library guide describes all the signature details in
https://github.com/uxlfoundation/oneDPL/blob/oneDPL-2022.8.0-release/documentation/library_guide/kernel_templates/kernel_configuration.rst.

## Open Questions

### Runtime Parameters

The `data_per_workitem` and `workgroup_size` parameters are currently provided at compile time,
but their optimal values depend on the number of elements to process, which is known only at run time.

It should be investigated whether these parameters can be made run-time configurable
without compromising efficiency.

.. `SYCL 2020 Specification`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:naming.kernels

### Sub-group Size as a Parameter

Choosing an appropriate sub-group size is important for performance,
as it impacts factors such as register pressure and
the number of sub-groups to synchronize.
The optimal value depends on the device architecture and
the characteristics of the workload, for example, the type of elements being processed.

The performance benefit should be demonstrated in practice,
given the added complexity to the interface.
