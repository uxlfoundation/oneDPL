# Kernel Configuration

## Introduction

`kernel_param` is a generic structure for configuring SYCL kernels.
Each kernel template is invoked with a `kernel_param` type or object.
Typically, a kernel template supports multiple values for the configuration parameters.
Optimal values may depend on the invoked kernel, the data size and type(s),
as well as on the used device.

## Signature

A synopsis of the `kernel_param` struct is provided below:

```c++
// defined in <oneapi/dpl/experimental/kernel_templates>

namespace oneapi::dpl::experimental::kt {

template <std::uint16_t DataPerWorkItem,
          std::uint16_t WorkGroupSize,
          typename KernelName = /*unspecified*/>
struct kernel_param;

}
```

### Static Member Constants

| Name                                               | Value             |
|----------------------------------------------------|-------------------|
| `static constexpr std::uint16_t data_per_workitem` | `DataPerWorkItem` |
| `static constexpr std::uint16_t workgroup_size`    | `WorkGroupSize`   |

`data_per_workitem` is the number of iterations to be processed by a work-item.
`workgroup_size` is he number of work-items within a work-group.

The ``data_per_workitem`` parameter has a special meaning in ESIMD-based kernel templates.
Usually, each work-item processes ``data_per_workitem`` input elements sequentially.
However, work-items in ESIMD-based kernel templates perform vectorization,
so the sequential work is ``data_per_workitem / vector_length`` elements, where ``vector_length``
is an implementation-defined vectorization factor.

### Member Types

| Type          | Definition   |
|---------------|--------------|
| `kernel_name` | `KernelName` |

`kernel_name` is an optional parameter that is used to set a kernel name.
If omitted, SYCL kernel name(s) will be automatically generated.
If provided, it must be a unique C++ typename that satisfies the requirements
for SYCL kernel names in the
[SYCL 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:naming.kernels)

Passing `kernel_name` might be required in case an implementation of SYCL
is not fully compliant with the
[SYCL 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:naming.kernels)
and does not support optional kernel names.

The provided name can be augmented by oneDPL
when used with a template that creates multiple SYCL kernels.

## Open Questions

### Runtime Parameters

The `data_per_workitem` and `workgroup_size` parameters are currently provided at compile time,
but their optimal values depend on the number of elements to process, which is known only at run time.

It should be investigated whether these parameters can be made run-time configurable
without compromising efficiency.
