# Deprecate FPGA support

The proposal is implemented in oneDPL 2022.12.

## Introduction

FPGA support in oneDPL dates back to beta versions. At the time, Intel had
an FPGA business and tried to make programming FPGAs easier with DPC++.
oneDPL attempted to build on that and allow the use of standard C++ algorithms
for FPGAs.

Because SYCL kernels for FPGA should be written very differently from those for GPU and CPU
even for simple algorithms such as `for_each`, oneDPL introduced a separate execution policy type,
which allows compile-time dispatch of an algorithm to an FPGA-tailored implementation,
However, that implementation was only developed for the simplest pattern, `parallel_for`
(see https://github.com/uxlfoundation/oneDPL/blob/main/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_fpga.h).

Due to unclear perspectives and lack of practical feedback, the FPGA execution policy
was never added to the oneDPL specification.

Since then, Intel has spun off its FPGA business; support for FPGAs in the oneAPI DPC++ compiler
has been removed about a year ago. There is no evidence of use, nor of interest to use oneDPL to program FPGAs.

## Proposal

We propose to deprecate FPGA support in oneDPL algorithms now, and remove it a year after.

The proposal affects the
[FPGA execution policies](https://uxlfoundation.github.io/oneDPL/parallel_api/execution_policies.html#use-the-fpga-policy)
`fpga_policy` and `dpcpp_fpga`, the `make_fpga_policy` function, and the related feature enabling macros `ONEDPL_FPGA_*`.

The use of the deprecated classes and functions should result in a compile-time warning about API deprecation.
The oneDPL documentation should add deprecation notices for the affected APIs.

## Open Questions

An open question is if the FPGA-tailored implementation of `parallel_for` is worth preserving
once FPGA support is removed. One possible option is to keep it as an experimental kernel template.

## Next Steps

The FPGA support in oneDPL will likely be fully removed in the first half of 2027.
As a prerequisite, the open questions above should be addressed.
After the removal, this document should move to `rfcs/archived`.
