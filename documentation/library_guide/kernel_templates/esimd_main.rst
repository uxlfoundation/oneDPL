ESIMD-Based Kernel Templates
############################

The ESIMD kernel templates are based on |esimd_sycl_extension|_ of |dpcpp_compiler|.
This technology only supports Intel GPU devices.

Kernel Parameter Interpretation
================================

The :doc:`kernel_param <kernel_configuration>` parameters map to Intel GPU hardware as follows:

- ``data_per_workitem``: The number of data elements processed by a single hardware thread
  (work-item in ESIMD terminology), which issues explicit vector operations. The hardware thread
  processes these elements using SIMD instructions with an implementation-defined vector length.

- ``workgroup_size``: The number of hardware threads in a work-group. Each hardware thread
  executes scalar and SIMD operations independently.

Available Templates
===================

These templates are available in the ``oneapi::dpl::experimental::kt::gpu::esimd`` namespace. The following are implemented:

* :doc:`radix_sort <esimd/radix_sort>`
* :doc:`radix_sort_by_key <esimd/radix_sort_by_key>`

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   esimd/radix_sort
   esimd/radix_sort_by_key

