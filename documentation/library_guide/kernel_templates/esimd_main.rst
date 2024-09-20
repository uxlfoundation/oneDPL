ESIMD-Based Kernel Templates
############################

The ESIMD kernel templates are based on |esimd_sycl_extension|_ of |dpcpp_cpp|.
This technology only supports Intel GPU devices.

These templates are available in the ``oneapi::dpl::experimental::kt::gpu::esimd`` namespace. The following are implemented:

* :doc:`radix_sort <esimd/radix_sort>`
* :doc:`radix_sort_by_key <esimd/radix_sort_by_key>`

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   esimd/radix_sort
   esimd/radix_sort_by_key

-------------------
System Requirements
-------------------

- Hardware: Intel® Data Center GPU Max Series.
- Compiler: Intel® oneAPI DPC++/C++ Compiler 2023.2 and newer.
- Operating Systems:

  - Red Hat Enterprise Linux* 9.2,
  - SUSE Linux Enterprise Server* 15 SP5,
  - Ubuntu* 22.04.
