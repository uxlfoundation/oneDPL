SYCL-Based Kernel Templates
############################

The SYCL kernel templates are based on the SYCL 2020 programming model of |dpcpp_compiler|.
Kernel templates may impose restrictions on supported devices or adapters which are documented in their
respective sections.

Kernel Parameter Interpretation
================================

The definition of a work-item as it relates to :doc:`kernel_param <kernel_configuration>` parameters directly aligns
with the `SYCL specification <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_kernel_execution_model>`_.
With the |dpcpp_compiler|, a work-item corresponds to a SIMD lane on SIMD architecutes and a thread on SIMT
architectures.

.. note::

   If migrating from :doc:`ESIMD kernel templates <esimd_main>`, then
   dividing ESIMD ``data_per_workitem`` by 32 approximates the SYCL ``data_per_workitem``
   and multiplying ESIMD ``workgroup_size`` by 32 approximates the SYCL ``workgroup_size`` on Intel GPUs.
   However, configurations from one model may not be supported by the other, and optimal parameters should
   be determined through performance testing.

Available Templates
===================

The following templates are available in the ``oneapi::dpl::experimental::kt::gpu`` namespace,
with the implementation optimized for GPU devices:

* :doc:`radix_sort <sycl/radix_sort>`
* :doc:`radix_sort_by_key <sycl/radix_sort_by_key>`
* :doc:`single_pass_scan <sycl/single_pass_scan>`

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   sycl/radix_sort
   sycl/radix_sort_by_key
   sycl/single_pass_scan

