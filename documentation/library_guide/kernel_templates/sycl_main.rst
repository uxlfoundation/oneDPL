SYCL-Based Kernel Templates
############################

The SYCL kernel templates are based on the SYCL 2020 programming model of |dpcpp_compiler|.
Kernel templates may impose restrictions on supported devices or adapters which are documented in their
respective sections.

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

