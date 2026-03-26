SYCL-Based Kernel Templates
############################

The SYCL kernel templates are based on the SYCL 2020 programming model of |dpcpp_compiler|.
This technology only supports Intel GPU devices.

These templates are available in the ``oneapi::dpl::experimental::kt::gpu`` namespace. The following are implemented:

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

