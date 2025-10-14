Kernel Templates API
####################

Kernel templates is an experimental API that provides algorithms built on top of SYCL* kernels and customizable with various parameters,
such as the number of elements to be processed by a work-item and the size of a work-group.
These algorithms aim to deliver optimal performance. However, they may depend on assumptions
that are not satisfied by all devices, prioritizing efficiency over generality.

It is recommended to use kernel templates when there is an opportunity to customize an algorithm
for a particular workload (for example, the number of elements and their type),
or for a specific device (for example, based on the available local memory).

To use the API, include the ``<oneapi/dpl/experimental/kernel_templates>`` header file.
The primary API namespace is ``oneapi::dpl::experimental::kt``, and nested namespaces are used to further categorize the templates.

* :doc:`Kernel Configuration <kernel_templates/kernel_configuration>`. Generic structure for configuring a kernel template.
* :doc:`ESIMD-based kernel templates <kernel_templates/esimd_main>`. Algorithms implemented with the Explicit SIMD SYCL extension.
* :doc:`Inclusive scan algorithm <kernel_templates/single_pass_scan>`. Inclusive scan kernel template algorithm using a single-pass approach.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   kernel_templates/kernel_configuration
   kernel_templates/esimd_main
   kernel_templates/single_pass_scan
