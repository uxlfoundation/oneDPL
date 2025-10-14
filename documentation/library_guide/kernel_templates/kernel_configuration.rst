Kernel Configuration
####################

-------------------------------
``kernel_param`` Class Template
-------------------------------

``kernel_param`` is a generic structure for configuring SYCL* kernels.
Each kernel template is invoked with a ``kernel_param`` type or object.
Typically, a kernel template supports multiple values for the configuration parameters.
Optimal values may depend on the invoked kernel, the data size and type(s), as well as on the used device.

A synopsis of the ``kernel_param`` struct is provided below:

.. code:: cpp

   // defined in <oneapi/dpl/experimental/kernel_templates>

   namespace oneapi::dpl::experimental::kt {

   template <std::uint16_t DataPerWorkItem,
             std::uint16_t WorkGroupSize,
             typename KernelName = /*unspecified*/>
   struct kernel_param;

   }


Static Member Constants
-----------------------

+------------------------------------------------------+---------------------+----------------------------------------+
| Name                                                 | Value               | Description                            |
+======================================================+=====================+========================================+
| ``static constexpr std::uint16_t data_per_workitem`` | ``DataPerWorkItem`` | The number of iterations to be         |
|                                                      |                     | processed by a work-item.              |
+------------------------------------------------------+---------------------+----------------------------------------+
| ``static constexpr std::uint16_t workgroup_size``    | ``WorkGroupSize``   | The number of work-items within        |
|                                                      |                     | a work-group.                          |
+------------------------------------------------------+---------------------+----------------------------------------+


.. note::

   The ``data_per_workitem`` parameter has a special meaning in :doc:`ESIMD-based kernel templates <esimd_main>`.
   Usually, each work-item processes ``data_per_workitem`` input elements sequentially.
   However, work-items in ESIMD-based kernel templates perform vectorization,
   so the sequential work is ``data_per_workitem / vector_length`` elements, where ``vector_length``
   is an implementation-defined vectorization factor.


Member Types
------------

+-----------------+----------------+----------------------------------------------------------------------------------+
| Type            | Definition     | Description                                                                      |
+=================+================+==================================================================================+
| ``kernel_name`` | ``KernelName`` | An optional parameter that is used to set a kernel name.                         |
|                 |                |                                                                                  |
|                 |                | .. note::                                                                        |
|                 |                |     The ``KernelName`` parameter might be required in case an implementation     |
|                 |                |     of SYCL is not fully compliant with the `SYCL 2020 Specification`_           |
|                 |                |     and does not support optional kernel names.                                  |
|                 |                |                                                                                  |
|                 |                | If omitted, SYCL kernel name(s) will be automatically generated.                 |
|                 |                |                                                                                  |
|                 |                | If provided, it must be a unique C++ typename that satisfies the requirements    |
|                 |                | for SYCL kernel names in the `SYCL 2020 Specification`_.                         |
|                 |                |                                                                                  |
|                 |                | .. note::                                                                        |
|                 |                |    The provided name can be augmented by oneDPL when used with                   |
|                 |                |    a template that creates multiple SYCL kernels.                                |
|                 |                |                                                                                  |
+-----------------+----------------+----------------------------------------------------------------------------------+

.. _`SYCL 2020 Specification`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:naming.kernels