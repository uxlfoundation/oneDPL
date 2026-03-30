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


Parameter Semantics
-------------------

The meaning of ``data_per_workitem`` and ``workgroup_size`` differs significantly between
:doc:`ESIMD-based <esimd_main>` and SYCL-based kernel templates due to their underlying execution models:

**ESIMD (Explicit SIMD) Kernel Templates:**

- ``data_per_workitem``: The number of data elements processed by a single **hardware thread**,
  which issues explicit SIMD (vector) operations. The hardware thread processes these elements
  using SIMD instructions with an implementation-defined vector length.

- ``workgroup_size``: The number of **hardware threads** in a work-group. Each hardware thread
  executes SIMD operations independently.

- **Total parallelism**: ``workgroup_size`` hardware threads, each processing ``data_per_workitem``
  elements via explicit SIMD operations.

**SYCL Kernel Templates:**

- ``data_per_workitem``: The number of data elements processed sequentially by a single **SIMD lane**
  (work-item in SYCL terminology). Each SIMD lane executes scalar operations on its assigned elements.

- ``workgroup_size``: The number of **SIMD lanes** (work-items) in a work-group. SIMD lanes within
  a sub-group execute in lockstep on a single hardware thread.

- **Total parallelism**: ``workgroup_size`` SIMD lanes, each processing ``data_per_workitem``
  elements sequentially.

.. note::

   In summary, for the same configuration values, ESIMD kernels utilize explicit vectorization
   within each hardware thread, while SYCL kernels rely on implicit vectorization across work-items
   that are grouped into sub-groups by the compiler and runtime.


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