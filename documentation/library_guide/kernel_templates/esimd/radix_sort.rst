Radix Sort
##########

-----------------------------
radix_sort Function Templates
-----------------------------

The ``radix_sort`` function sorts data using the radix sort algorithm.
The sorting is stable, preserving the relative order of elements with equal keys.
Both in-place and out-of-place overloads are provided. Out-of-place overloads do not alter the input sequence.

The functions implement a Onesweep* [#fnote1]_ algorithm variant.

A synopsis of the ``radix_sort`` function is provided below:

.. code:: cpp

   // defined in <oneapi/dpl/experimental/kernel_templates>

   namespace oneapi::dpl::experimental::kt::gpu::esimd {

   // Sort in-place
   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Iterator>
   sycl::event
   radix_sort (sycl::queue q, Iterator first, Iterator last,
               KernelParam param); // (1)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Range>
   sycl::event
   radix_sort (sycl::queue q, Range&& r, KernelParam param); // (2)


   // Sort out-of-place
   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Iterator1,
             typename Iterator2>
   sycl::event
   radix_sort (sycl::queue q, Iterator1 first, Iterator1 last,
               Iterator2 first_out, KernelParam param); // (3)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Range1, typename Range2>
   sycl::event
   radix_sort (sycl::queue q, Range1&& r, Range2&& r_out,
               KernelParam param); // (4)
   }

.. note::
   The ``radix_sort`` is currently available only for Intel® Data Center GPU Max Series,
   and requires Intel® oneAPI DPC++/C++ Compiler 2023.2 or newer.

Template Parameters
--------------------

+-----------------------------+---------------------------------------------------------------------------------------+
| Name                        | Description                                                                           |
+=============================+=======================================================================================+
| ``bool IsAscending``        | The sort order. Ascending: ``true``; Descending: ``false``.                           |
+-----------------------------+---------------------------------------------------------------------------------------+
| ``std::uint8_t RadixBits``  | The number of bits to sort for each radix sort algorithm pass.                        |
+-----------------------------+---------------------------------------------------------------------------------------+


Parameters
----------

+-----------------------------------------------+---------------------------------------------------------------------+
| Name                                          | Description                                                         |
+===============================================+=====================================================================+
| ``q``                                         | The SYCL* queue where kernels are submitted.                        |
+-----------------------------------------------+---------------------------------------------------------------------+
|                                               |                                                                     |
|                                               | The sequences to apply the algorithm to.                            |
| - ``first``, ``last`` (1),                    | Supported sequence types:                                           |
| - ``r`` (2),                                  |                                                                     |
| - ``first``, ``last``, ``first_out`` (3),     | - :ref:`USM pointers <use-usm>` (1,3),                              |
| - ``r``, ``r_out`` (4).                       | - :ref:`oneapi::dpl::begin and oneapi::dpl::end                     |
|                                               |   <use-buffer-wrappers>` (1,3).                                     |
|                                               | - ``sycl::buffer`` (2,4),                                           |
|                                               | - :ref:`oneapi::dpl::experimental::ranges::views::all               |
|                                               |   <viewable-ranges>` (2,4),                                         |
|                                               | - :ref:`oneapi::dpl::experimental::ranges::views::subrange          |
|                                               |   <viewable-ranges>` (2,4).                                         |
|                                               |                                                                     |
|                                               |                                                                     |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+
| ``param``                                     | A :doc:`kernel_param <../kernel_configuration>` object.             |
|                                               | Its ``data_per_workitem`` must be a positive multiple of 32.        |
|                                               |                                                                     |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+


**Type Requirements**:

- The element type of sequence(s) to sort must be a C++ integral or floating-point type
  other than ``bool`` with a width of up to 64 bits.

.. note::

   Current limitations:

   - Number of elements to sort must not exceed `2^30`.
   - ``RadixBits`` can only be `8`.
   - ``param.workgroup_size`` can only be `64`.

Return Value
------------

A ``sycl::event`` object representing the status of the algorithm execution.

--------------
Usage Examples
--------------


In-Place Example
----------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl radix_sort.cpp -o radix_sort -I /path/to/oneDPL/include && ./radix_sort

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      std::uint32_t* keys = sycl::malloc_shared<std::uint32_t>(n, q);

      // initialize
      keys[0] = 3, keys[1] = 2, keys[2] = 1, keys[3] = 5, keys[4] = 3, keys[5] = 3;

      // sort
      auto e = kt::gpu::esimd::radix_sort<false, 8>(q, keys, keys + n, kt::kernel_param<416, 64>{}); // (1)
      e.wait();

      // print
      for(std::size_t i = 0; i < n; ++i)
         std::cout << keys[i] << ' ';
      std::cout << '\n';

      sycl::free(keys, q);
      return 0;
   }

**Output**::

   5 3 3 3 2 1



Out-of-Place Example
--------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl radix_sort.cpp -o radix_sort -I /path/to/oneDPL/include && ./radix_sort

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      std::uint32_t* keys = sycl::malloc_shared<std::uint32_t>(n, q);
      std::uint32_t* keys_out = sycl::malloc_shared<std::uint32_t>(n, q);

      // initialize
      keys[0] = 3, keys[1] = 2, keys[2] = 1, keys[3] = 5, keys[4] = 3, keys[5] = 3;

      // sort
      auto e = kt::gpu::esimd::radix_sort<false, 8>(q, keys, keys + n, keys_out, kt::kernel_param<416, 64>{}); // (3)
      e.wait();

      // print
      for(std::size_t i = 0; i < n; ++i)
         std::cout << keys[i] << ' ';
      std::cout << '\n';
      for(std::size_t i = 0; i < n; ++i)
         std::cout << keys_out[i] << ' ';
      std::cout << '\n';

      sycl::free(keys, q);
      sycl::free(keys_out, q);
      return 0;
   }

**Output**::

   3 2 1 5 3 3
   5 3 3 3 2 1


.. _radix-sort-memory-requirements:

-------------------
Memory Requirements
-------------------

The algorithm uses global and local device memory (see `SYCL 2020 Specification
<https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_device_memory_model>`_)
for intermediate data storage. For the algorithm to operate correctly, there must be enough memory on the device.
If there is not enough global device memory, a ``std::bad_alloc`` exception is thrown.
The behavior is undefined if there is not enough local memory.
The amount of memory that is required depends on input data and configuration parameters, as described below.

Global Memory Requirements
--------------------------

Global memory is used for copying the input sequence(s) and storing internal data such as radix value counters.
The used amount depends on many parameters; below is an upper bound approximation:

   N\ :sub:`keys` + C * N\ :sub:`keys`

where the sequence with keys takes N\ :sub:`keys` space, and the additional space is C * N\ :sub:`keys`.

The value of `C` depends on ``param.data_per_workitem``, ``param.workgroup_size``, and ``RadixBits``.
For ``param.data_per_workitem`` set to `32`, ``param.workgroup_size`` to `64`, and ``RadixBits`` to `8`,
`C` approximately equals to `1`.
Incrementing ``RadixBits`` increases `C` up to twice, while doubling either
``param.data_per_workitem`` or ``param.workgroup_size`` leads to a halving of `C`.

.. note::

   If the number of elements to sort does not exceed ``param.data_per_workitem * param.workgroup_size``,
   ``radix_sort`` is executed by a single work-group and does not use any global memory.

..
   The estimation above is not very precise and it seems it is not necessary for the global memory.
   The C coefficient base is actually 0.53 instead of 1.
   An increment of RadixBits multiplies C by the factor of ~1.5 on average.


Local Memory Requirements
-------------------------

Local memory is used for reordering keys within a work-group,
and for storing internal data such as radix value counters.
The used amount depends on many parameters; below is an upper bound approximation:

   N\ :sub:`keys_per_workgroup` + C

where N\ :sub:`keys_per_workgroup` is the amount of memory to store keys.
`C` is some additional space for storing internal data.

N\ :sub:`keys_per_workgroup` equals to ``sizeof(key_type) * param.data_per_workitem * param.workgroup_size``,
`C` does not exceed `4KB`.

..
   C as 4KB stands on these points:
   1) Extra space is needed to store a histogram to distribute keys. It's size is 4 * (2^RadixBits).
   The estimation is correct for RadixBits 9 (2KB) and smaller. Support of larger RadixBits is not expected.
   1) N_keys + N_values is rounded up at 2KB border (temporarily as a workaround for a GPU driver bug).

..
   The estimation assumes that reordering keys/pairs takes more space than ranking keys.
   The ranking takes approximatelly "2 * workgroup_size * (2^RadixBits)" bytes.
   It suprpasses Intel Data Center GPU Max SLM capacity in only marginal cases,
   e.g., when RadixBits is 10 and workgroup_size is 64, or when RadixBits is 9 and workgroup_size is 128.
   It is ignored as an unrealistic case.

-----------------------------------------
Recommended Settings for Best Performance
-----------------------------------------

The general advice is to choose kernel parameters based on performance measurements and profiling information.
The initial configuration may be selected according to these high-level guidelines:

..
   TODO: add this part when param.workgroup_size supports more than one value:
   Increasing ``param.data_per_workitem`` should usually be preferred to increasing ``param.workgroup_size``,
   to avoid extra synchronization overhead within a work-group.

- When the number of elements to sort (N) is small (~16K or less) and the algorithm is ``radix_sort``,
  generally sorting is done more efficiently by a single work-group.
  Increase the ``param`` values to make ``N <= param.data_per_workitem * param.workgroup_size``.

- When the number of elements to sort ``N`` is between 16K and 1M, utilizing all available
  compute cores is key for better performance. Allow creating enough work chunks to feed all
  X\ :sup:`e`-cores [#fnote2]_ on a GPU: ``param.data_per_workitem * param.workgroup_size ≈ N / xe_core_count``.

- When the number of elements to sort is large (more than ~1M), maximizing the number of elements
  processed by a work-group, which equals to ``param.data_per_workitem * param.workgroup_size``,
  reduces synchronization overheads between work-groups and usually benefits the overall performance.

.. warning::

   Avoid setting too large ``param.data_per_workitem`` and ``param.workgroup_size`` values.
   Make sure that :ref:`Memory requirements <radix-sort-memory-requirements>` are satisfied.

.. note::

   ``param.data_per_workitem`` is the only available parameter to tune the performance,
   since ``param.workgroup_size`` currently supports only one value (`64`).


.. [#fnote1] Andy Adinets and Duane Merrill (2022). Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. https://arxiv.org/abs/2206.01784.
.. [#fnote2] The X\ :sup:`e`-core term is described in the `oneAPI GPU Optimization Guide
   <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/intel-xe-gpu-architecture.html#XE-CORE>`_.
   Check the number of cores in the device specification, such as `Intel® Data Center GPU Max specification
   <https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series/products.html>`_.
