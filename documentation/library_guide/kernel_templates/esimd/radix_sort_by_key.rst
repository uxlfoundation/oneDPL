Radix Sort By Key
#################

------------------------------------
radix_sort_by_key Function Templates
------------------------------------

The ``radix_sort_by_key`` function sorts keys using the radix sort algorithm, applying the same order to the corresponding values.
The sorting is stable, preserving the relative order of elements with equal keys.
Both in-place and out-of-place overloads are provided. Out-of-place overloads do not alter the input sequences.

The functions implement a Onesweep* [#fnote1]_ algorithm variant.

A synopsis of the ``radix_sort_by_key`` function is provided below:

.. code:: cpp

   // defined in <oneapi/dpl/experimental/kernel_templates>

   namespace oneapi::dpl::experimental::kt::gpu::esimd {

   // Sort in-place
   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Iterator1, typename Iterator2>
   sycl::event
   radix_sort_by_key (sycl::queue q, Iterator1 keys_first, Iterator1 keys_last,
                      Iterator2 values_first, KernelParam param); // (1)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename KeysRng, typename ValuesRng>
   sycl::event
   radix_sort_by_key (sycl::queue q, KeysRng&& keys,
                      ValuesRng&& values, KernelParam param); // (2)


   // Sort out-of-place
   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename KeysIterator1,
             typename ValuesIterator1, typename KeysIterator2,
             typename ValuesIterator2>
   sycl::event
   radix_sort_by_key (sycl::queue q, KeysIterator1 keys_first,
                      KeysIterator1 keys_last, ValuesIterator1 values_first,
                      KeysIterator2 keys_out_first, ValuesIterator2 values_out_first,
                      KernelParam param); // (3)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename KeysRng1, typename ValuesRng1,
             typename KeysRng2, typename ValuesRng2>
   sycl::event
   radix_sort_by_key (sycl::queue q, KeysRng1&& keys, ValuesRng1&& values,
                      KeysRng2&& keys_out, ValuesRng2&& values_out,
                      KernelParam param); // (4)
   }

.. note::
   The ``radix_sort_by_key`` is currently available only for Intel® Data Center GPU Max Series,
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
| - ``keys_first``, ``keys_last``,              | Supported sequence types:                                           |
|   ``values_first`` (1),                       |                                                                     |
| - ``keys``, ``values`` (2),                   | - :ref:`USM pointers <use-usm>` (1,3),                              |
| - ``keys_first``, ``keys_last``,              | - :ref:`oneapi::dpl::begin and oneapi::dpl::end                     |
|   ``values_first``, ``keys_out_first``,       |   <use-buffer-wrappers>` (1,3).                                     |
|   ``values_out_first`` (3)                    | - ``sycl::buffer`` (2,4),                                           |
| - ``keys``, ``values``,                       | - :ref:`oneapi::dpl::experimental::ranges::views::all               |
|   ``keys_out``, ``values_out`` (4).           |   <viewable-ranges>` (2,4),                                         |
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
   //    icpx -fsycl radix_sort_by_key.cpp -o radix_sort_by_key -I /path/to/oneDPL/include && ./radix_sort_by_key

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      sycl::buffer<std::uint32_t> keys{sycl::range<1>(n)};
      sycl::buffer<char> values{sycl::range<1>(n)};

      // initialize
      {
         sycl::host_accessor k_acc{keys, sycl::write_only};
         k_acc[0] = 3, k_acc[1] = 2, k_acc[2] = 1, k_acc[3] = 5, k_acc[4] = 3, k_acc[5] = 3;

         sycl::host_accessor v_acc{values, sycl::write_only};
         v_acc[0] = 'r', v_acc[1] = 'o', v_acc[2] = 's', v_acc[3] = 'd', v_acc[4] = 't', v_acc[5] = 'e';
      }

      // sort
      auto e = kt::gpu::esimd::radix_sort_by_key<true, 8>(q, keys, values, kt::kernel_param<96, 64>{}); // (2)
      e.wait();

      // print
      {
         sycl::host_accessor k_acc{keys, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << k_acc[i] << ' ';
         std::cout << '\n';

         sycl::host_accessor v_acc{values, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << v_acc[i] << ' ';
         std::cout << '\n';
      }

      return 0;
   }

**Output**::

   1 2 3 3 3 5
   s o r t e d

Out-of-Place Example
--------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl radix_sort_by_key.cpp -o radix_sort_by_key -I /path/to/oneDPL/include && ./radix_sort_by_key

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      sycl::buffer<std::uint32_t> keys{sycl::range<1>(n)};
      sycl::buffer<std::uint32_t> keys_out{sycl::range<1>(n)};
      sycl::buffer<char> values{sycl::range<1>(n)};
      sycl::buffer<char> values_out{sycl::range<1>(n)};


      // initialize
      {
         sycl::host_accessor k_acc{keys, sycl::write_only};
         k_acc[0] = 3, k_acc[1] = 2, k_acc[2] = 1, k_acc[3] = 5, k_acc[4] = 3, k_acc[5] = 3;

         sycl::host_accessor v_acc{values, sycl::write_only};
         v_acc[0] = 'r', v_acc[1] = 'o', v_acc[2] = 's', v_acc[3] = 'd', v_acc[4] = 't', v_acc[5] = 'e';
      }

      // sort
      auto e = kt::gpu::esimd::radix_sort_by_key<true, 8>(q, keys, values, keys_out, values_out,
                                                          kt::kernel_param<96, 64>{}); // (4)
      e.wait();

      // print
      {
         sycl::host_accessor k_acc{keys, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << k_acc[i] << ' ';
         std::cout << '\n';

         sycl::host_accessor v_acc{values, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << v_acc[i] << ' ';
         std::cout << "\n\n";
         
         sycl::host_accessor k_out_acc{keys_out, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << k_out_acc[i] << ' ';
         std::cout << '\n';

         sycl::host_accessor v_out_acc{values_out, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << v_out_acc[i] << ' ';
         std::cout << '\n';
      }

      return 0;
   }

**Output**::

   3 2 1 5 3 3
   r o s d t e

   1 2 3 3 3 5
   s o r t e d


.. _radix-sort-by-key-memory-requirements:

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

   N\ :sub:`keys` + N\ :sub:`values` + C * N\ :sub:`keys`

where the sequence with keys takes N\ :sub:`keys` space, the sequence with values takes N\ :sub:`values` space,
and the additional space is C * N\ :sub:`keys`.

The value of `C` depends on ``param.data_per_workitem``, ``param.workgroup_size``, and ``RadixBits``.
For ``param.data_per_workitem`` set to `32`, ``param.workgroup_size`` to `64`, and ``RadixBits`` to `8`,
`C` approximately equals to `1`.
Incrementing ``RadixBits`` increases `C` up to twice, while doubling either
``param.data_per_workitem`` or ``param.workgroup_size`` leads to a halving of `C`.

..
   The estimation above is not very precise and it seems it is not necessary for the global memory.
   The C coefficient base is actually 0.53 instead of 1.
   An increment of RadixBits multiplies C by the factor of ~1.5 on average.

   Additionally, C exceeds 1 for radix_sort_by_key,
   when N is small and the global histogram takes more space than the sequences.
   This space is small, single WG implementation will be added, therefore this is neglected.

Local Memory Requirements
-------------------------

Local memory is used for reordering key-value pairs within a work-group,
and for storing internal data such as radix value counters.
The used amount depends on many parameters; below is an upper bound approximation:

   N\ :sub:`keys_per_workgroup` + N\ :sub:`values_per_workgroup` + C

where N\ :sub:`keys_per_workgroup` and N\ :sub:`values_per_workgroup` are the amounts of memory
to store keys and values, respectively. `C` is some additional space for storing internal data.

N\ :sub:`keys_per_workgroup` equals to ``sizeof(key_type) * param.data_per_workitem * param.workgroup_size``,
N\ :sub:`values_per_workgroup` equals to ``sizeof(value_type) * param.data_per_workitem * param.workgroup_size``,
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

- When the number of elements to sort ``N`` is less than 1M, utilizing all available
  compute cores is key for better performance. Allow creating enough work chunks to feed all
  X\ :sup:`e`-cores [#fnote2]_ on a GPU: ``param.data_per_workitem * param.workgroup_size ≈ N / xe_core_count``.

- When the number of elements to sort is large (more than ~1M), maximizing the number of elements
  processed by a work-group, which equals to ``param.data_per_workitem * param.workgroup_size``,
  reduces synchronization overheads between work-groups and usually benefits the overall performance.

.. warning::

   Avoid setting too large ``param.data_per_workitem`` and ``param.workgroup_size`` values.
   Make sure that :ref:`Memory requirements <radix-sort-by-key-memory-requirements>` are satisfied.

.. note::

   ``param.data_per_workitem`` is the only available parameter to tune the performance,
   since ``param.workgroup_size`` currently supports only one value (`64`).


.. [#fnote1] Andy Adinets and Duane Merrill (2022). Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. https://arxiv.org/abs/2206.01784.
.. [#fnote2] The X\ :sup:`e`-core term is described in the `oneAPI GPU Optimization Guide
   <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/intel-xe-gpu-architecture.html#XE-CORE>`_.
   Check the number of cores in the device specification, such as `Intel® Data Center GPU Max specification
   <https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series/products.html>`_.
