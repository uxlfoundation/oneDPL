Batched Stable Sort
###################

--------------------------------------
batched_stable_sort Function Templates
--------------------------------------

The ``batched_stable_sort`` function sorts a batch of equally sized, contiguous segments, each independently of the
others, in a single call.
The sorting is stable, preserving the relative order of equal elements within a segment.
Both in-place and out-of-place overloads are provided. Out-of-place overloads do not alter the input sequence.

The algorithm is selected at compile time with a sort configuration type:

- ``merge_single_group_tag``: a work-group merge path sort. Each segment is sorted by a single work-group,
  and multiple segments may be packed into one work-group.
- ``radix_single_group_tag<RadixBits>``: a one work-group radix sort. Each segment is sorted by a single work-group,
  and multiple segments may be packed into one work-group.
- ``radix_many_groups_tag<RadixBits>``: a segmented Onesweep* [#fnote1]_ radix sort. Each segment may be sorted by
  multiple work-groups.

Every configuration orders keys in the same way, so the output does not depend on the configuration, which may be
chosen for performance alone (see :ref:`Selecting a Sort Configuration <batched-sort-selecting-config>`).

.. note::
   ``batched_stable_sort`` is currently available for Intel® Arc™ B-Series and Intel® Data Center GPU Max Series.
   ``radix_many_groups_tag`` has the same requirements as :doc:`radix_sort <radix_sort>`: the Intel® oneAPI
   DPC++/C++ Compiler 2025.1.0 or greater is required, and the Unified Runtime adapter over Level Zero must be used
   (which is the default adapter for Intel GPUs). For more information, please refer to |dpcpp_device_selection|_.
   ``radix_single_group_tag`` requires a compiler with ``SYCL_EXT_ONEAPI_SUB_GROUP_MASK`` support, and a device
   which supports a sub-group size of `32`.

A synopsis of the ``batched_stable_sort`` function is provided below:

.. code:: cpp

   // defined in <oneapi/dpl/experimental/kernel_templates>

   namespace oneapi::dpl::experimental::kt::gpu {

   // Sort configurations
   struct merge_single_group_tag;

   template <std::uint8_t RadixBits = 8>
   struct radix_single_group_tag;

   template <std::uint8_t RadixBits = 8>
   struct radix_many_groups_tag;

   // Sort in-place
   template <typename SortConfig, bool IsAscending = true,
             typename KernelParam, typename Iterator>
   sycl::event
   batched_stable_sort (sycl::queue q, Iterator first, Iterator last,
                        typename std::iterator_traits<Iterator>::difference_type segment_size,
                        KernelParam param = {}); // (1)

   template <typename SortConfig, bool IsAscending = true,
             typename KernelParam, typename Range>
   sycl::event
   batched_stable_sort (sycl::queue q, Range&& r,
                        /*difference type of Range*/ segment_size,
                        KernelParam param = {}); // (2)

   // Sort out-of-place
   template <typename SortConfig, bool IsAscending = true,
             typename KernelParam, typename Iterator1, typename Iterator2>
   sycl::event
   batched_stable_sort (sycl::queue q, Iterator1 first, Iterator1 last, Iterator2 first_out,
                        typename std::iterator_traits<Iterator1>::difference_type segment_size,
                        KernelParam param = {}); // (3)

   template <typename SortConfig, bool IsAscending = true,
             typename KernelParam, typename Range1, typename Range2>
   sycl::event
   batched_stable_sort (sycl::queue q, Range1&& r, Range2&& r_out,
                        /*difference type of Range1*/ segment_size,
                        KernelParam param = {}); // (4)

   // Each of (1)-(4) has a counterpart which accepts caller-provided temporary storage
   // directly before param, for example:
   template <typename SortConfig, bool IsAscending = true,
             typename KernelParam, typename Iterator>
   sycl::event
   batched_stable_sort (sycl::queue q, Iterator first, Iterator last,
                        typename std::iterator_traits<Iterator>::difference_type segment_size,
                        void* temp_storage, std::size_t temp_storage_bytes,
                        KernelParam param = {}); // (1t)

   // Temporary storage requirements, in bytes
   template <typename SortConfig, typename KeyT, typename KernelParam>
   constexpr std::size_t
   batched_stable_sort_bytes_req (std::size_t segment_size, std::size_t total_size,
                                  KernelParam param = {}); // (5)
   }

Template Parameters
-------------------

+-----------------------------+---------------------------------------------------------------------------------------+
| Name                        | Description                                                                           |
+=============================+=======================================================================================+
| ``typename SortConfig``     | The sort configuration: ``merge_single_group_tag``, ``radix_single_group_tag``, or    |
|                             | ``radix_many_groups_tag``. It has no default value and must always be specified.      |
+-----------------------------+---------------------------------------------------------------------------------------+
| ``bool IsAscending``        | The sort order. Ascending: ``true``; Descending: ``false``.                           |
+-----------------------------+---------------------------------------------------------------------------------------+
| ``std::uint8_t RadixBits``  | The number of bits to sort for each radix sort algorithm pass                         |
|                             | (``radix_single_group_tag`` and ``radix_many_groups_tag`` only).                      |
+-----------------------------+---------------------------------------------------------------------------------------+
| ``typename KeyT``           | The element type of the sort to query the temporary storage requirements of (5).      |
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
+-----------------------------------------------+---------------------------------------------------------------------+
| ``segment_size``                              | The number of elements in each segment. The input sequence holds    |
|                                               | ``n / segment_size`` consecutive segments, where ``n`` is the       |
|                                               | number of elements.                                                 |
+-----------------------------------------------+---------------------------------------------------------------------+
| ``temp_storage``, ``temp_storage_bytes``      | Caller-provided temporary storage and its size in bytes.            |
|                                               | See :ref:`Temporary Storage <batched-sort-temporary-storage>`.      |
+-----------------------------------------------+---------------------------------------------------------------------+
| ``total_size``                                | The total number of elements to sort, ``n`` (5).                    |
+-----------------------------------------------+---------------------------------------------------------------------+
| ``param``                                     | A :doc:`kernel_param <../kernel_configuration>` object.             |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+


**Type Requirements**:

- The element type of sequence(s) to sort must be one of the following:

   - A C++ integral or floating-point type other than ``bool`` with a width of up to 64 bits.
   - ``sycl::half``.
   - ``sycl::ext::oneapi::bfloat16`` (requires Intel® oneAPI DPC++/C++ Compiler 2026.0 or greater).

**Ordering**: elements are ordered as by :doc:`radix_sort <radix_sort>`. For floating-point elements, ``-0.0`` and
``+0.0`` are equal, so their relative order is preserved, and NaNs are ordered by their sign and payload, after
``+infinity`` for positive NaNs and before ``-infinity`` for negative NaNs in ascending order.

**Preconditions**: the behavior is undefined if any of the following are violated. In debug builds, the violations
are reported with an assertion.

- ``segment_size > 0``, and ``n`` is a multiple of ``segment_size``. If ``n`` is `0`, the call has no effect.
- For ``merge_single_group_tag`` and ``radix_single_group_tag``,
  ``segment_size <= param.data_per_workitem * param.workgroup_size``.
- The output sequence holds at least ``n`` elements.
- The input and output sequences of the out-of-place overloads do not overlap.

.. note::

   Current limitations:

   - Number of elements to sort must not exceed 2\ :sup:`30`.
   - ``RadixBits`` can only be `8`.
   - ``param.data_per_workitem * param.workgroup_size`` must not exceed `65535`
     for ``radix_single_group_tag`` and ``radix_many_groups_tag``.
   - ``param.workgroup_size`` can be `512` or `1024` for ``radix_many_groups_tag``,
     and a multiple of `32` no greater than `1024` for ``radix_single_group_tag``.

Return Value
------------

(1)-(4): a ``sycl::event`` object representing the status of the algorithm execution.

(5): the number of bytes of temporary storage required for ``total_size`` elements in segments of
``segment_size`` elements. It is `0` for ``merge_single_group_tag`` and ``radix_single_group_tag``.

--------------
Usage Examples
--------------


In-Place Example with Temporary Storage
---------------------------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl batched_stable_sort.cpp -o batched_stable_sort -I /path/to/oneDPL/include && ./batched_stable_sort

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      std::ptrdiff_t segment_size = 3;
      sycl::queue q{sycl::gpu_selector_v};
      std::uint32_t* keys = sycl::malloc_shared<std::uint32_t>(n, q);

      using config_t = kt::gpu::radix_many_groups_tag<8>;
      using param_t = kt::kernel_param<8, 512>;

      // allocate temporary storage once; it may be reused by sorts with the same requirements
      std::size_t temp_bytes = kt::gpu::batched_stable_sort_bytes_req<config_t, std::uint32_t>(segment_size, n,
                                                                                               param_t{}); // (5)
      void* temp = sycl::malloc_device(temp_bytes, q);

      // initialize: two segments of three elements
      keys[0] = 3, keys[1] = 2, keys[2] = 1, keys[3] = 5, keys[4] = 3, keys[5] = 3;

      // sort each segment in ascending order
      auto e = kt::gpu::batched_stable_sort<config_t>(q, keys, keys + n, segment_size, temp, temp_bytes,
                                                      param_t{}); // (1t)
      e.wait();

      // print
      for(std::size_t i = 0; i < n; ++i)
         std::cout << keys[i] << ' ';
      std::cout << '\n';

      sycl::free(temp, q);
      sycl::free(keys, q);
      return 0;
   }

**Output**::

   1 2 3 3 3 5


Out-of-Place Example
--------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl batched_stable_sort.cpp -o batched_stable_sort -I /path/to/oneDPL/include && ./batched_stable_sort

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      std::ptrdiff_t segment_size = 3;
      sycl::queue q{sycl::gpu_selector_v};
      std::uint32_t* keys = sycl::malloc_shared<std::uint32_t>(n, q);
      std::uint32_t* keys_out = sycl::malloc_shared<std::uint32_t>(n, q);

      // initialize: two segments of three elements
      keys[0] = 1, keys[1] = 3, keys[2] = 2, keys[3] = 3, keys[4] = 5, keys[5] = 3;

      // sort each segment in descending order
      using config_t = kt::gpu::merge_single_group_tag;
      auto e = kt::gpu::batched_stable_sort<config_t, false>(q, keys, keys + n, keys_out, segment_size,
                                                             kt::kernel_param<4, 64>{}); // (3)
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

   1 3 2 3 5 3
   3 2 1 5 3 3


.. _batched-sort-memory-requirements:

-------------------
Memory Requirements
-------------------

The algorithms use global and local device memory (see `SYCL 2020 Specification
<https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_device_memory_model>`_)
for intermediate data storage. For the algorithm to operate correctly, there must be enough memory on the device.
If there is not enough global device memory for the internally allocated temporary storage,
a ``std::bad_alloc`` exception is thrown. The behavior is undefined if there is not enough local memory.

Global Memory Requirements
--------------------------

``merge_single_group_tag`` and ``radix_single_group_tag`` use no temporary global memory.

``radix_many_groups_tag`` uses global memory for copying the input sequence and for radix value counters,
which are kept for each segment and for each work-group tile of a segment:

   N\ :sub:`keys` + 4 * 2\ :sup:`RadixBits` * S * (G + T)

where the sequence with keys takes N\ :sub:`keys` space,
`S` is the number of radix sort passes, ``ceil(sizeof(key_type) * 8 / RadixBits)``,
`G` is the number of segments, ``n / segment_size``,
and `T` is the total number of tiles, ``G * ceil(segment_size / (param.data_per_workitem * param.workgroup_size))``.
For example, with ``std::uint32_t`` keys and ``RadixBits`` set to `8`, each segment and each tile take `4KB`.
``batched_stable_sort_bytes_req`` returns the exact amount.

.. warning::

   ``radix_many_groups_tag`` is not recommended for small segments. The counters are kept for every segment
   regardless of its length, so as segments get shorter, they take more memory and time than the keys.
   Use ``merge_single_group_tag`` or ``radix_single_group_tag`` for segments which fit into a work-group.

Local Memory Requirements
-------------------------

All configurations use local memory to hold a work-group tile of keys:

   N\ :sub:`keys_per_workgroup` + C

where N\ :sub:`keys_per_workgroup` is equal to ``sizeof(key_type) * param.data_per_workitem * param.workgroup_size``.

- For ``merge_single_group_tag``, `C` is `0`. Only the segments which are packed into a work-group are held,
  so the requirement is lower when ``param.data_per_workitem * param.workgroup_size`` is not a multiple of
  ``segment_size``.
- For ``radix_single_group_tag``, `C` is `0`. The radix value counters reuse the memory of the tile,
  so the requirement is the larger of the tile and the counters, ``(param.workgroup_size / 32) * 2^RadixBits * 2``
  bytes (`16KB` for ``param.workgroup_size`` set to `1024` and ``RadixBits`` set to `8`).
- For ``radix_many_groups_tag``, `C` does not exceed `4KB`, as for :doc:`radix_sort <radix_sort>`.

For example, at `64KB` of local memory, a tile is limited to 16K ``std::uint32_t`` keys or 8K ``std::uint64_t`` keys.

.. _batched-sort-temporary-storage:

Temporary Storage
-----------------

The overloads without ``temp_storage`` allocate and free temporary global memory internally when it is required.
The overloads with ``temp_storage`` use the provided memory instead, and do not allocate or free it.
The behavior is undefined unless all of the following are satisfied:

- ``temp_storage`` is a USM device or shared allocation, made on the same context as ``q``,
  and accessible from the device of ``q``. There is no alignment requirement; the size returned by
  ``batched_stable_sort_bytes_req`` accounts for any internal alignment.
- ``temp_storage_bytes`` is no less than the value returned by ``batched_stable_sort_bytes_req``
  for the same ``SortConfig``, key type, ``KernelParam``, ``segment_size``, and ``n``.
- The memory is not accessed otherwise until the returned event completes.

When no temporary storage is required, ``temp_storage`` may be ``nullptr`` with ``temp_storage_bytes`` set to `0`.

.. _batched-sort-selecting-config:

-----------------------------------------
Recommended Settings for Best Performance
-----------------------------------------

Selecting a Sort Configuration
------------------------------

The best configuration depends mainly on the segment size:

- ``merge_single_group_tag`` is usually fastest for short segments (e.g., up to ~256-1K elements).
- ``radix_single_group_tag`` is usually fastest for longer segments which fit into a work-group tile.
- ``radix_many_groups_tag`` is required for segments which do not fit into a work-group tile.
  It should not be used for short segments, see :ref:`Global Memory Requirements <batched-sort-memory-requirements>`.

Each call runs a single configuration. A small wrapper may dispatch between them by segment size,
with ``kernel_param`` tuned independently for each:

.. code:: cpp

   namespace kt = oneapi::dpl::experimental::kt;

   template <bool IsAscending, typename KeyT>
   sycl::event
   my_batched_sort(sycl::queue q, KeyT* keys, KeyT* keys_out, std::ptrdiff_t n, std::ptrdiff_t segment_size)
   {
      // Thresholds and kernel_params are illustrative; tune them for the target device and types.
      if (segment_size <= 256)
         return kt::gpu::batched_stable_sort<kt::gpu::merge_single_group_tag, IsAscending>(
            q, keys, keys + n, keys_out, segment_size, kt::kernel_param<4, 64>{});
      if (segment_size <= 4096)
         return kt::gpu::batched_stable_sort<kt::gpu::radix_single_group_tag<8>, IsAscending>(
            q, keys, keys + n, keys_out, segment_size, kt::kernel_param<8, 512>{});
      return kt::gpu::batched_stable_sort<kt::gpu::radix_many_groups_tag<8>, IsAscending>(
         q, keys, keys + n, keys_out, segment_size, kt::kernel_param<8, 512>{});
   }

Selecting Kernel Parameters
---------------------------

The general advice is to choose kernel parameters based on performance measurements and profiling information.
The initial configuration may be selected according to these high-level guidelines:

- For ``merge_single_group_tag`` and ``radix_single_group_tag``, the tile,
  ``param.data_per_workitem * param.workgroup_size``, must hold a segment. A larger tile packs more segments
  into a work-group, at the cost of local memory.
- For ``radix_many_groups_tag``, follow the guidelines for :doc:`radix_sort <radix_sort>`,
  applied to the segment size rather than to the total number of elements.

.. tip::

   - Optimal parameters may differ by the data type and the entropy of the underlying data, so it is important
     that this is reflected in your tuning experiments.

   - Avoid setting too large ``param.data_per_workitem`` and ``param.workgroup_size`` values
     by ensuring that :ref:`Memory requirements <batched-sort-memory-requirements>` are satisfied.

   - Large performance drops with an increase to ``param.data_per_workitem`` are indicative of excessive register spillage.
     `Ahead-of-Time (AOT) Compilation <https://www.intel.com/content/www/us/en/developer/articles/technical/ahead-of-time-compilation.html>`_
     may be used to emit warnings when this occurs.

.. [#fnote1] Andy Adinets and Duane Merrill (2022). Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. https://arxiv.org/abs/2206.01784.
