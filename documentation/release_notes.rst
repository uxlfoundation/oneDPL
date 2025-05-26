Intel® oneAPI DPC++ Library (oneDPL) Release Notes
###################################################

Overview
=========

The Intel® oneAPI DPC++ Library (oneDPL) accompanies the Intel® oneAPI DPC++/C++ Compiler
and provides high-productivity APIs aimed to minimize programming efforts of C++ developers
creating efficient heterogeneous applications.

New in 2022.9.0
===============

New Features
------------
- Added parallel range algorithms in ``namespace oneapi::dpl::ranges``: ``fill``, ``move``, ``replace``, ``replace_if``,
  ``remove``, ``remove_if``, ``mismatch``, ``minmax_element``, ``min``, ``max``, ``find_first_of``, ``find_end``,
  ``is_sorted_until``. These algorithms operate with C++20 random access ranges.
- Improved performance of set operation algorithms when using device policies: ``set_union``, ``set_difference``,
  ``set_intersection``, ``set_symmetric_difference``.
- Improved performance of ``copy``, ``fill``, ``for_each``, ``replace``, ``reverse``, ``rotate``, ``transform`` and 30+
  other algorithms with device policies on GPUs when using ``std::reverse_iterator``.
- Added ADL-based customization point ``is_onedpl_indirectly_device_accessible``, which can be used to mark iterator
  types as *indirectly device accessible*. Added public trait ``oneapi::dpl::is_directly_device_accessible[_v]`` to
  query if types are indirectly device accessible.

Fixed Issues
------------
- Eliminated runtime exceptions encountered when compiling code that called ``inclusive_scan``, ``copy_if``,
  ``partition``, ``unique``, ``reduce_by_segment``, and related algorithms with device policies using
  the open source oneAPI DPC++ Compiler without specifying an optimization flag.
- Fixed a compilation error in ``reduce_by_segment`` regarding return type deduction when called with a device policy.
- Eliminated multiple compile time warnings throughout the library.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- The `set_intersection`, `set_difference`, `set_symmetric_difference`, and `set_union` algorithms with a device policy
require GPUs with double-precision support on Windows, regardless of the value type of the input sequences.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- Incorrect results may be observed when calling ``sort`` with a device policy on Intel® Arc™ graphics 140V with data
  sizes of 4-8 million elements.
- ``histogram`` algorithm requires the output value type to be an integral type no larger than four bytes
  when used with a device policy on hardware that does not support 64-bit atomic operations.
- ``histogram`` may provide incorrect results with device policies in a program built with ``-O0`` option and the driver
  version is 2448.13 or older.
- For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``, 
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
- Incorrect results may be produced by ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment``
  with ``unseq`` or ``par_unseq`` policy when compiled by Intel® oneAPI DPC++/C++ Compiler 2024.1 or earlier
  with ``-fiopenmp``, ``-fiopenmp-simd``, ``-qopenmp``, ``-qopenmp-simd`` options on Linux.
  To avoid the issue, pass ``-fopenmp`` or ``-fopenmp-simd`` option instead.
- With libstdc++ version 10, the compilation error *SYCL kernel cannot use exceptions* occurs
  when calling the range-based ``adjacent_find``, ``is_sorted`` or ``is_sorted_until`` algorithms with device policies.
- The range-based ``count_if`` may produce incorrect results on Intel® Data Center GPU Max Series when the driver version
  is "Rolling 2507.12" and newer.

New in 2022.8.0
===============

New Features
------------
- Added support of host policies for ``histogram`` algorithms.
- Added support for an undersized output range in the range-based ``merge`` algorithm.
- Improved performance of the ``merge`` and sorting algorithms
  (``sort``, ``stable_sort``, ``sort_by_key``, ``stable_sort_by_key``) that rely on Merge sort [#fnote1]_,
  with device policies for large data sizes.
- Improved performance of ``copy``, ``fill``, ``for_each``, ``replace``, ``reverse``, ``rotate``, ``transform`` and 30+
  other algorithms with device policies on GPUs.
- Improved oneDPL use with SYCL implementations other than Intel® oneAPI DPC++/C++ Compiler.


Fixed Issues
------------
- Fixed an issue with ``drop_view`` in the experimental range-based API.
- Fixed compilation errors in ``find_if`` and ``find_if_not`` with device policies where the user provided predicate is
  device copyable but not trivially copyable.
- Fixed incorrect results or synchronous SYCL exceptions for several algorithms when compiled with ``-O0`` and executed
  on a GPU device.
- Fixed an issue preventing inclusion of the ``<numeric>`` header after ``<execution>`` and ``<algorithm>`` headers.
- Fixed several issues in the ``sort``, ``stable_sort``, ``sort_by_key`` and ``stable_sort_by_key`` algorithms that:

   * Allows the use of non-trivially-copyable comparators.
   * Eliminates duplicate kernel names.
   * Resolves incorrect results on devices with sub-group sizes smaller than four.
   * Resolved synchronization errors that were seen on Intel® Arc™ B-series GPU devices.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- Incorrect results may be observed when calling ``sort`` with a device policy on Intel® Arc™ graphics 140V with data
  sizes of 4-8 million elements.
- ``sort``, ``stable_sort``, ``sort_by_key`` and ``stable_sort_by_key`` algorithms fail to compile
  when using Clang 17 and earlier versions, as well as compilers based on these versions,
  such as Intel® oneAPI DPC++/C++ Compiler 2023.2.0.
- When compiling code that uses device policies with the open source oneAPI DPC++ Compiler (clang++ driver),
  synchronous SYCL runtime exceptions regarding unfound kernels may be encountered unless an optimization flag is
  specified (for example ``-O1``) as opposed to relying on the compiler's default optimization level.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``histogram`` algorithm requires the output value type to be an integral type no larger than four bytes
  when used with an FPGA policy.
- ``histogram`` may provide incorrect results with device policies in a program built with ``-O0`` option.
- Compilation issues may be encountered when passing zip iterators to ``exclusive_scan_by_segment`` on Windows. 
- For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``, 
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
- Incorrect results may be produced by ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment``
  with ``unseq`` or ``par_unseq`` policy when compiled by Intel® oneAPI DPC++/C++ Compiler
  with ``-fiopenmp``, ``-fiopenmp-simd``, ``-qopenmp``, ``-qopenmp-simd`` options on Linux.
  To avoid the issue, pass ``-fopenmp`` or ``-fopenmp-simd`` option instead.

New in 2022.7.0
===============

New Features
------------
- Improved performance of the ``adjacent_find``, ``all_of``, ``any_of``, ``copy_if``, ``exclusive_scan``, ``equal``,
  ``find``, ``find_if``, ``find_end``, ``find_first_of``, ``find_if_not``, ``inclusive_scan``, ``includes``,
  ``is_heap``, ``is_heap_until``, ``is_partitioned``, ``is_sorted``, ``is_sorted_until``, ``lexicographical_compare``,
  ``max_element``, ``min_element``, ``minmax_element``, ``mismatch``, ``none_of``, ``partition``, ``partition_copy``,
  ``reduce``, ``remove``, ``remove_copy``, ``remove_copy_if``, ``remove_if``, ``search``, ``search_n``,
  ``stable_partition``, ``transform_exclusive_scan``, ``transform_inclusive_scan``, ``unique``, and ``unique_copy``
  algorithms with device policies. 
- Improved performance of ``sort``, ``stable_sort`` and ``sort_by_key`` algorithms with device policies when using Merge
  sort [#fnote1]_.
- Added ``stable_sort_by_key`` algorithm in ``namespace oneapi::dpl``. 
- Added parallel range algorithms in ``namespace oneapi::dpl::ranges``: ``all_of``, ``any_of``,
  ``none_of``, ``for_each``, ``find``, ``find_if``, ``find_if_not``, ``adjacent_find``, ``search``, ``search_n``,
  ``transform``, ``sort``, ``stable_sort``, ``is_sorted``, ``merge``, ``count``, ``count_if``, ``equal``, ``copy``,
  ``copy_if``, ``min_element``, ``max_element``. These algorithms operate with C++20 random access ranges
  and views while also taking an execution policy similarly to other oneDPL algorithms.
- Added support for operators ==, !=, << and >> for RNG engines and distributions.
- Added experimental support for the Philox RNG engine in ``namespace oneapi::dpl::experimental``.
- Added the ``<oneapi/dpl/version>`` header containing oneDPL version macros and new feature testing macros.

Fixed Issues
------------
- Fixed unused variable and unused type warnings.
- Fixed memory leaks when using ``sort`` and ``stable_sort`` algorithms with the oneTBB backend.
- Fixed a build error for ``oneapi::dpl::begin`` and ``oneapi::dpl::end`` functions used with
  the Microsoft* Visual C++ standard library and with C++20.
- Reordered template parameters of the ``histogram`` algorithm to match its function parameter order.
  For affected ``histogram`` calls we recommend to remove explicit specification of template parameters
  and instead add explicit type conversions of the function arguments as necessary.
- ``gpu::esimd::radix_sort`` and ``gpu::esimd::radix_sort_by_key`` kernel templates now throw ``std::bad_alloc``
  if they fail to allocate global memory.
- Fixed a potential hang occurring with ``gpu::esimd::radix_sort`` and
  ``gpu::esimd::radix_sort_by_key`` kernel templates. 
- Fixed documentation for ``sort_by_key`` algorithm, which used to be mistakenly described as stable, despite being
  possibly unstable for some execution policies. If stability is required, use ``stable_sort_by_key`` instead. 
- Fixed an error when calling ``sort`` with device execution policies on CUDA devices.
- Allow passing C++20 random access iterators to oneDPL algorithms.
- Fixed issues caused by initialization of SYCL queues in the predefined device execution policies.
  These policies have been updated to be immutable (``const``) objects.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- ``histogram`` may provide incorrect results with device policies in a program built with -O0 option.
- Inclusion of ``<oneapi/dpl/dynamic_selection>`` prior to ``<oneapi/dpl/random>`` may result in compilation errors.
  Include ``<oneapi/dpl/random>`` first as a workaround.
- Incorrect results may occur when using ``oneapi::dpl::experimental::philox_engine`` with no predefined template
  parameters and with `word_size` values other than 64 and 32.
- Incorrect results or a synchronous SYCL exception may be observed with the following algorithms built
  with -O0 option and executed on a GPU device: ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``copy_if``, ``remove``, ``remove_copy``, ``remove_copy_if``, ``remove_if``,
  ``partition``, ``partition_copy``, ``stable_partition``, ``unique``, ``unique_copy``, and ``sort``.
- The value type of the input sequence should be convertible to the type of the initial element for the following
  algorithms with device execution policies: ``transform_inclusive_scan``, ``transform_exclusive_scan``,
  ``inclusive_scan``, and ``exclusive_scan``.
- The following algorithms with device execution policies may exceed the C++ standard requirements on the number
  of applications of user-provided predicates or equality operators: ``copy_if``, ``remove``, ``remove_copy``,
  ``remove_copy_if``, ``remove_if``, ``partition_copy``, ``unique``, and ``unique_copy``. In all cases,
  the predicate or equality operator is applied ``O(n)`` times.
- The ``adjacent_find``, ``all_of``, ``any_of``, ``equal``, ``find``, ``find_if``, ``find_end``, ``find_first_of``,
  ``find_if_not``, ``includes``, ``is_heap``, ``is_heap_until``, ``is_sorted``, ``is_sorted_until``, ``mismatch``,
  ``none_of``, ``search``, and ``search_n`` algorithms may cause a segmentation fault when used with a device execution
  policy on a CPU device, and built on Linux with Intel® oneAPI DPC++/C++ Compiler 2025.0.0 and -O0 -g compiler options.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``histogram`` algorithm requires the output value type to be an integral type no larger than 4 bytes
  when used with an FPGA policy.
- Compilation issues may be encountered when passing zip iterators to ``exclusive_scan_by_segment`` on Windows. 
- For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``, 
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
- ``sort``, ``stable_sort``, ``sort_by_key``, ``stable_sort_by_key``, ``partial_sort_copy`` algorithms
  may work incorrectly or cause a segmentation fault when used a device execution policy on a CPU device,
  and built on Linux with Intel® oneAPI DPC++/C++ Compiler and -O0 -g compiler options.
  To avoid the issue, pass ``-fsycl-device-code-split=per_kernel`` option to the compiler.
- Incorrect results may be produced by ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment``
  with ``unseq`` or ``par_unseq`` policy when compiled by Intel® oneAPI DPC++/C++ Compiler
  with ``-fiopenmp``, ``-fiopenmp-simd``, ``-qopenmp``, ``-qopenmp-simd`` options on Linux.
  To avoid the issue, pass ``-fopenmp`` or ``-fopenmp-simd`` option instead.
- Incorrect results may be produced by ``reduce``, ``reduce_by_segment``, and ``transform_reduce``
  with 64-bit data types when compiled by Intel® oneAPI DPC++/C++ Compiler versions 2021.3 and newer
  and executed on a GPU device. For a workaround, define the ``ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION``
  macro to ``1`` before including oneDPL header files.
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2022.6.0
===============
News
------------
- `oneAPI DPC++ Library Manual Migration Guide`_ to simplify the migration of Thrust* and CUB* APIs from CUDA*. 
- ``radix_sort`` and ``radix_sort_by_key`` kernel templates were moved into
  ``oneapi::dpl::experimental::kt::gpu::esimd`` namespace. The former ``oneapi::dpl::experimental::kt::esimd``
  namespace is deprecated and will be removed in a future release.
- The ``for_loop``, ``for_loop_strided``, ``for_loop_n``,  ``for_loop_n_strided`` algorithms
  in `namespace oneapi::dpl::experimental` are enforced to fail with device execution policies.

New Features
------------
- Added experimental ``inclusive_scan`` kernel template algorithm residing in
  the ``oneapi::dpl::experimental::kt::gpu`` namespace. 
- ``radix_sort`` and ``radix_sort_by_key`` kernel templates are extended with overloads for out-of-place sorting.
  These overloads preserve the input sequence and sort data into the user provided output sequence.
- Improved performance of the ``reduce``, ``min_element``, ``max_element``, ``minmax_element``, ``is_partitioned``,
  ``lexicographical_compare``, ``binary_search``, ``lower_bound``, and ``upper_bound`` algorithms with device policies.
-  ``sort``, ``stable_sort``, ``sort_by_key`` algorithms now use Radix sort [#fnote1]_
   for sorting ``sycl::half`` elements compared with ``std::less`` or ``std::greater``.

Fixed Issues
------------
- Fixed compilation errors when using ``reduce``, ``min_element``, ``max_element``, ``minmax_element``,
  ``is_partitioned``, and ``lexicographical_compare`` with Intel oneAPI DPC++/C++ compiler 2023.0 and earlier.
- Fixed possible data races in the following algorithms used with device execution policies:
  ``remove_if``, ``unique``, ``inplace_merge``, ``stable_partition``, ``partial_sort_copy``, ``rotate``.
- Fixed excessive copying of data in ``std::vector`` allocated with a USM allocator for standard library
  implementations which have allocator information in the ``std::vector::iterator`` type.
- Fixed an issue where checking ``std::is_default_constructible`` for ``transform_iterator`` with a functor
  that is not default-constructible could cause a build error or an incorrect result.
- Fixed handling of `sycl device copyable`_ for internal and public oneDPL types.
- Fixed handling of ``std::reverse_iterator`` as input to oneDPL algorithms using a device policy.
- Fixed ``set_intersection`` to always copy from the first input sequence to the output,
  where previously some calls would copy from the second input sequence.
- Fixed compilation errors when using ``oneapi::dpl::zip_iterator`` with the oneTBB backend and C++20.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- ``histogram`` algorithm requires the output value type to be an integral type no larger than 4 bytes
  when used with an FPGA policy.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- When compiled with ``-fsycl-pstl-offload`` option of Intel oneAPI DPC++/C++ compiler and with
  libstdc++ version 8 or libc++, ``oneapi::dpl::execution::par_unseq`` offloads
  standard parallel algorithms to the SYCL device similarly to ``std::execution::par_unseq``
  in accordance with the ``-fsycl-pstl-offload`` option value.
- When using the dpl modulefile to initialize the user's environment and compiling with ``-fsycl-pstl-offload``
  option of Intel® oneAPI DPC++/C++ compiler, a linking issue or program crash may be encountered due to the directory
  containing libpstloffload.so not being included in the search path. Use the env/vars.sh to configure the working
  environment to avoid the issue.
- Compilation issues may be encountered when passing zip iterators to ``exclusive_scan_by_segment`` on Windows. 
- For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``, 
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
- ``sort``, ``stable_sort``, ``sort_by_key``, ``partial_sort_copy`` algorithms may work incorrectly or cause
  a segmentation fault when used a DPC++ execution policy for CPU device, and built
  on Linux with Intel® oneAPI DPC++/C++ Compiler and -O0 -g compiler options.
  To avoid the issue, pass ``-fsycl-device-code-split=per_kernel`` option to the compiler.
- Incorrect results may be produced by ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment``
  with ``unseq`` or ``par_unseq`` policy when compiled by Intel® oneAPI DPC++/C++ Compiler
  with ``-fiopenmp``, ``-fiopenmp-simd``, ``-qopenmp``, ``-qopenmp-simd`` options on Linux.
  To avoid the issue, pass ``-fopenmp`` or ``-fopenmp-simd`` option instead.
- Incorrect results may be produced by ``reduce``, ``reduce_by_segment``, and ``transform_reduce``
  with 64-bit data types when compiled by Intel® oneAPI DPC++/C++ Compiler versions 2021.3 and newer
  and executed on GPU devices.
  For a workaround, define the ``ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION`` macro to ``1`` before
  including oneDPL header files.
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2022.5.0
===============

New Features
------------
- Added new ``histogram`` algorithms for generating a histogram from an input sequence into
  an output sequence representing either equally spaced or user-defined bins.
  These algorithms are currently only available for device execution policies.
- Supported zip_iterator for ``transform`` algorithm.

Fixed Issues
------------
- Fixed handling of ``permutation_iterator`` as input to oneDPL algorithms for a variety of
  source iterator and permutation types which caused issues.
- Fixed ``zip_iterator`` to be `sycl device copyable`_ for trivially copyable source iterator types.
- Added a workaround for reduction algorithm failures with 64-bit data types. Define
  the ``ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION`` macro to ``1`` before including oneDPL header files.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- Crashes or incorrect results may occur when using ``oneapi::dpl::reverse_iterator`` or
  ``std::reverse_iterator`` as input to oneDPL algorithms with device execution policies.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- When compiled with ``-fsycl-pstl-offload`` option of Intel oneAPI DPC++/C++ compiler and with
  libstdc++ version 8 or libc++, ``oneapi::dpl::execution::par_unseq`` offloads
  standard parallel algorithms to the SYCL device similarly to ``std::execution::par_unseq``
  in accordance with the ``-fsycl-pstl-offload`` option value.
- When using the dpl modulefile to initialize the user's environment and compiling with ``-fsycl-pstl-offload``
  option of Intel® oneAPI DPC++/C++ compiler, a linking issue or program crash may be encountered due to the directory
  containing libpstloffload.so not being included in the search path. Use the env/vars.sh to configure the working
  environment to avoid the issue.
- Compilation issues may be encountered when passing zip iterators to ``exclusive_scan_by_segment`` on Windows.
- Incorrect results may be produced by ``set_intersection`` with a DPC++ execution policy,
  where elements are copied from the second input range rather than the first input range. 
- For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``, 
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
- ``sort``, ``stable_sort``, ``sort_by_key``, ``partial_sort_copy`` algorithms may work incorrectly or cause
  a segmentation fault when used a DPC++ execution policy for CPU device, and built
  on Linux with Intel® oneAPI DPC++/C++ Compiler and -O0 -g compiler options.
  To avoid the issue, pass ``-fsycl-device-code-split=per_kernel`` option to the compiler.
- Incorrect results may be produced by ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment``
  with ``unseq`` or ``par_unseq`` policy when compiled by Intel® oneAPI DPC++/C++ Compiler
  with ``-fiopenmp``, ``-fiopenmp-simd``, ``-qopenmp``, ``-qopenmp-simd`` options on Linux.
  To avoid the issue, pass ``-fopenmp`` or ``-fopenmp-simd`` option instead.
- Incorrect results may be produced by ``reduce``, ``reduce_by_segment``, and ``transform_reduce``
  with 64-bit data types when compiled by Intel® oneAPI DPC++/C++ Compiler versions 2021.3 and newer
  and executed on GPU devices.
  For a workaround, define the ``ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION`` macro to ``1`` before
  including oneDPL header files.
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2022.4.0
===============

New Features
------------
- Added experimental ``radix_sort`` and ``radix_sort_by_key`` algorithms residing in
  the ``oneapi::dpl::experimental::kt::esimd`` namespace. These algorithms are first
  in the family of _kernel templates_ that allow configuring a variety of parameters
  including the number of elements to process by a work item, and the size of a workgroup.
  The algorithms only work with Intel® Data Center GPU Max Series.
- Added new ``transform_if`` algorithm for applying a transform function conditionally
  based on a predicate, with overloads provided for one and two input sequences
  that use correspondingly unary and binary operations and predicates.
- Optimizations used with Intel® oneAPI DPC++/C++ Compiler are expanded to the open source oneAPI DPC++ compiler.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- ``esimd::radix_sort`` and ``esimd::radix_sort_by_key`` kernel templates fail to compile when a program
  is built with -g, -O0, -O1 compiler options.
- ``esimd::radix_sort_by_key`` kernel template produces wrong results with the following combinations
  of ``kernel_param`` and types of keys and values:
    - ``sizeof(key_type) + sizeof(val_type) == 12``, ``kernel_param::workgroup_size == 64``, and ``kernel_param::data_per_workitem == 96``
    - ``sizeof(key_type) + sizeof(val_type) == 16``, ``kernel_param::workgroup_size == 64``, and ``kernel_param::data_per_workitem == 64``

New in 2022.3.0
===============

New Features
------------
- Added an experimental feature to dynamically select an execution context, e.g., a SYCL queue.
  The feature provides selection functions such as ``select``, ``submit`` and ``submit_and_wait``,
  and several selection policies: ``fixed_resource_policy``, ``round_robin_policy``,
  ``dynamic_load_policy``, and ``auto_tune_policy``.
- ``unseq`` and ``par_unseq`` policies now enable vectorization also for Intel oneAPI DPC++/C++ Compiler.
- Added support for passing zip iterators as segment value data in ``reduce_by_segment``, ``exclusive_scan_by_segment``,
  and ``inclusive_scan_by_segment``.
- Improved performance of the ``merge``, ``sort``, ``stable_sort``, ``sort_by_key``,
  ``reduce``, ``min_element``, ``max_element``, ``minmax_element``, ``is_partitioned``, and
  ``lexicographical_compare`` algorithms with DPC++ execution policies.

Fixed Issues
------------
- Fixed the ``reduce_async`` function to not ignore the provided binary operation.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- When compiled with ``-fsycl-pstl-offload`` option of Intel oneAPI DPC++/C++ compiler and with
  libstdc++ version 8 or libc++, ``oneapi::dpl::execution::par_unseq`` offloads
  standard parallel algorithms to the SYCL device similarly to ``std::execution::par_unseq``
  in accordance with the ``-fsycl-pstl-offload`` option value.
- When using the dpl modulefile to initialize the user's environment and compiling with ``-fsycl-pstl-offload``
  option of Intel® oneAPI DPC++/C++ compiler, a linking issue or program crash may be encountered due to the directory
  containing libpstloffload.so not being included in the search path. Use the env/vars.sh to configure the working
  environment to avoid the issue.
- Compilation issues may be encountered when passing zip iterators to ``exclusive_scan_by_segment`` on Windows.
- Incorrect results may be produced by ``set_intersection`` with a DPC++ execution policy,
  where elements are copied from the second input range rather than the first input range. 
- For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``, 
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
- ``sort``, ``stable_sort``, ``sort_by_key``, ``partial_sort_copy`` algorithms may work incorrectly or cause
  a segmentation fault when used a DPC++ execution policy for CPU device, and built
  on Linux with Intel® oneAPI DPC++/C++ Compiler and -O0 -g compiler options.
  To avoid the issue, pass ``-fsycl-device-code-split=per_kernel`` option to the compiler.
- Incorrect results may be produced by ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment``
  with ``unseq`` or ``par_unseq`` policy when compiled by Intel® oneAPI DPC++/C++ Compiler
  with ``-fiopenmp``, ``-fiopenmp-simd``, ``-qopenmp``, ``-qopenmp-simd`` options on Linux.
  To avoid the issue, pass ``-fopenmp`` or ``-fopenmp-simd`` option instead.
- Incorrect results may be produced by ``reduce``, ``reduce_by_segment``, and ``transform_reduce``
  with 64-bit data types when compiled by Intel® oneAPI DPC++/C++ Compiler versions 2021.3 and newer
  and executed on GPU devices.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2022.2.0
===============

New Features
------------
- Added ``sort_by_key`` algorithm for key-value sorting.
- Improved performance of the ``reduce``, ``min_element``, ``max_element``, ``minmax_element``,
  ``is_partitioned``, and ``lexicographical_compare`` algorithms with DPC++ execution policies.
- Improved performance of the ``reduce_by_segment``, ``inclusive_scan_by_segment``, and
  ``exclusive_scan_by_segment`` algorithms for binary operators with known identities
  when using DPC++ execution policies.
- Added ``value_type`` to all views in ``oneapi::dpl::experimental::ranges``. 
- Extended ``oneapi::dpl::experimental::ranges::sort`` to support projections applied to the range elements prior to comparison.

Fixed Issues
------------
- The minimally required CMake version is raised to 3.11 on Linux and 3.20 on Windows.
- Added new CMake package ``oneDPLIntelLLVMConfig.cmake`` to resolve issues using CMake 3.20+ on Windows for icx and icx-cl.
- Fixed an error in the ``sort`` and ``stable_sort`` algorithms when performing a descending sort
  on signed numeric types with negative values.
- Fixed an error in ``reduce_by_segment`` algorithm when a non-commutative predicate is used.
- Fixed an error in ``sort`` and ``stable_sort`` algorithms for integral types wider than 4 bytes.
- Fixed an error for some compilers where OpenMP or SYCL backend was selected by CMake scripts without full compiler support.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- Incorrect results may be produced with in-place scans using ``unseq`` and ``par_unseq`` policies on
  CPUs with the Intel® C++ Compiler 2021.8.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2022.1.1
===============

New Features
------------
- Improved ``sort`` algorithm performance for the arithmetic data types with ``std::less`` or ``std::greater`` comparison operator and DPC++ policy.

Fixes Issues
------------
- Fixed an error that caused segmentation faults in ``transform_reduce``, ``minmax_element``, and related algorithms when ran on CPU devices. 
- Fixed a compilation error in ``transform_reduce``, ``minmax_element``, and related algorithms on FPGAs.
- Fixed ``permutation_iterator`` to support C-style array as a permutation map.
- Fixed a radix-sort issue with 64-bit signed integer types.

New in 2022.1.0
===============

New Features
------------
- Added ``generate``, ``generate_n``, ``transform`` algorithms to `Tested Standard C++ API`_.
- Improved performance of the ``inclusive_scan``, ``exclusive_scan``, ``reduce`` and
  ``max_element`` algorithms with DPC++ execution policies.

Fixed Issues
------------
- Added a workaround for the ``TBB headers not found`` issue occurring with libstdc++ version 9 when
  oneTBB headers are not present in the environment. The workaround requires inclusion of
  the oneDPL headers before the libstdc++ headers.
- When possible, oneDPL CMake scripts now enforce C++17 as the minimally required language version.
- Fixed an error in the ``exclusive_scan`` algorithm when the output iterator is equal to the
  input iterator (in-place scan).

Known Issues and Limitations
----------------------------
Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.


New in 2022.0.0
===============

New Features
------------
- Added the functionality from ``<complex>`` and more APIs from ``<cmath>`` and ``<limits>``
  standard headers to `Tested Standard C++ API`_.
- Improved performance of ``sort`` and ``stable_sort``  algorithms on GPU devices when using Radix sort [#fnote1]_.

Fixed Issues
------------
- Fixed permutation_iterator to work with C++ lambda functions for index permutation.
- Fixed an error in ``oneapi::dpl::experimental::ranges::guard_view`` and ``oneapi::dpl::experimental::ranges::zip_view``
  when using ``operator[]`` with an index exceeding the limits of a 32 bit integer type.
- Fixed errors when data size is 0 in ``upper_bound``, ``lower_bound`` and ``binary_search`` algorithms.

Changes affecting backward compatibility
----------------------------------------
- Removed support of C++11 and C++14.
- Changed the size and the layout of the ``discard_block_engine`` class template.
  
  For further details, please refer to `2022.0 Changes`_.

Known Issues and Limitations
----------------------------
Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2021.7.1
===============

New Features
------------
- Added possibility to construct a zip_iterator out of a std::tuple of iterators.
- Added 9 more serial-based versions of algorithms: ``is_heap``, ``is_heap_until``, ``make_heap``, ``push_heap``,
  ``pop_heap``, ``is_sorted``, ``is_sorted_until``, ``partial_sort``, ``partial_sort_copy``.
  Please refer to `Tested Standard C++ API`_.
  
Fixed Issues
------------
- Added namespace alias ``dpl = oneapi::dpl`` into all public headers.
- Fixed error in ``reduce_by_segment`` algorithm.
- Fixed wrong results error in algorithms call with permutation iterator.
  
Known Issues and Limitations
----------------------------
Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.
  
New in 2021.7.0
===============

Deprecation Notice
------------------
- Deprecated support of C++11 for Parallel API with host execution policies (``seq``, ``unseq``, ``par``, ``par_unseq``).
  C++17 is the minimal required version going forward.

Fixed Issues
------------
- Fixed a kernel name definition error in range-based algorithms and ``reduce_by_segment`` used with
  a device_policy object that has no explicit kernel name.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2021.6.1
===============

Fixed Issues
------------
- Fixed compilation errors with C++20.
- Fixed ``CL_OUT_OF_RESOURCES`` issue for Radix sort algorithm executed on CPU devices.
- Fixed crashes in ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment`` algorithms applied to
  device-allocated USM.

Known Issues and Limitations
----------------------------
- No new issues in this release. 

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.

New in 2021.6
=============

New Features
------------
- Added a new implementation for ``par`` and ``par_unseq`` execution policies based on OpenMP* 4.5 pragmas.
  It can be enabled with the ``ONEDPL_USE_OPENMP_BACKEND`` macro.
  For more details, see `Macros`_ page in oneDPL Guide.
- Added the range-based version of the ``reduce_by_segment`` algorithm and improved performance of
  the iterator-based ``reduce_by_segment`` APIs. 
  Please note that the use of the ``reduce_by_segment`` algorithm requires C++17.
- Added the following algorithms (serial versions) to `Tested Standard C++ API`_: ``for_each_n``, ``copy``,
  ``copy_backward``, ``copy_if``, ``copy_n``, ``is_permutation``, ``fill``, ``fill_n``, ``move``, ``move_backward``.

Changes affecting backward compatibility
----------------------------------------
- Fixed ``param_type`` API of random number distributions to satisfy C++ standard requirements.
  The new definitions of ``param_type`` are not compatible with incorrect definitions in previous library versions.
  Recompilation is recommended for all codes that might use ``param_type``.

Fixed Issues
------------
- Fixed hangs and errors when oneDPL is used together with oneAPI Math Kernel Library (oneMKL) in
  Data Parallel C++ (DPC++) programs.
- Fixed possible data races in the following algorithms used with DPC++ execution
  policies: ``sort``, ``stable_sort``, ``partial_sort``, ``nth_element``.

Known Issues and Limitations
----------------------------
- No new issues in this release.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.

New in 2021.5
=============

New Features
------------
- Added new random number distributions: ``exponential_distribution``, ``bernoulli_distribution``,
  ``geometric_distribution``, ``lognormal_distribution``, ``weibull_distribution``, ``cachy_distribution``,
  ``extreme_value_distribution``.
- Added the following algorithms (serial versions) to `Tested Standard C++ API`_: ``all_of``, ``any_of``, 
  ``none_of``, ``count``, ``count_if``, ``for_each``, ``find``, ``find_if``, ``find_if_not``.
- Improved performance of ``search`` and ``find_end`` algorithms on GPU devices.

Fixed Issues
------------
- Fixed SYCL* 2020 features deprecation warnings.
- Fixed some corner cases of ``normal_distribution`` functionality.
- Fixed a floating point exception occurring on CPU devices when a program uses a lot of oneDPL algorithms and DPC++ kernels.
- Fixed possible hanging and data races of the following algorithms used with DPC++ execution policies: ``count``, ``count_if``, ``is_partitioned``, ``lexicographical_compare``, ``max_element``, ``min_element``, ``minmax_element``,    ``reduce``, ``transform_reduce``.

Known Issues and Limitations
----------------------------

New in This Release
^^^^^^^^^^^^^^^^^^^
- The definition of lambda functions used with parallel algorithms should not depend on preprocessor macros
  that makes it different for the host and the device. Otherwise, the behavior is undefined.

Existing Issues
^^^^^^^^^^^^^^^
- ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with vector execution policies
  when building a program with GCC 10 and using -O0 option.
- Some algorithms may hang when a program is built with -O0 option, executed on GPU devices and large number of elements is to be processed.
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library.
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
- The implementation does not yet provide ``namespace oneapi::std`` as defined in the oneDPL Specification.
- The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher)
  or Clang 7 (or higher).
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects
  whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.

New in 2021.4
=============

New Features
------------
-  Added the range-based versions of the following algorithms: ``any_of``, ``adjacent_find``,
   ``copy_if``, ``none_of``, ``remove_copy_if``, ``remove_copy``, ``replace_copy``, 
   ``replace_copy_if``, ``reverse``, ``reverse_copy``, ``rotate_copy``, ``swap_ranges``,
   ``unique``, ``unique_copy``.
-  Added new asynchronous algorithms: ``inclusive_scan_async``, ``exclusive_scan_async``,
   ``transform_inclusive_scan_async``, ``transform_exclusive_scan_async``.
-  Added structured binding support for ``zip_iterator::value_type``.

Fixed Issues
------------
-  Fixed an issue with asynchronous algorithms returning ``future<ptr>`` with unified shared memory (USM).

Known Issues and Limitations
----------------------------

New in This Release
^^^^^^^^^^^^^^^^^^^
-  With Intel® oneAPI DPC++/C++ Compiler, ``unseq`` and ``par_unseq`` execution policies do not use OpenMP SIMD pragmas
   due to compilation issues with the ``-fopenm-simd`` option, possibly resulting in suboptimal performance.
-  The ``oneapi::dpl::experimental::ranges::reverse`` algorithm does not compile with ``-fno-sycl-unnamed-lambda`` option.

Existing Issues
^^^^^^^^^^^^^^^
- ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with vector execution policies
  when building a program with GCC 10 and using -O0 option.
- Some algorithms may hang when a program is built with -O0 option, executed on GPU devices and large number of elements is to be processed.
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library.
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
- The implementation does not yet provide ``namespace oneapi::std`` as defined in the oneDPL Specification.
- The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher)
  or Clang 7 (or higher).
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects
  whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.

New in 2021.3
=============

New Features
------------
-  Added the range-based versions of the following algorithms: ``all_of``, ``any_of``, ``count``,
   ``count_if``, ``equal``, ``move``, ``remove``, ``remove_if``, ``replace``, ``replace_if``.
-  Added the following utility ranges (views): ``generate``, ``fill``, ``rotate``.

Changes to Existing Features
-----------------------------
-  Improved performance of ``discard_block_engine`` (including ``ranlux24``, ``ranlux48``,
   ``ranlux24_vec``, ``ranlux48_vec`` predefined engines) and ``normal_distribution``.
- Added two constructors to ``transform_iterator``: the default constructor and a constructor from an iterator without a transformation.
  ``transform_iterator`` constructed these ways uses transformation functor of type passed in template arguments.
- ``transform_iterator`` can now work on top of forward iterators.

Fixed Issues
------------
-  Fixed execution of ``swap_ranges`` algorithm with ``unseq``, ``par`` execution policies.
-  Fixed an issue causing memory corruption and double freeing in scan-based algorithms compiled with
   -O0 and -g options and run on CPU devices.
-  Fixed incorrect behavior in the ``exclusive_scan`` algorithm that occurred when the input and output iterator ranges overlapped.
-  Fixed error propagation for async runtime exceptions by consistently calling ``sycl::event::wait_and_throw`` internally.
-  Fixed the warning: ``local variable will be copied despite being returned by name [-Wreturn-std-move]``.

Known Issues and Limitations
-----------------------------
- No new issues in this release. 

Existing Issues
^^^^^^^^^^^^^^^^
- ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with vector execution policies
  when building a program with GCC 10 and using -O0 option.
- Some algorithms may hang when a program is built with -O0 option, executed on GPU devices and large number of elements is to be processed.
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library.
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
- The implementation does not yet provide ``namespace oneapi::std`` as defined in the oneDPL Specification.
- The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher)
  or Clang 7 (or higher).
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects
  whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.

New in 2021.2
=============

New Features
------------
-  Added support of parallel, vector and DPC++ execution policies for the following algorithms: ``shift_left``, ``shift_right``.
-  Added the range-based versions of the following algorithms: ``sort``, ``stable_sort``, ``merge``.
-  Added experimental asynchronous algorithms: ``copy_async``, ``fill_async``, ``for_each_async``, ``reduce_async``, ``sort_async``, ``transform_async``, ``transform_reduce_async``.
   These algorithms are declared in ``oneapi::dpl::experimental`` namespace and implemented only for DPC++ policies.
   In order to make these algorithms available the ``<oneapi/dpl/async>`` header should be included. Use of the asynchronous API requires C++11.
-  Utility function ``wait_for_all`` enables waiting for completion of an arbitrary number of events.
-  Added the ``ONEDPL_USE_PREDEFINED_POLICIES`` macro, which enables predefined policy objects and
   ``make_device_policy``, ``make_fpga_policy`` functions without arguments. It is turned on by default.

Changes to Existing Features
-----------------------------
- Improved performance of the following algorithms: ``count``, ``count_if``, ``is_partitioned``,
  ``lexicographical_compare``, ``max_element``, ``min_element``,  ``minmax_element``, ``reduce``, ``transform_reduce``,
  and ``sort``, ``stable_sort`` when using Radix sort [#fnote1]_.
- Improved performance of the linear_congruential_engine RNG engine (including ``minstd_rand``, ``minstd_rand0``,
  ``minstd_rand_vec``, ``minstd_rand0_vec`` predefined engines).

Fixed Issues
------------
- Fixed runtime errors occurring with ``find_end``, ``search``, ``search_n`` algorithms when a program is built with -O0 option and executed on CPU devices.
- Fixed the majority of unused parameter warnings.

Known Issues and Limitations
-----------------------------
- ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with vector execution policies
  when building a program with GCC 10 and using -O0 option.
- Some algorithms may hang when a program is built with -O0 option, executed on GPU devices and large number of elements is to be processed.
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library.
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
- The implementation does not yet provide ``namespace oneapi::std`` as defined in the oneDPL Specification.
- The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher)
  or Clang 7 (or higher).
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects
  whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.

New in 2021.1 Gold
===================

Key Features
-------------
- This version implements the oneDPL Specification v1.0, including parallel algorithms,
  DPC++ execution policies, special iterators, and other utilities.
- oneDPL algorithms can work with data in DPC++ buffers as well as in unified shared memory (USM).
- For several algorithms, experimental API that accepts ranges (similar to C++20) is additionally provided.
- A subset of the standard C++ libraries for Microsoft* Visual C++, GCC, and Clang is supported
  in DPC++ kernels, including ``<array>``, ``<complex>``, ``<functional>``, ``<tuple>``,
  ``<type_traits>``, ``<utility>`` and other standard library API.
  For the detailed list, please refer to `oneDPL Guide`_.
- Standard C++ random number generators and distributions for use in DPC++ kernels.


Known Issues and Limitations
-----------------------------
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library.
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms are prone to ``CL_BUILD_PROGRAM_FAILURE``
  when a program uses Radix sort [#fnote1]_, is built with -O0 option and executed on CPU devices.
- The implementation does not yet provide ``namespace oneapi::std`` as defined in the oneDPL Specification.
- The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher)
  or Clang 7 (or higher).
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects
  whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.

.. [#fnote1] The sorting algorithms in oneDPL use Radix sort for arithmetic data types and
   ``sycl::half`` (since oneDPL 2022.6) compared with ``std::less`` or ``std::greater``, otherwise Merge sort.
.. _`oneDPL Guide`: https://uxlfoundation.github.io/oneDPL/index.html
.. _`Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`: https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-threading-building-blocks-release-notes.html
.. _`restrictions and known limitations`: https://uxlfoundation.github.io/oneDPL/introduction.html#restrictions.
.. _`Tested Standard C++ API`: https://uxlfoundation.github.io/oneDPL/api_for_sycl_kernels/tested_standard_cpp_api.html#tested-standard-c-api-reference
.. _`Macros`: https://uxlfoundation.github.io/oneDPL/macros.html
.. _`2022.0 Changes`: https://uxlfoundation.github.io/oneDPL/oneDPL_2022.0_changes.html
.. _`sycl device copyable`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec::device.copyable
.. _`oneAPI DPC++ Library Manual Migration Guide`: https://www.intel.com/content/www/us/en/developer/articles/guide/oneapi-dpcpp-library-manual-migration.html 
