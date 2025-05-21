|onedpl_long| Introduction
#######################################

The |onedpl_long| (|onedpl_short|) is implemented in accordance with the `oneDPL
Specification <https://uxlfoundation.github.io/oneAPI-spec/spec/elements/oneDPL/source/index.html>`_.

To support heterogeneity, |onedpl_short| uses `SYCL <https://registry.khronos.org/SYCL/>`_.
More information about SYCL can be found in the `SYCL Specification`_.

Before You Begin
================

Visit the |onedpl_short| `Release Notes
<https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-dpcpp-library-release-notes.html>`_
page for:

* Where to Find the Release
* Overview
* New Features
* Fixed Issues
* Deprecation Notice
* Known Issues and Limitations
* Previous Release Notes

Install the `Intel® oneAPI Base Toolkit (Base Kit) <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html>`_
to use |onedpl_short|.

System Requirements
===================

Prerequisites
*************

C++17 is the minimal supported version of the C++ standard.
That means, any use of |onedpl_short| may require a C++17 compiler.
While some APIs of the library may accidentally work with earlier versions of the C++ standard, it is no more guaranteed.

To call Parallel API with the C++ standard aligned policies, you need to install the following software:

* A C++ compiler with support for OpenMP* 4.0 (or higher) SIMD constructs
* Depending on what parallel backend you want to use, install either:

  * |onetbb_long| or |tbb_long| 2019 and later,
  * A C++ compiler with support for OpenMP 4.5 (or higher).

For more information about parallel backends, see :doc:`Execution Policies <parallel_api/execution_policies>`.

To use Parallel API with the device execution policies, you need to install the following software:

* A C++ compiler with support for SYCL 2020.

Develop and Build Your Code with |onedpl_short|
===============================================

All |onedpl_short| header files are in the ``oneapi/dpl`` directory. To use the |onedpl_short| API,
include the corresponding header in your source code with the ``#include <oneapi/dpl/…>`` directive.
For better coexistence with the C++ standard library, include |onedpl_short| header files before the standard C++ ones.

|onedpl_short| introduces the ``namespace oneapi::dpl`` for its classes and functions. For brevity,
``namespace dpl`` is defined as an alias to ``oneapi::dpl`` and can be used interchangeably.

To use :doc:`tested C++ standard APIs <api_for_sycl_kernels/tested_standard_cpp_api>` in SYCL device code,
include the corresponding C++ standard header files and use the ``std`` namespace.

Follow the steps below to build your code with |onedpl_short|:

#. To build with the |dpcpp_cpp|, see the |dpcpp_gsg|_ for details.
#. Set the environment variables for |onedpl_short| and |onetbb_short|.

Here is an example of a command line used to compile code that contains |onedpl_short| parallel algorithms
on Linux* (depending on the code, parameters within [] could be unnecessary)::

  icpx [-fsycl] [-fiopenmp] program.cpp [-ltbb] -o program

You may also use the |pstl_offload_option|_ of |dpcpp_cpp| powered by |onedpl_short|
to build the standard C++ code for execution on a SYCL device::

  icpx -fsycl -fsycl-pstl-offload=gpu program.cpp -o program

This option redirects C++ parallel algorithms invoked with the ``std::execution::par_unseq`` policy
to |onedpl_short| algorithms. It does not change the behavior of the |onedpl_short| algorithms and
execution policies that are directly used in the code.

Useful Information
==================

.. _library-restrictions:

Difference with Standard C++ Parallel Algorithms
************************************************

* oneDPL execution policies only result in parallel execution if random access iterators are provided,
  the execution will remain serial for other iterator types.
* Function objects passed in to algorithms executed with device policies must provide ``const``-qualified ``operator()``.
  The `SYCL specification`_ states that writing to such an object during a SYCL kernel is undefined behavior.
* For the following algorithms, ``par_unseq`` and ``unseq`` policies do not result in SIMD execution:
  ``includes``, ``inplace_merge``, ``merge``, ``set_difference``, ``set_intersection``,
  ``set_symmetric_difference``, ``set_union``, ``stable_partition``, ``unique``.
* The following algorithms require additional O(n) memory space for parallel execution:
  ``copy_if``, ``inplace_merge``, ``partial_sort``, ``partial_sort_copy``, ``partition_copy``,
  ``remove``, ``remove_if``, ``rotate``, ``sort``, ``stable_sort``, ``unique``, ``unique_copy``.

Restrictions
************

When called with device execution policies, |onedpl_short| algorithms apply the same restrictions as
|dpcpp_short| does (see the |dpcpp_cpp| documentation and the SYCL specification for details), such as:

* Adding buffers to a lambda capture list is not allowed for lambdas passed to an algorithm.
* Passing data types, which are not trivially copyable, is only allowed via USM,
  but not via buffers or host-allocated containers.
* Objects of pointer-to-member types cannot be passed to an algorithm.
* The definition of lambda functions used with parallel algorithms should not depend on preprocessor macros
  that makes it different for the host and the device. Otherwise, the behavior is undefined.
* When used within SYCL kernels or transferred to/from a device, a container class can only hold objects
  whose type meets SYCL requirements for use in kernels and for data transfer, respectively.
* Calling the API that throws exception is not allowed within callable objects passed to an algorithm.

Known Limitations
*****************

* The ``oneapi::dpl::execution::par_unseq`` policy is affected by ``-fsycl-pstl-offload`` option of |dpcpp_cpp|
  when |onedpl_short| substitutes this policy for the ``std::execution::par_unseq`` policy
  missing in a standard C++ library, particularly in libstdc++ version 8 and in libc++.
* For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``,
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
* For ``transform_exclusive_scan``, ``transform_inclusive_scan`` algorithms the result of the unary operation should be
  convertible to the type of the initial value if one is provided, otherwise it is convertible to the type of values
  in the processed data sequence: ``std::iterator_traits<IteratorType>::value_type``.
* ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with
  unsequenced execution policies when building a program with GCC 10 and using ``-O0`` option.
* Compiling ``reduce`` and ``transform_reduce`` algorithms with |dpcpp_cpp| versions 2021 and older
  may result in a runtime error. To fix this issue, use |dpcpp_cpp| version 2022 or newer.
* When compiling on Windows, add the option ``/EHsc`` to the compilation command to avoid errors with oneDPL's experimental
  ranges API that uses exceptions.
* The ``using namespace oneapi;`` directive in a |onedpl_short| program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  the ``oneapi::dpl`` namespace, the shorter ``dpl`` namespace alias, or create your own alias.
* ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
* Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including: ``std::ldexp``, ``std::frexp``), and the following functions when used with ``std::complex<float>``
  as argument(s):  ``std::acosh``, ``std::asin``, ``std::asinh``, ``std::asoc``, ``std::log10``, ``std::log``, ``std::pow``,
  ``std::sqrt`` require device support for double precision.
* STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft Visual C++ standard library.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft Visual C++ standard library. For a workaround, define the
  ``_USE_STD_VECTOR_ALGORITHMS`` macro to `` 0`` to the source file before including any headers.
* ``exclusive_scan``, ``inclusive_scan``, ``exclusive_scan_by_segment``,
  ``inclusive_scan_by_segment``, ``transform_exclusive_scan``, ``transform_inclusive_scan``,
  when used with C++ standard aligned policies, impose limitations on the initial value type if an
  initial value is provided, and on the value type of the input iterator if an initial value is
  not provided.
  Firstly, it must satisfy the ``DefaultConstructible`` requirements.
  Secondly, a default-constructed instance of that type should act as the identity element for the binary scan function.
* ``reduce_by_segment``, when used with C++ standard aligned policies, imposes limitations on the value type.
  Firstly, it must satisfy the ``DefaultConstructible`` requirements.
  Secondly, a default-constructed instance of that type should act as the identity element for the binary reduction function.
* The initial value type for ``exclusive_scan``, ``inclusive_scan``, ``exclusive_scan_by_segment``,
  ``inclusive_scan_by_segment``, ``reduce``, ``reduce_by_segment``, ``transform_reduce``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan`` should satisfy the ``MoveAssignable`` and the ``CopyConstructible`` requirements.
* For ``max_element``, ``min_element``, ``minmax_element``, ``partial_sort``, ``partial_sort_copy``, ``sort``, ``stable_sort``
  the dereferenced value type of the provided iterators should satisfy the ``DefaultConstructible`` requirements.
* For ``remove``, ``remove_if``, ``unique`` the dereferenced value type of the provided
  iterators should be ``MoveConstructible``.
* When compiling with ``-O0 -g`` options on Linux with the Intel® oneAPI DPC++/C++ Compiler version 2025.0 or earlier
  the ``sort``, ``stable_sort``, ``sort_by_key``, ``stable_sort_by_key``, and ``partial_sort_copy`` may work incorrectly
  or cause a segmentation fault when used with a device execution policy on a CPU device. To avoid this issue, pass the
  ``-fsycl-device-code-split=per_kernel`` option to the compiler or use Intel® oneAPI DPC++/C++ Compiler version 2025.1
  or newer.
* ``esimd::radix_sort`` and ``esimd::radix_sort_by_key`` kernel templates fail to compile when a program
  is built with ``-g``, ``-O0``, ``-O1`` compiler options and a Linux General Purpose Intel GPUs Driver version older
  than ``2423.32`` (Rolling) and ``2350.61`` (LTS) is used.
  See the `Release Types <https://dgpu-docs.intel.com/releases/releases.html>`_
  to find information about the relevant Rolling and LTS releases.
* ``std::ranges::drop_view`` from libstdc++ version 10 may throw exceptions.
  This can lead to a "SYCL kernel cannot use exceptions" compilation error
  when it is used to pass data to a range-based algorithm with a device policy.

.. _`SYCL Specification`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
