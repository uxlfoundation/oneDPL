|onedpl_long| Introduction
#######################################

The |onedpl_long| (|onedpl_short|) is implemented in accordance with the `oneDPL
Specification <https://spec.oneapi.io/versions/latest/elements/oneDPL/source/index.html>`_.

To support heterogeneity, |onedpl_short| works with the DPC++ API. More information can be found in the
`oneAPI Specification <https://spec.oneapi.io/versions/latest/elements/sycl/source/index.html>`_.

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

Install the `Intel® oneAPI Base Toolkit (Base Kit) <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html#gs.xaontv>`_
to use |onedpl_short|.

System Requirements
===================

Prerequisites
*************

C++17 is the minimal supported version of the C++ standard.
That means, any use of |onedpl_short| may require a C++17 compiler.
While some APIs of the library may accidentally work with earlier versions of the C++ standard, it is no more guaranteed.

To call Parallel API with the C++ standard policies, you need to install the following software:

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

Below is an example of a command line used to compile code that contains |onedpl_short| parallel algorithms
on Linux* (depending on the code, parameters within [] could be unnecessary):

.. code:: cpp

  icpx [-fsycl] [-fiopenmp] program.cpp [-ltbb] -o program

Useful Information
==================

Difference with Standard C++ Parallel Algorithms
************************************************

* oneDPL execution policies only result in parallel execution if random access iterators are provided,
  the execution will remain serial for other iterator types.
* Function objects passed in to algorithms executed with device policies must provide ``const``-qualified ``operator()``.
  `The SYCL specification <https://registry.khronos.org/SYCL/>`_ states that writing to such an object during a SYCL
  kernel is undefined behavior.
* For the following algorithms, par_unseq and unseq policies do not result in vectorized execution:
  ``includes``, ``inplace_merge``, ``merge``, ``set_difference``, ``set_intersection``,
  ``set_symmetric_difference``, ``set_union``, ``stable_partition``, ``unique``.
* The following algorithms require additional O(n) memory space for parallel execution:
  ``copy_if``, ``inplace_merge``, ``partial_sort``, ``partial_sort_copy``, ``partition_copy``,
  ``remove``, ``remove_if``, ``rotate``, ``sort``, ``stable_sort``, ``unique``, ``unique_copy``.


Restrictions
************

When called with |dpcpp_short| execution policies, |onedpl_short| algorithms apply the same restrictions as
|dpcpp_short| does (see the |dpcpp_short| specification and the SYCL specification for details), such as:

* Adding buffers to a lambda capture list is not allowed for lambdas passed to an algorithm.
* Passing data types, which are not trivially copyable, is only allowed via USM,
  but not via buffers or host-allocated containers.
* The definition of lambda functions used with parallel algorithms should not depend on preprocessor macros
  that makes it different for the host and the device. Otherwise, the behavior is undefined.
* When used within SYCL kernels or transferred to/from a device, a container class can only hold objects
  whose type meets SYCL requirements for use in kernels and for data transfer, respectively.
* Calling the API that throws exception is not allowed within callable objects passed to an algorithm.

Known Limitations
*****************

* When compiled with ``-fsycl-pstl-offload`` option of Intel oneAPI DPC++/C++ compiler and with
  ``libstdc++`` version 8 or ``libc++``, ``oneapi::dpl::execution::par_unseq`` offloads
  standard parallel algorithms to the SYCL device similarly to ``std::execution::par_unseq``
  in accordance with the ``-fsycl-pstl-offload`` option value.
* For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``,
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
* For ``transform_exclusive_scan``, ``transform_inclusive_scan`` algorithms the result of the unary operation should be
  convertible to the type of the initial value if one is provided, otherwise it is convertible to the type of values
  in the processed data sequence: ``std::iterator_traits<IteratorType>::value_type``.
* ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with
  vector execution policies when building a program with GCC 10 and using ``-O0`` option.
* Compiling ``reduce`` and ``transform_reduce`` algorithms with the Intel DPC++ Compiler, versions 2021 and older,
  may result in a runtime error. To fix this issue, use an Intel DPC++ Compiler version 2022 or newer.
* When compiling on Windows, add the option ``/EHsc`` to the compilation command to avoid errors with oneDPL's experimental
  ranges API that uses exceptions.
* The ``using namespace oneapi;`` directive in a |onedpl_short| program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  the ``oneapi::dpl`` namespace, the shorter ``dpl`` namespace alias, or create your own alias.
* ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
* Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.
* ``exclusive_scan``, ``inclusive_scan``, ``exclusive_scan_by_segment``,
  ``inclusive_scan_by_segment``, ``transform_exclusive_scan``, ``transform_inclusive_scan``,
  when used with C++ standard policies, impose limitations on the initial value type if an
  initial value is provided, and on the value type of the input iterator if an initial value is
  not provided.
  Firstly, it must satisfy the ``DefaultConstructible`` requirements.
  Secondly, a default-constructed instance of that type should act as the identity element for the binary scan function.
* ``reduce_by_segment``, when used with C++ standard policies, imposes limitations on the value type.
  Firstly, it must satisfy the ``DefaultConstructible`` requirements.
  Secondly, a default-constructed instance of that type should act as the identity element for the binary reduction function.
* The initial value type for ``exclusive_scan``, ``inclusive_scan``, ``exclusive_scan_by_segment``,
  ``inclusive_scan_by_segment``, ``reduce``, ``reduce_by_segment``, ``transform_reduce``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan`` should satisfy the ``MoveAssignable`` and the ``CopyConstructible`` requirements.
* For ``max_element``, ``min_element``, ``minmax_element``, ``partial_sort``, ``partial_sort_copy``, ``sort``, ``stable_sort``
  the dereferenced value type of the provided iterators should satisfy the ``DefaultConstructible`` requirements.
* For ``remove``, ``remove_if``, ``unique`` the dereferenced value type of the provided
  iterators should be ``MoveConstructible``.
* The algorithms that process uninitialized storage: ``uninitialized_copy``, ``uninitialized_copy_n``, ``uninitialized_fill``, ``uninitialized_fill_n``, ``uninitialized_fill_n``, ``uninitialized_move``,
  ``uninitialized_move_n``, ``uninitialized_default_construct``, ``uninitialized_default_construct_n``, ``uninitialized_value_construct``, ``uninitialized_value_construct_n``
  should be called with a device policy when using device data and should be called with a host policy when using host data. Otherwise, the result is undefined.
* The algorithms that destroy data: ``destroy`` and ``destroy_n`` should be called with a host policy when using host data that was initialized on the host, and should be called with a device policy when using device data that was initialized on the device. Otherwise, the result is undefined.


.. _`Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`: https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-threading-building-blocks-release-notes.html
