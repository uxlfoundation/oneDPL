Execution Policies
##################

The implementation supports the device execution policies used to run the massive parallel
computational model for heterogeneous systems. The policies are specified in
the |onedpl_long| (|onedpl_short|) section of the `oneAPI Specification
<https://spec.oneapi.io/versions/latest/elements/oneDPL/source/parallel_api.html#dpc-execution-policy>`_.

For any of the implemented algorithms, pass one of the execution policy objects as the first
argument in a call to specify the desired execution behavior. The policies have
the following meaning:

================================= ==============================
Execution Policy Value            Description
================================= ==============================
``seq``                           Sequential execution.
--------------------------------- ------------------------------
``unseq``                         Unsequenced SIMD execution. This policy requires that
                                  all functions provided are SIMD-safe.
--------------------------------- ------------------------------
``par``                           Parallel execution by multiple threads.
--------------------------------- ------------------------------
``par_unseq``                     Combined effect of ``unseq`` and ``par``.
--------------------------------- ------------------------------
``dpcpp_default``                 Massive parallel execution on devices using |dpcpp_short|.
--------------------------------- ------------------------------
``dpcpp_fpga``                    Massive parallel execution on FPGA devices.
================================= ==============================

The implementation is based on Parallel STL from the
`LLVM Project <https://github.com/llvm/llvm-project/tree/main/pstl>`_.

|onedpl_short| supports two parallel backends for execution with ``par`` and ``par_unseq`` policies:

#. TBB backend (enabled by default) uses |onetbb_long| or |tbb_long| for parallel execution.

#. OpenMP backend uses OpenMP* pragmas for parallel execution. Visit
   :doc:`Macros <../macros>` for the information how to enable the OpenMP backend.

Follow these steps to add Parallel API to your application:

#. Add ``#include <oneapi/dpl/execution>`` to your code.
   Then include one or more of the following header files, depending on the algorithms you
   intend to use:

   #. ``#include <oneapi/dpl/algorithm>``
   #. ``#include <oneapi/dpl/numeric>``
   #. ``#include <oneapi/dpl/memory>``

#. Pass a |onedpl_short| execution policy object, defined in the ``oneapi::dpl::execution``
   namespace, to a parallel algorithm.
#. Use the C++ standard execution policies:

   #. Compile the code with options that enable OpenMP parallelism and/or vectorization pragmas.
   #. Link with the |onetbb_long| or |tbb_long| dynamic library for TBB-based parallelism.

#. Use the device execution policies:

   #. Compile the code with options that enable support for SYCL 2020.

Use the C++ Standard Execution Policies
=======================================

Example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <vector>

  int main()
  {
      std::vector<int> data( 1000 );
      std::fill(oneapi::dpl::execution::par_unseq, data.begin(), data.end(), 42);
      return 0;
  }

Use the Device Execution Policies
========================================

The device execution policy specifies where a parallel algorithm runs.
It encapsulates a SYCL device or queue and allows you to
set an optional kernel name. Device execution policies can be used with all
standard C++ algorithms that support execution policies.

To create a policy object, you may use one of the following constructor arguments:

* A SYCL queue
* A SYCL device
* A SYCL device selector
* An existing policy object with a different kernel name

A kernel name is set with a policy template argument.
Providing a kernel name for a policy is optional, if your compiler supports implicit
names for SYCL kernel functions. The |dpcpp_cpp| supports it by default;
for other compilers it may need to be enabled with compilation options such as
``-fsycl-unnamed-lambda``. Refer to your compiler documentation for more information.

The ``oneapi::dpl::execution::dpcpp_default`` object is a predefined immutable object of
the ``device_policy`` class. It is created with a default kernel name and uses a default queue.
Use it to construct customized policy objects or pass directly when invoking an algorithm.

If ``dpcpp_default`` is passed directly to more than one algorithm, you must ensure that the
compiler you use supports implicit kernel names (see above) and this option is turned on.

The ``make_device_policy`` function templates simplify ``device_policy`` creation.

Usage Examples
==============

The code examples below assume you are ``using namespace oneapi::dpl::execution;``
and ``using namespace sycl;`` directives when referring to policy classes and functions:

.. code:: cpp

   auto policy_a = device_policy<class PolicyA> {};
   std::for_each(policy_a, ...);

.. code:: cpp

  auto policy_b = device_policy<class PolicyB> {device{gpu_selector_v}};
  std::for_each(policy_b, ...);

.. code:: cpp

  auto policy_c = device_policy<class PolicyC> {device{cpu_selector_v}};
  std::for_each(policy_c, ...);

.. code:: cpp

  auto policy_d = make_device_policy<class PolicyD>(dpcpp_default);
  std::for_each(policy_d, ...);

.. code:: cpp

  auto policy_e = make_device_policy(queue{property::queue::in_order()});
  std::for_each(policy_e, ...);

Use the FPGA Policy
===================

The ``fpga_policy`` class is a device policy tailored to achieve
better performance of parallel algorithms on FPGA hardware devices.

Use the policy when you run the application on a FPGA hardware device or FPGA emulation device
with the following steps:

#. Define the ``ONEDPL_FPGA_DEVICE`` macro to run on FPGA devices and the ``ONEDPL_FPGA_EMULATOR``
   to run on FPGA emulation devices.
#. Add ``#include <oneapi/dpl/execution>`` to your code.
#. Create a policy object by providing an unroll factor (see the **Note** below),
   a class type for a unique kernel name as template arguments (both optional), and one of the
   following constructor arguments:

   #. A SYCL queue constructed for the
      `FPGA Selector <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_fpga_device_selector.asciidoc>`_
      (the behavior is undefined with any other queue).
   #. An existing FPGA policy object with a different kernel name and/or unroll factor.

#. Pass the created policy object to a parallel algorithm.

The default constructor of ``fpga_policy`` wraps a SYCL queue created
for ``fpga_selector``, or for ``fpga_emulator_selector``
if the ``ONEDPL_FPGA_EMULATOR`` is defined.

``oneapi::dpl::execution::dpcpp_fpga`` is a predefined immutable object of
the ``fpga_policy`` class created with a default unroll factor and a default kernel name.
Use it to create customized policy objects or pass directly when invoking an algorithm.

.. Note::

   Specifying the unroll factor for a policy enables loop unrolling in the implementation of
   |onedpl_short| algorithms. The default value is 1.
   To find out how to choose a more precise value, refer to the |unroll_pragma|_
   and |loop_analysis|_ content in the |fpga_handbook|_.

The ``make_fpga_policy`` function templates simplify ``fpga_policy`` creation.

FPGA Policy Usage Examples
==========================

The code below assumes you have added ``using namespace oneapi::dpl::execution;`` for policies and
``using namespace sycl;`` for queues and device selectors:

.. code:: cpp

  constexpr auto unroll_factor = 8;
  auto fpga_policy_a = fpga_policy<unroll_factor, class FPGAPolicyA>{};
  auto fpga_policy_b = make_fpga_policy(queue{intel::fpga_selector{}});
  auto fpga_policy_c = make_fpga_policy<unroll_factor, class FPGAPolicyC>();


Error Handling with Device Execution Policies
====================================================

The SYCL error handling model supports two types of errors: Synchronous errors cause the SYCL host
runtime libraries throw exceptions. Asynchronous errors may only be processed in a user-supplied error handler
associated with a SYCL queue.

For algorithms executed with device policies, handling all errors, synchronous or asynchronous, is a
responsibility of the caller. Specifically:

* No exceptions are thrown explicitly by algorithms.
* Exceptions thrown by runtime libraries at the host CPU, including SYCL synchronous exceptions,
  are passed through to the caller.
* SYCL asynchronous errors are not handled.

To process SYCL asynchronous errors, the queue associated with a device policy must be
created with an error handler object. The predefined policy objects (``dpcpp_default``, etc.) have
no error handlers; do not use them if you need to process asynchronous errors.