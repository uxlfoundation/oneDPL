.. SPDX-FileCopyrightText: 2019-2022 Intel Corporation
.. SPDX-FileCopyrightText: Contributors to the oneAPI Specification project.
..
.. SPDX-License-Identifier: CC-BY-4.0

Execution Policies
------------------

C++ Standard Aligned Execution Policies
+++++++++++++++++++++++++++++++++++++++

oneDPL has the set of execution policies semantically aligned with the `C++ Standard`_, 6th edition (C++20):

.. code:: cpp

  // Defined in <oneapi/dpl/execution>

  namespace oneapi {
    namespace dpl {
      namespace execution {

        class sequenced_policy { /*unspecified*/ };
        class parallel_policy { /*unspecified*/ };
        class parallel_unsequenced_policy { /*unspecified*/ };
        class unsequenced_policy { /*unspecified*/ };

        inline constexpr sequenced_policy seq { /*unspecified*/ };
        inline constexpr parallel_policy par { /*unspecified*/ };
        inline constexpr parallel_unsequenced_policy par_unseq { /*unspecified*/ };
        inline constexpr unsequenced_policy unseq { /*unspecified*/ };

      }
    }
  }

See "Execution policies" in the `C++ Standard`_ for more information.

.. note::
   The ``std::is_execution_policy`` type trait resolves to ``std::false_type`` for oneDPL execution policies.
   Implementations and programs should instead use the :ref:`oneDPL type trait <exec-policy-type-trait>`.

Device Execution Policies
+++++++++++++++++++++++++

A device execution policy represents a `SYCL`_ device and queue to run oneDPL algorithms.

.. code:: cpp

  // Defined in <oneapi/dpl/execution>

  namespace oneapi {
    namespace dpl {
      namespace execution {

        template <typename KernelName = /*unspecified*/>
        class device_policy;

        inline const device_policy<> dpcpp_default;

        template <typename KernelName = /*unspecified*/>
        device_policy<KernelName>
        make_device_policy( sycl::queue );

        template <typename KernelName = /*unspecified*/>
        device_policy<KernelName>
        make_device_policy( sycl::device );

        template <typename NewKernelName, typename OldKernelName>
        device_policy<NewKernelName>
        make_device_policy( const device_policy<OldKernelName>& = dpcpp_default );

      }
    }
  }

device_policy Class
^^^^^^^^^^^^^^^^^^^

.. code:: cpp

  template <typename KernelName = /*unspecified*/>
  class device_policy
  {
  public:
      using kernel_name = KernelName;

      device_policy();
      template <typename OtherName>
      device_policy( const device_policy<OtherName>& );
      explicit device_policy( sycl::queue );
      explicit device_policy( sycl::device );

      sycl::queue queue() const;
      operator sycl::queue() const;
  };

An object of the ``device_policy`` type is associated with a ``sycl::queue`` that is used
to run algorithms on a SYCL device. When an algorithm runs with ``device_policy``
it is capable of processing SYCL buffers (passed via ``oneapi::dpl::begin/end`` or other
:doc:`buffer wrappers <buffer_wrappers>`), data in the host memory and data in Unified Shared Memory (USM), including
device USM. Data placed in the host memory and USM can be passed to oneDPL algorithms
as pointers and random access iterators, including :doc:`oneDPL iterators <iterators>`. The way to transfer data from
the host memory to a device and back is unspecified; per-element data movement to/from a temporary storage is a possible
valid implementation.

The ``KernelName`` template parameter, also aliased as ``kernel_name`` within the class template,
is to explicitly provide a name for SYCL kernels executed by an algorithm the policy is passed to.

The ``device_policy`` type is copy-constructible, copy-assignable, move-constructible and move-assignable.
A policy instance constructed as or assigned a copy of another instance is associated with
a ``sycl::queue`` that compares equal, as defined by `SYCL`_, to the queue of that other instance.
A policy instance constructed as or assigned a move of another instance is associated with the queue
of that other instance without copying it. It is unspecified whether the moved-from policy instance
is associated with any queue after the move.

.. code:: cpp

  device_policy()

Construct a policy object associated with a queue created with the default device selector.
  
.. code:: cpp

  template <typename OtherName>
  device_policy( const device_policy<OtherName>& policy )

Construct a policy object as if by copying ``policy`` and changing
the kernel name to ``kernel_name`` defined for the new policy.

.. code:: cpp

  explicit device_policy( sycl::queue queue )

Construct a policy object associated with the given queue.

.. code:: cpp

  explicit device_policy( sycl::device device )

Construct a policy object associated with a queue created for the given device.

.. code:: cpp

  sycl::queue queue() const

Return the queue the policy is associated with.

.. code:: cpp

  operator sycl::queue() const

Allow implicit conversion of the policy to a ``sycl::queue`` object.

Predefined Device Policy
^^^^^^^^^^^^^^^^^^^^^^^^

``dpcpp_default`` is a predefined execution policy object to run algorithms on the default SYCL device.
It is a global immutable (``const``) instance of type ``device_policy<>``.
[*Note*: ``dpcpp_default`` can be copied but cannot be moved. -- *end note*]

make_device_policy Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``make_device_policy`` function templates simplify ``device_policy`` creation.

.. code:: cpp

  template <typename KernelName = /*unspecified*/>
  device_policy<KernelName>
  make_device_policy( sycl::queue queue )

Return a policy object associated with ``queue``, with a kernel name possibly provided
as the template argument, otherwise unspecified.

.. code:: cpp

  template <typename KernelName = /*unspecified*/>
  device_policy<KernelName>
  make_device_policy( sycl::device device )

Return a policy object to run algorithms on ``device``, with a kernel name possibly provided
as the template argument, otherwise unspecified.
  
.. code:: cpp

  template <typename NewKernelName, typename OldKernelName>
  device_policy<NewKernelName>
  make_device_policy( const device_policy<OldKernelName>& policy = dpcpp_default )

Return a policy object constructed from ``policy``, with a new kernel name provided as the template
argument. If no policy object is provided, the new policy is constructed from ``dpcpp_default``.

.. _exec-policy-type-trait:

Execution Policy Type Trait
+++++++++++++++++++++++++++

oneDPL provides type trait utilities to detect its execution policy types at compile time for the purpose of
function overload resolution:

.. code:: cpp

  // Defined in <oneapi/dpl/execution>

  namespace oneapi {
    namespace dpl {

      template <class T>
      struct is_execution_policy { /*see below*/ };

      template <class T>
      inline constexpr bool is_execution_policy_v = oneapi::dpl::is_execution_policy<T>::value;

      namespace execution {

        template <class T>
        struct is_execution_policy { /*see below*/ };

        template <class T>
        inline constexpr bool is_execution_policy_v = oneapi::dpl::execution::is_execution_policy<T>::value;

      }
    }
  }

``oneapi::dpl::is_execution_policy`` and ``oneapi::dpl::execution::is_execution_policy`` must be treated
as name aliases to the same class template. It is unspecified, which namespace contains the class template
definition and its specializations.

.. note::
   The ``oneapi::dpl::execution::is_execution_policy`` class originated in the oneDPL specification version 1.0,
   while ``oneapi::dpl::is_execution_policy`` has been added later to better align with the C++ standard.
   
   For new code, use of the type trait utilities in ``namespace oneapi::dpl`` is highly recommended. Those
   in ``namespace oneapi::dpl::execution`` are provided for backward compatibility and may be deprecated in the future.

``is_execution_policy<T>`` must have the characteristics of ``std::true_type`` if ``T`` is one of the above specified or
implementation-defined oneDPL execution policy types, otherwise it must have the characteristics of ``std::false_type``.
Following the C++ Standard, ``is_execution_policy<T>`` does not automatically strip references and cv-qualifiers from
its template argument. [*Note*: Use it with ``std::decay_t<T>`` or similar type transformation utilities. -- *end note*]

.. _`C++ Standard`: https://isocpp.org/std/the-standard
.. _`SYCL`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
