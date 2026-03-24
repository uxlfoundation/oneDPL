.. SPDX-FileCopyrightText: 2019-2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

SYCL Kernels API
================

Supported C++ Standard Library APIs and Algorithms
++++++++++++++++++++++++++++++++++++++++++++++++++

oneDPL defines a subset of the `C++ Standard`_ library APIs for use in SYCL kernels. These APIs
can be employed in the kernels similarly to how they are employed in code for a typical
CPU-based platform.

.. _`C++ Standard`: https://isocpp.org/std/the-standard


Random Number Generation
++++++++++++++++++++++++

oneDPL provides a subset of the standard C++ pseudo-random number generation functionality
suitable to use within SYCL kernels. The APIs are defined in the ``<oneapi/dpl/random>`` header.

Supported Functionality
-----------------------

- Engine class templates:
   - ``linear_congruential_engine``
   - ``subtract_with_carry_engine``
   - ``philox_engine``
- Engine adaptor class templates:
   - ``discard_block_engine``
- Engines and engine adaptors with predefined parameters:
   - ``minstd_rand0``
   - ``minstd_rand``
   - ``ranlux24_base``
   - ``ranlux48_base``
   - ``ranlux24``
   - ``ranlux48``
   - ``philox4x32``
   - ``philox4x64``
- Distribution class templates:
   - ``uniform_int_distribution``
   - ``uniform_real_distribution``
   - ``normal_distribution``
   - ``exponential_distribution``
   - ``bernoulli_distribution``
   - ``geometric_distribution``
   - ``weibull_distribuion``
   - ``lognormal_distribution``
   - ``cauchy_distribution``
   - ``extreme_value_distribution``

.. note::
   The ``philox_engine`` class template and the corresponding predefined engines match the respective API
   in the working draft of the next C++ standard edition (C++26).

``linear_congruential_engine``, ``subtract_with_carry_engine``, and ``philox_engine`` satisfy the uniform random bit generator requirements.

Limitations
-----------

The following deviations from the `C++ Standard`_ may apply:

- ``random_device`` and ``seed_seq`` classes and related APIs in other classes are not required;
- specifying the size of a random number engine's state is not required;
- distributions are only required to operate with floating point types applicable to supported SYCL devices.

Extensions
----------

As an extension to the `C++ Standard`_, ``sycl::vec<Type, N>`` can be used as the data type template parameter for
engines, engine adaptors, and distributions, where ``Type`` is one of data types supported by the corresponding
class template in the standard. For such template instantiations, the ``result_type`` is also defined to
``sycl::vec<Type, N>``.

Engines, engine adaptors, and distributions additionally define ``scalar_type``, equivalent to the following:

- ``using scalar_type = typename result_type::element_type;`` if ``result_type`` is ``sycl::vec<Type, N>``,
- otherwise, ``using scalar_type = result_type;``

The ``scalar_type`` is used instead of ``result_type`` in all contexts where a scalar data type is expected, including

- the type of configuration parameters and properties,
- the seed value type,
- the input parameters of constructors,
- the return value type of ``min()`` and ``max()`` methods, etc.

Since ``scalar_type`` is the same as ``result_type`` except for template instantiations with ``sycl::vec``,
class templates still meet the applicable requirements of the `C++ Standard`_.

When instantiated with ``sycl::vec<Type,N>``, ``linear_congruential_engine``, ``subtract_with_carry_engine``, and ``philox_engine`` may not
formally satisfy the uniform random bit generator requirements defined by the `C++ Standard`_. Instead, the following
alternative requirements apply: for an engine object ``g`` of type ``G``,

- ``G::scalar_type`` is an unsigned integral type same as ``sycl::vec<Type,N>::element_type``,
- ``G::min()`` and ``G::max()`` return a value of ``G::scalar_type``,
- for each index ``i`` in the range [``0``, ``N``), ``G::min() <= g()[i]`` and ``g()[i] <= G::max()``.

Effectively, these engines satisfy the standard *uniform random bit generator* requirements for each element
of a ``sycl::vec`` returned by their ``operator()``.

Similarly, for a distribution object ``d`` of a type ``D`` that is a template instantiated with ``sycl::vec<Type,N>``:

- ``D::scalar_type`` is the same as ``sycl::vec<Type,N>::element_type``,
- ``D::min()`` and ``D::max()`` return a value of ``D::scalar_type``, and ``D::min() <= D::max()``,
- ``operator()`` of a distribution returns a ``sycl::vec<Type,N>`` filled with random values
  in the closed interval ``[D::min(), D::max()]``;

The following engines and engine adaptors with predefined parameters are defined:

.. code:: cpp

    template <int N>
    using minstd_rand0_vec =
        linear_congruential_engine<sycl::vec<uint_fast32_t, N>, 16807, 0, 2147483647>;

    template <int N>
    using minstd_rand_vec =
        linear_congruential_engine<sycl::vec<uint_fast32_t, N>, 48271, 0, 2147483647>;

    template <int N>
    using ranlux24_base_vec = subtract_with_carry_engine<sycl::vec<uint_fast32_t, N>, 24, 10, 24>;

    template <int N>
    using ranlux48_base_vec = subtract_with_carry_engine<sycl::vec<uint_fast64_t, N>, 48, 5, 12>;

    template <int N>
    using ranlux24_vec = discard_block_engine<ranlux24_base_vec<N>, 223, 23>;

    template <int N>
    using ranlux48_vec = discard_block_engine<ranlux48_base_vec<N>, 389, 11>;

    template <int N>
    using philox4x32_vec = philox_engine<sycl::vec<uint_fast32_t, N>, 32, 4, 10, 0xCD9E8D57,
                                         0x9E3779B9, 0xD2511F53, 0xBB67AE85>;

    template <int N>
    using philox4x64_vec = philox_engine<sycl::vec<uint_fast64_t, N>, 64, 4, 10, 0xCA5A826395121157,
                                         0x9E3779B97F4A7C15, 0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;

Except for producing a ``sycl::vec`` of random values per invocation, the behavior of these engines is equivalent to
the corresponding scalar engines, as described in the following table:

.. container:: tablenoborder

      .. list-table::
         :header-rows: 1

         * -     Engines and engine adaptors based on ``sycl::vec<>``
           -     C++ standard analogue
           -     The 10000th scalar random value consecutively produced by a default-constructed object
         * -     ``minstd_rand0_vec``
           -     ``minstd_rand0``
           -     1043618065
         * -     ``minstd_rand_vec``
           -     ``minstd_rand``
           -     399268537
         * -     ``ranlux24_base_vec``
           -     ``ranlux24_base``
           -     7937952
         * -     ``ranlux48_base_vec``
           -     ``ranlux48_base``
           -     61839128582725
         * -     ``ranlux24_vec``
           -     ``ranlux24``
           -     9901578
         * -     ``ranlux48_vec``
           -     ``ranlux48``
           -     1112339016
         * -     ``philox4x32_vec``
           -     ``philox4x32``
           -     1955073260
         * -     ``philox4x64_vec``
           -     ``philox4x64``
           -     3409172418970261260

Function Objects
++++++++++++++++

The oneDPL function objects are defined in the ``<oneapi/dpl/functional>`` header.

.. code:: cpp

    namespace oneapi {
    namespace dpl {
        struct identity
        {
            template <typename T>
            constexpr T&&
            operator()(T&& t) const noexcept;
        };
    }
    }

The ``oneapi::dpl::identity`` class implements an identity operation. Its function operator
receives an instance of a type and returns the argument unchanged.
