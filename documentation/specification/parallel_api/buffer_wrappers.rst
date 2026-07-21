.. SPDX-FileCopyrightText: 2019-2022 Intel Corporation
.. SPDX-FileCopyrightText: Contributors to the oneAPI Specification project.
..
.. SPDX-License-Identifier: CC-BY-4.0

Buffer Wrappers
---------------

A *buffer wrapper* is an iterator-like type for processing data in `SYCL`_ buffers with oneDPL algorithms.
It is either the type of a *buffer position object* returned by
``oneapi::dpl::begin`` or ``oneapi::dpl::end`` or a type composed of at least one buffer position object,
with the same capabilities and limitations as buffer position objects.
Buffer wrappers can be used as source types for :doc:`oneDPL iterators <iterators>`, and the resulting type is also a
buffer wrapper.

Buffer wrappers substitute for iterators as arguments and return values of oneDPL algorithms, however they cannot be
used as iterators in other contexts, including dereference, as these types do not fully satisfy the C++ standard
requirements for an iterator.

Buffer Position Objects
+++++++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/iterator>

  namespace oneapi {
    namespace dpl {

      template < typename T, typename AllocatorT, typename TagT >
      /*unspecified*/ begin( sycl::buffer<T, /*dim=*/1, AllocatorT> buf,
                             TagT tag = sycl::read_write );

      template < typename T, typename AllocatorT, typename TagT >
      /*unspecified*/ begin( sycl::buffer<T, /*dim=*/1, AllocatorT> buf,
                             TagT tag, sycl::property::no_init );

      template < typename T, typename AllocatorT >
      /*unspecified*/ begin( sycl::buffer<T, /*dim=*/1, AllocatorT> buf,
                             sycl::property::no_init );


      template < typename T, typename AllocatorT, typename TagT >
      /*unspecified*/ end( sycl::buffer<T, /*dim=*/1, AllocatorT> buf,
                           TagT tag = sycl::read_write );

      template < typename T, typename AllocatorT, typename TagT >
      /*unspecified*/ end( sycl::buffer<T, /*dim=*/1, AllocatorT> buf,
                           TagT tag, sycl::property::no_init );

      template < typename T, typename AllocatorT >
      /*unspecified*/ end( sycl::buffer<T, /*dim=*/1, AllocatorT> buf,
                           sycl::property::no_init );

    }
  }

``oneapi::dpl::begin`` and ``oneapi::dpl::end`` are helper functions
for passing `SYCL`_ buffers to oneDPL algorithms.
These functions accept a SYCL buffer and return a *buffer position object*,
which type satisfies the following requirements but is otherwise unspecified:

- It is copy-constructible and copy-assignable.
- It is comparable with ``operator==`` and ``operator!=``.
- The expressions ``a + n`` and ``a - n``, where ``a`` is a buffer position object and ``n``
  is an integer value, are valid and evaluate to a buffer position object such that
  the corresponding position in the buffer follows (precedes) that of ``a`` by ``n``.
  If this new position is out of the buffer bounds, the behavior is undefined.
- The expression ``a - b``, where ``a`` and ``b`` are buffer position objects,
  is valid and evaluates to an integer value ``n`` such that ``a == b + n``.
- It provides the ``get_buffer()`` method that for a buffer position object returns the SYCL buffer,
  which the object was built over.

.. note:: 
   Only buffer position objects are required to support the ``get_buffer()`` method, other buffer wrappers
   composed of buffer position objects are not required to support this method.

If these operators and expressions are applied to buffer position objects that are not built
over the same SYCL buffer, the behavior is undefined.

When invoking an algorithm, the buffer passed to ``begin`` should be the same
as the buffer passed to ``end``. Otherwise, the behavior is undefined.

An algorithm may return a buffer position object, which can then be compared and used
in expressions with other objects built over the same buffer.

.. code:: cpp

   namespace dpl = oneapi::dpl; // see "Namespaces"
   sycl::buffer buf {/*...*/};
   auto pos = dpl::find(dpl::execution::dpcpp_default, dpl::begin(buf), dpl::end(buf), value);
   int value_index = (pos == dpl::end(buf)) ? -1 : (pos - dpl::begin(buf));


SYCL deduction tags (the ``TagT`` parameters) and ``sycl::property::no_init`` 
allow to specify an access mode to be used by algorithms for accessing a SYCL buffer.
The mode serves as a hint, and can be overridden depending on semantics of the algorithm.
When invoking an algorithm, the same access mode arguments should be used
for ``begin`` and ``end``. Otherwise, the behavior is undefined.

.. code:: cpp

   namespace dpl = oneapi::dpl;
   sycl::buffer buf {/*...*/};
   auto policy = dpl::execution::dpcpp_default;

   auto buf_begin = dpl::begin(buf, sycl::write_only);
   auto buf_end_1 = dpl::end(buf, sycl::write_only); // arguments match begin()
   dpl::fill(policy, buf_begin, buf_end_1, 42);      // OK
   auto buf_end_2 = dpl::end(buf, sycl::write_only, sycl::no_init); // arguments do not match
   dpl::fill(policy, buf_begin, buf_end_2, 42);                     // undefined behavior

.. _`SYCL`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
