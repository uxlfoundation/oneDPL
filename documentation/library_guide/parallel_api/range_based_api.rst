Experimental Range-Based API
############################

The ``<oneapi/dpl/ranges>`` header file contains experimental classes and functions that implement
the functionality similar to what is provided by the C++20 Ranges Library, yet only requires C++17.
This allows you to combine |onedpl_short| data parallel execution capabilities with some aspects
of modern range-based API. The functionality is only implemented for the device execution policies.

.. Note::
   The use of the experimental range-based API requires the C++ standard libraries
   coming with GCC 8.1 (or higher) or Clang 7 (or higher).

.. Warning::
   This experimental functionality will be gradually substituted by the
   :doc:`parallel range algorithms <parallel_range_algorithms>` and eventually discontinued.

Range Views
-----------

.. _viewable-ranges:

The following viewable ranges are defined in the ``oneapi::dpl::experimental::ranges`` namespace:

* ``views::iota``: A range factory that generates a sequence of elements by repeatedly incrementing an initial value.
* ``views::all``: A custom utility that represents a view of all or a part of ``sycl::buffer`` elements
  for reading and writing on a device.
* ``views::all_read``: A custom utility that represents a view of all or a part of ``sycl::buffer`` elements
  for reading on a device.
* ``views::all_write``: A custom utility that represents a view of all or a part of ``sycl::buffer`` elements
  for writing on a device.
* ``views::host_all``: A custom utility that represents a view of all or a part of ``sycl::buffer`` elements
  for reading and writing on the host.
* ``views::subrange``: A utility that represents a view of unified shared memory (USM) data range
  defined by two USM pointers.
* ``views::zip``: A custom range adaptor that produces one ``zip_view`` from other several views.
* ``views::transform``: A range adaptor that represents a view of an underlying sequence after applying
  a transformation to each element.
* ``views::reverse``: A range adaptor that produces a reversed sequence of elements provided by another view.
* ``views::take``: A range adaptor that produces a view of the first N elements from another view.
* ``views::drop``: A range adaptor that produces a view excluding the first N elements from another view.

Only these ranges, ``sycl::buffer``, and their combinations can be passed to the experimental range-based algorithms.

A ``sycl::buffer`` wrapped with ``views::all`` and similar utilities, ``views::subrange`` over USM, and ``views::iota``
are considered *base ranges*. The range adaptors may be combined into a pipeline with a base range at the beginning.
For example:

.. code:: cpp

    sycl::buffer<int> buf(data, sycl::range<1>(10));
    auto range_1 = views::iota(0, 10) | views::reverse;
    auto range_2 = views::all(buf) | views::take(10);

For ranges based on a SYCL buffer, data access is only permitted on a device, while ``size()`` and ``empty()``
methods are allowed to be called on both host and device.

Range-Based Algorithms
----------------------

The functions for experimental range based algorithms resemble the standard C++ parallel algorithm overloads
where all data sequences represented by ranges instead of iterators or iterator pairs, for example:

.. code:: cpp

   template <typename ExecutionPolicy, typename Range1, typename Range2>
   void copy(ExecutionPolicy&& exec, Range1&& source, Range2&& destination);

Note that ``source`` is used instead of two iterators to represent the input, and ``destination`` represents the output.

The following algorithms are available to use with the ranges. These algorithms are defined in the
``oneapi::dpl::experimental::ranges`` namespace and can only be invoked with device execution policies.
To use these algorithms, include both ``<oneapi/dpl/ranges>`` and ``<oneapi/dpl/execution>`` header files.

* ``adjacent_find``
* ``all_of``
* ``any_of``
* ``copy``
* ``copy_if``
* ``count``
* ``count_if``
* ``equal``
* ``exclusive_scan``
* ``find``
* ``find_if``
* ``find_if_not``
* ``find_end``
* ``find_first_of``
* ``for_each``
* ``inclusive_scan``
* ``is_sorted``
* ``is_sorted_until``
* ``min_element``
* ``max_element``
* ``merge``
* ``minmax_element``
* ``move``
* ``none_of``
* ``reduce``
* ``reduce_by_segment``
* ``remove``
* ``remove_if``
* ``remove_copy``
* ``remove_copy_if``
* ``replace``
* ``replace_if``
* ``replace_copy``
* ``replace_copy_if``
* ``reverse``
* ``reverse_copy``
* ``rotate_copy``
* ``search``
* ``sort``
* ``stable_sort``
* ``swap_ranges``
* ``transform``
* ``transform_reduce``
* ``transform_exclusive_scan``
* ``transform_inclusive_scan``
* ``unique``
* ``unique_copy``

Usage Example
-------------

.. code:: cpp

    namespace rangexp = oneapi::dpl::experimental::ranges;

    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        auto view = rangexp::views::all(A) | rangexp::views::reverse;
        auto range_res = rangexp::views::all_write(B);

        rangexp::copy(oneapi::dpl::execution::dpcpp_default, view, range_res);
    }
