Parallel Range Algorithms
#########################

C++20 introduces the `Ranges library <https://en.cppreference.com/w/cpp/ranges>`_ and
`range algorithms <https://en.cppreference.com/w/cpp/algorithm/ranges>`_ as a modern paradigm for expressing
generic operations on data sequences.

|onedpl_long| (|onedpl_short|) extends it with *parallel range algorithms*, which can be used with the standard range
classes to leverage |onedpl_short| ability of parallel execution on both the host computer and data parallel devices.

oneDPL only supports random access ranges, because they allow simultaneous constant-time access to elements
at any position in the range. This enables efficient workload distribution among multiple threads or processing units,
which is essential for achieving high performance in parallel execution.

.. Note::

  The use of parallel range algorithms requires C++20 and the C++ standard libraries coming with GCC 10 (or higher),
  Clang 16 (or higher) and Microsoft* Visual Studio* 2019 16.10 (or higher).

Supported Range Views
---------------------

`Views <https://en.cppreference.com/w/cpp/ranges/view>`_ are lightweight ranges typically used to describe
data transformation pipelines. The C++20 standard defines two categories of standard range views, called
*factories* and *adaptors*:

* A range factory generates its data elements on access via an index or an iterator to the range.
* A range adaptor transforms its underlying data range(s) or view(s) into a new view with modified behavior.

The following C++ standard random access adaptors and factories can be used with the |onedpl_short|
parallel range algorithms:

* ``std::ranges::views::all``: A range adaptor that returns a view that includes all elements of a range
  (only with standard-aligned execution policies).
* ``std::ranges::subrange``: A utility that produces a view from an iterator and a sentinel or from a range.
* ``std::span``: A view to a contiguous data sequence. 
* ``std::ranges::iota_view``: A range factory that generates a sequence of elements by repeatedly incrementing
  an initial value.
* ``std::ranges::single_view``: A view that contains exactly one element of a specified value.
* ``std::ranges::transform_view``: A range adaptor that produces a view that applies a transformation to each element
  of another view.
* ``std::ranges::reverse_view``: A range adaptor that produces a reversed sequence of elements provided by another view.
* ``std::ranges::take_view``: A range adaptor that produces a view of the first N elements from another view.
* ``std::ranges::drop_view``: A range adaptor that produces a view excluding the first N elements from another view.

Visit :doc:`pass_data_algorithms` for more information, especially on the :ref:`use of range views <use-range-views>`
with device execution policies.

Supported Algorithms
--------------------

The ``<oneapi/dpl/algorithm>`` header defines the parallel range algorithms in the ``namespace oneapi::dpl::ranges``.
All algorithms work with both standard-aligned (host) and device execution policies.

The ``ONEDPL_HAS_RANGE_ALGORITHMS`` :ref:`feature macro <feature-macros>` may be used to test for the presence of
parallel range algorithms.

.. _range-algorithms-202409L:

If ``ONEDPL_HAS_RANGE_ALGORITHMS`` is defined to ``202409L`` or a greater value, the following algorithms are provided:

* ``for_each``
* ``transform``
* ``find``
* ``find_if``
* ``find_if_not``
* ``adjacent_find``
* ``all_of``
* ``any_of``
* ``none_of``
* ``search``
* ``search_n``
* ``count``
* ``count_if``
* ``equal``
* ``sort``
* ``stable_sort``
* ``is_sorted``
* ``min_element``
* ``max_element``
* ``copy``
* ``copy_if``
* ``merge``

.. _range-algorithms-202505L:

If ``ONEDPL_HAS_RANGE_ALGORITHMS`` is defined to ``202505L`` or a greater value, the following algorithms are provided:

* ``fill``
* ``move``
* ``replace``
* ``replace_if``
* ``remove``
* ``remove_if``
* ``mismatch``
* ``minmax_element``
* ``minmax``
* ``min``
* ``max``
* ``find_first_of``
* ``find_end``
* ``is_sorted_until``

Usage Example for Parallel Range Algorithms
-------------------------------------------

.. code:: cpp

    {        
        std::vector<int> vec_in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<int> vec_out{vec_in.size()};

        auto view_in = std::ranges::views::all(vec_in) | std::ranges::views::reverse;
        oneapi::dpl::ranges::copy(oneapi::dpl::execution::par, view_in, vec_out);
    }
    {
        using usm_shared_allocator = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
        // Allocate for the queue used by the execution policy
        usm_shared_allocator alloc{oneapi::dpl::execution::dpcpp_default.queue()};

        std::vector<int, usm_shared_allocator> vec_in{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, alloc};
        std::vector<int, usm_shared_allocator> vec_out{vec_in.size(), alloc};

        auto view_in = std::ranges::subrange(vec_in.begin(), vec_in.end()) | std::ranges::views::reverse;
        oneapi::dpl::ranges::copy(oneapi::dpl::execution::dpcpp_default, view_in, std::span(vec_out));
    }

Implementation Notes
--------------------
The ``sort`` and ``stable_sort`` algorithms use ``std::swap`` and not ``std::ranges::iter_swap`` for swapping elements.
As a result, customizations targeting ``std::ranges::iter_swap`` will not be respected.

.. rubric:: See also:

:doc:`range_based_api`
