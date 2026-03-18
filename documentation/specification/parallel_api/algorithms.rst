.. SPDX-FileCopyrightText: 2019-2022 Intel Corporation
.. SPDX-FileCopyrightText: Contributors to the oneAPI Specification project.
..
.. SPDX-License-Identifier: CC-BY-4.0

Parallel Algorithms
-------------------

oneDPL parallel algorithms are function templates analogous to the algorithms with execution policies defined
in the `C++ Standard`_, 5th and 6th editions (C++17/20), as well as additional non-standard function templates.

oneDPL parallel algorithms reside in ``namespace oneapi::dpl``.
Standard-aligned algorithms are defined in the ``<oneapi/dpl/algorithm>``, ``<oneapi/dpl/numeric>``,
and ``<oneapi/dpl/memory>`` header files, aligned with how the `C++ Standard`_ places the respective functions
into the standard header files.

The parallel algorithms execute according to a oneDPL execution policy supplied as the first argument.

Where applicable, oneDPL supports
`list initialization of value parameters <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2248r8.html>`_
of the algorithms, as in the working draft of the next C++ standard edition (C++26).

Additional Algorithms
+++++++++++++++++++++

In addition to the standard-aligned parallel algorithms, oneDPL provides the following algorithm functions.

For compatibility with the previous versions of the specification, besides their indicated header files
all these algorithms are defined in ``<oneapi/dpl/algorithm>``. [*Note*: This may be deprecated in the future. -- *end note*]

.. code:: cpp

    // Defined in <oneapi/dpl/numeric>
    template<typename Policy, typename InputKeyIt, typename InputValueIt,
        typename OutputValueIt,
        typename T = typename std::iterator_traits<InputValueIt>::value_type,
        typename BinaryPred =
            std::equal_to<typename std::iterator_traits<InputKeyIt>::value_type>,
        typename BinaryOp =
            std::plus<typename std::iterator_traits<InputValueIt>::value_type>>
    OutputValueIt
    exclusive_scan_by_segment(Policy&& policy, InputKeyIt keys_first,
        InputKeyIt keys_last, InputValueIt values_first, OutputValueIt values_result,
        T initial_value = 0,
        BinaryPred binary_pred =
            std::equal_to<typename std::iterator_traits<InputKeyIt>::value_type>(),
        BinaryOp binary_op =
            std::plus<typename std::iterator_traits<InputValueIt>::value_type>());

``oneapi::dpl::exclusive_scan_by_segment`` performs partial prefix scans by applying the
``binary_op`` operation to a sequence of values. Each partial scan applies to a contiguous
subsequence determined by the keys associated with the values being equal according to the
``binary_pred`` predicate, and the first element of each scan is the initial value provided.
The return value is an iterator targeting the end of the result sequence.

The initial value used if one is not provided is an instance of the ``value_type`` of the
``InputValueIt`` iterator type initialized to 0. If no binary predicate is provided for the
comparison of keys an instance of ``std::equal_to`` with the ``value_type`` of the
``InputKeyIt`` iterator type is used. Finally, an instance of ``std::plus`` with the
``value_type`` of the ``InputValueIt`` iterator type is used if no binary operator is
provided to combine the elements of the value subsequences.

.. code:: cpp

    // Defined in <oneapi/dpl/numeric>
    template<typename Policy, typename InputKeyIt, typename InputValueIt,
        typename OutputValueIt,
        typename BinaryPredcate =
            std::equal_to<typename std::iterator_traits<InputKeyIt>::value_type,
        typename BinaryOp =
            std::plus<typename std::iterator_traits<InputValueIt>::value_type>>
    OutputValueIt
    inclusive_scan_by_segment(Policy&& policy, InputKeyIt keys_first,
        InputKeyIt keys_last, InputValueIt values_first, OutputValueIt values_result
        BinaryPred binary_pred =
            std::equal_to<typename std::iterator_traits<InputKeyIt>::value_type>(),
        BinaryOp binary_op =
            std::plus<typename std::iterator_traits<InputValueIt>::value_type>());

``oneapi::dpl::inclusive_scan_by_segment`` performs partial prefix scans by applying the
``binary_op`` operation to a sequence of values. Each partial scan applies to a contiguous
subsequence determined by the keys associated with the values being equal according to the
``binary_pred`` predicate. The return value is an iterator targeting the end of the result
sequence.

If no binary predicate is provided for the comparison of keys an instance of ``std::equal_to``
with the ``value_type`` of the ``InputKeyIt`` iterator type is used. An instance of
``std::plus`` with the ``value_type`` of the ``InputValueIt`` iterator type is used if
no binary operator is provided to combine the elements of the value subsequences.

.. code:: cpp

    // Defined in <oneapi/dpl/numeric>
    template<typename Policy, typename InputKeyIt, typename InputValueIt,
        typename OutputKeyIt, typename OutputValueIt,
        typename BinaryPredcate =
            std::equal_to<typename std::iterator_traits<InputKeyIt>::value_type,
        typename BinaryOp =
            std::plus<typename std::iterator_traits<InputValueIt>::value_type>>
    std::pair<OutputKeyIt,OutputValueIt>
    reduce_by_segment(Policy&& policy, InputKeyIt keys_first, InputKeyIt keys_last,
        InputValueIt values_first, OutputKeyIt keys_result,
        OutputValueIt values_result,
        BinaryPred binary_pred =
            std::equal_to<typename std::iterator_traits<InputKeyIt>::value_type>(),
        BinaryOp binary_op =
            std::plus<typename std::iterator_traits<InputValueIt>::value_type>());

``oneapi::dpl::reduce_by_segment`` performs partial reductions on a sequence of values. Each
reduction is computed with the ``binary_op`` operation for a contiguous subsequence of values
determined by the associated keys being equal according to the ``binary_pred`` predicate.
For each subsequence the first of the equal keys is stored into ``keys_result`` and the computed
reduction is stored into ``values_result``. The return value is a pair of
iterators holding the end of the resulting sequences.

If no binary predicate is provided for the comparison of keys an instance of ``std::equal_to``
with the ``value_type`` of the ``InputKeyIt`` iterator type is used. An instance of
``std::plus`` with the ``value_type`` of the ``InputValueIt`` iterator type is used to
combine the values in each subsequence identified if a binary operator is not provided.

.. code:: cpp

    // Defined in <oneapi/dpl/algorithm>
    template<typename Policy, typename InputIt1, typename InputIt2, typename OutputIt,
        typename Comparator =
            std::less<typename std::iterator_traits<InputIt>::value_type>>
    OutputIt
    binary_search(Policy&& policy, InputIt1 start, InputIt1 end,
        InputIt2 value_first, InputIt2 value_last, OutputIterator result,
        Comparator comp =
            std::less<typename std::iterator_traits<InputIt1>::value_type>());

``oneapi::dpl::binary_search`` performs a binary search over the data in ``[start, end)``
for each value in ``[value_first, value_last)``. If the value exists in the data searched then
the corresponding element in ``[result, result + std::distance(value_first, value_last))`` is set to
true, otherwise it is set to false.

The elements of ``[start, end)`` must be partitioned with respect to the comparator used,
or with respect to ``std::less`` if no comparator is provided. For all
elements ``e`` in ``[start, end)`` and a given search value ``v`` in ``[value_first, value_last)``,
``comp(e, v)`` implies ``!comp(v, e)``.

.. code:: cpp

    // Defined in <oneapi/dpl/algorithm>
    template<typename Policy, typename InputIt1, typename InputIt2, typename OutputIt,
        typename Comparator =
            std::less<typename std::iterator_traits<InputIt>::value_type>>
    OutputIt
    lower_bound(Policy&& policy, InputIt1 start, InputIt1 end,
        InputIt2 value_first, InputIt2 value_last, OutputIterator result,
        Comparator comp =
            std::less<typename std::iterator_traits<InputIt1>::value_type>());

``oneapi::dpl::lower_bound`` performs a binary search over the data in ``[start, end)`` for
each value in ``[value_first, value_last)`` to find the lowest index at which the search value
could be inserted in ``[start, end)`` without violating the ordering defined by the comparator
provided. That lowest index is then assigned to the corresponding element in
``[result, result + std::distance(value_first, value_last))``.

The elements of ``[start, end)`` must be partitioned with respect to the comparator used,
or with respect to ``std::less`` if no comparator is provided.

.. code:: cpp

    // Defined in <oneapi/dpl/algorithm>
    template<typename Policy, typename InputIt1, typename InputIt2, typename OutputIt,
        typename Comparator =
            std::less<typename std::iterator_traits<InputIt>::value_type>>
    OutputIt
    upper_bound(Policy&& policy, InputIt1 start, InputIt1 end,
        InputIt2 value_first, InputIt2 value_last, OutputIterator result,
        Comparator comp =
            std::less<typename std::iterator_traits<InputIt1>::value_type>());

``oneapi::dpl::upper_bound`` performs a binary search over the data in ``[start, end)``
for each value in ``[value_first, value_last)`` to find the highest index at which the search
value could be inserted in ``[start, end)`` without violating the ordering defined by the
comparator provided. That highest index is then assigned to the corresponding element in
``[result, result + std::distance(value_first, value_last))``.

The elements of ``[start, end)`` must be partitioned with respect to the comparator used,
or with respect to ``std::less`` if no comparator is provided.

.. code:: cpp

    // Defined in <oneapi/dpl/algorithm>
  template <typename Policy, typename InputIt, typename OutputIt, typename UnaryOp,
      typename UnaryPredicate>
  OutputIt
  transform_if(Policy&& policy, InputIt start, InputIt end, OutputIt result, UnaryOp op,
      UnaryPredicate pred);                                                               // (1)

  template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt, 
      typename BinaryOp, typename BinaryPredicate>
  OutputIt
  transform_if(Policy&& policy, InputIt1 start1, InputIt1 end1, InputIt2 start2, OutputIt result,
      BinaryOp op, BinaryPredicate pred);                                                 // (2)

``oneapi::dpl::transform_if`` applies a given function to the elements of the input sequence(s) that
satisfy a given predicate, and stores the result to the output. Depending on the arguments, the algorithm:

1. Evaluates the unary predicate ``pred`` for each position ``i`` of the sequence
   ``[start, end)`` and if ``pred(start[i]) == true``, it performs the unary operation
   ``op(start[i])`` and stores the result into ``result[i]``. If
   ``pred(start[i]) == false``, the data element ``result[i]`` is not modified from its
   initial value. The return value is an iterator targeting past the last considered element of
   the output sequence, that is, ``result + (end - start)``.

2. Evaluates the binary predicate ``pred`` for each position ``i`` of the sequence
   ``[start1, end1)`` and if ``pred(start1[i], start2[i]) == true``, it performs the
   binary operation ``op(start1[i], start2[i])`` and stores the result into ``result[i]``.
   If ``pred(start1[i], start2[i]) == false``, the data element ``result[i]`` is not
   modified from its initial value. The return value is an iterator targeting past the last
   considered element of the output sequence, that is, ``result + (end1 - start1)``.

.. code:: cpp

    // Defined in <oneapi/dpl/algorithm>
    template<typename Policy, typename KeyIt, typename ValueIt,
        typename Comparator = std::less<typename std::iterator_traits<KeyIt>::value_type>>
    void
    sort_by_key(Policy&& policy, KeyIt keys_first, KeyIt keys_last,
        ValueIt values_first,
        Comparator comp = std::less<typename std::iterator_traits<KeyIt>::value_type>());

``oneapi::dpl::sort_by_key`` sorts the sequence of keys ``[keys_first, keys_last)``
and simultaneously permutes associated values at the same positions in the range
``[values_first, values_first + std::distance(keys_first, keys_last))``
to match the order of the sorted keys. That is, a key and its associated value
will have the same index in their respective sequences after sorting.

Keys are sorted with respect to the provided comparator object ``comp``. That means, for any
two iterators ``i`` and ``j`` to the sorted sequence of keys such that ``i`` precedes ``j``,
``comp(*j, *i) == false``.
If no ``comp`` object is provided, keys are sorted with respect to ``std::less``.

Sorting is unstable. That means, keys which do not precede one another with respect to the given
comparator and their associated values might be ordered arbitrarily relative to each other.

``KeyIt`` and ``ValueIt`` must satisfy the requirements of ``ValueSwappable``,
and ``Comparator`` must satisfy the requirements for the ``Compare`` parameter of ``std::sort``,
as defined by the `C++ Standard`_.

.. code:: cpp

    // Defined in <oneapi/dpl/algorithm>
    template<typename Policy, typename KeyIt, typename ValueIt,
        typename Comparator = std::less<typename std::iterator_traits<KeyIt>::value_type>>
    void
    stable_sort_by_key(Policy&& policy, KeyIt keys_first, KeyIt keys_last,
        ValueIt values_first,
        Comparator comp = std::less<typename std::iterator_traits<KeyIt>::value_type>());

``oneapi::dpl::stable_sort_by_key`` sorts the sequence of keys ``[keys_first, keys_last)``
and simultaneously permutes associated values at the same positions in the range
``[values_first, values_first + std::distance(keys_first, keys_last))``
to match the order of the sorted keys. That is, a key and its associated value
will have the same index in their respective sequences after sorting.

Keys are sorted with respect to the provided comparator object ``comp``. That means, for any
two iterators ``i`` and ``j`` to the sorted sequence of keys such that ``i`` precedes ``j``,
``comp(*j, *i) == false``.
If no ``comp`` object is provided, keys are sorted with respect to ``std::less``.

Sorting is stable. That means, keys which do not precede one another with respect to the given
comparator and their associated values maintain the relative order as in the original sequences.

``KeyIt`` and ``ValueIt`` must satisfy the requirements of ``ValueSwappable``,
and ``Comparator`` must satisfy the requirements for the ``Compare`` parameter of ``std::sort``,
as defined by the `C++ Standard`_.

.. code:: cpp

    // Defined in <oneapi/dpl/numeric>
    template <typename Policy, typename InputIt, typename Size, typename ValueType,
        typename OutputIt>
    OutputIt
    histogram(Policy&& exec, InputIt start, InputIt end, Size num_intervals,
        ValueType first_interval_begin, ValueType last_interval_end, OutputIt histogram_first); // (1)

    template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt>
    OutputIt
    histogram(Policy&& exec, InputIt1 start, InputIt1 end, InputIt2 boundary_start,
              InputIt2 boundary_end, OutputIt histogram_first);                                 // (2)

``oneapi::dpl::histogram`` computes the histogram over the data in ``[start, end)``
by counting the number of elements that map to each of a set of bins and storing the counts into
the output sequence starting from ``histogram_first``. Input values that do not map to a defined
bin are skipped silently. The value type of ``OutputIt`` must be an integral type of sufficient
size to store the counts of the histogram without overflow. The return value is an iterator targeting
past the last element of the output sequence starting from ``histogram_first``.

1. The elements of ``[start, end)`` are mapped into ``num_intervals`` bins that evenly divide the range
   ``[first_interval_begin, last_interval_end)``. Each bin is of size
   ``bin_size = (last_interval_end - first_interval_begin) / num_intervals`` as represented by a real
   number without rounding or truncation. An input element ``start[i]`` maps to a bin
   ``histogram_first[j]`` if and only if
   ``(first_interval_begin + j * bin_size <= start[i]) && (start[i] < first_interval_begin + (j + 1) * bin_size)``.
   Both `ValueType` and the value type of ``InputIt`` must be arithmetic types.

2. The elements of ``[start, end)`` are mapped into ``std::distance(boundary_start, boundary_end) - 1)``
   bins defined by the values in ``[boundary_start, boundary_end)``. An input 
   element ``start[i]`` maps to a bin ``histogram_first[j]`` if and only if
   ``(boundary_start[j] <= start[i]) && (start[i] < boundary_start[j + 1])``.  The value types
   of ``InputIt1`` and ``InputIt2`` must be arithmetic types. The values in
   ``[boundary_start, boundary_end)`` must be sorted with respect to ``operator<``.

.. _`C++ Standard`: https://isocpp.org/std/the-standard
