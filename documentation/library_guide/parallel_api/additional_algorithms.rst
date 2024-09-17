Additional Algorithms
######################

The definitions of the algorithms listed below are available through the ``<oneapi/dpl/algorithm>``
header.  All algorithms are implemented in the ``oneapi::dpl`` namespace.

* ``reduce_by_segment``: performs partial reductions on a sequence's values and keys. Each
  reduction is computed with a given reduction operation for a contiguous subsequence of values, which are
  determined by keys being equal according to a predicate. A return value is a pair of iterators holding
  the end of the output sequences for keys and values.

  For correct computation, the reduction operation should be associative. If no operation is specified,
  the default operation for the reduction is ``std::plus``, and the default predicate is ``std::equal_to``.
  The algorithm requires that the type of the elements used for values be default constructible. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    output_keys:   [0,1]
    output_values: [1+2+3=6,4+5+6=15]

* ``inclusive_scan_by_segment``: performs partial prefix scans on a sequence's values. Each
  scan applies to a contiguous subsequence of values, which are determined by the keys associated with the
  values being equal. The return value is an iterator targeting the end of the result sequence.

  For correct computation, the prefix scan operation should be associative. If no operation is specified,
  the default operation is ``std::plus``, and the default predicate is ``std::equal_to``. The algorithm
  requires that the type of the elements used for values be default constructible. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    result: [1,1+2=3,1+2+3=6,4,4+5=9,4+5+6=15]

* ``exclusive_scan_by_segment``: performs partial prefix scans on a sequence's values. Each
  scan applies to a contiguous subsequence of values that are determined by the keys associated with the values
  being equal, and sets the first element to the initial value provided. The return value is an iterator
  targeting the end of the result sequence.

  For correct computation, the prefix scan operation should be associative. If no operation is specified,
  the default operation is ``std::plus``, and the default predicate is ``std::equal_to``. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    initial value: [0]
    result: [0,0+1=1,0+1+2=3,0,0+4=4,0+4+5=9]

* ``binary_search``: performs a binary search of the input sequence for each of the values in
  the search sequence provided.  For each element of the search sequence the algorithm writes a boolean value
  to the result sequence that indicates whether the search value was found in the input sequence. An iterator
  to one past the last value in the result sequence is returned. The algorithm assumes the input sequence has
  been sorted by the comparator provided. If no comparator is provided, then a function object that uses
  ``operator<`` to compare the elements is used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [true, true, false, false, true]

* ``lower_bound``: performs a binary search of the input sequence for each of the values in
  the search sequence provided to identify the lowest index in the input sequence where the search value could
  be inserted without violating the sorted ordering of the input sequence.  The lowest index for each search
  value is written to the result sequence, and the algorithm returns an iterator to one past the last value
  written to the result sequence. If no comparator is provided, then a function object that uses ``operator<``
  to compare the elements is used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [0, 1, 8, 10, 8]

* ``upper_bound``: performs a binary search of the input sequence for each of the values in
  the search sequence provided to identify the highest index in the input sequence where the search value could
  be inserted without violating the sorted ordering of the input sequence.  The highest index for each search
  value is written to the result sequence, and the algorithm returns an iterator to one past the last value
  written to the result sequence. If no comparator is provided, then a function object that uses ``operator<``
  to compare the elements is used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [1, 4, 8, 10, 10]

* ``sort_by_key``: performs a key-value sort.
  The algorithm sorts a sequence of keys using a given comparison function object.
  If it is not provided, the elements are compared with ``operator<``.
  A sequence of values is simultaneously permuted according to the sorted order of keys.
  There must be at least as many values as the keys, otherwise the behavior is undefined.

  For example::

    keys:   [3,    5,   0,   4,   3,   0]
    values: ['a', 'b', 'c', 'd', 'e', 'f']
    output_keys:   [0,    0,   3,   3,   4,   5]
    output_values: ['c', 'f', 'a', 'e', 'd', 'b']

.. note::
     ``sort_by_key`` currently implements a stable sort for device execution policies,
     but may implement an unstable sort in the future.
     Use ``stable_sort_by_key`` if stability is essential.

* ``stable_sort_by_key``: performs a key-value sort similar to ``sort_by_key``,
  but with the added guarantee of stability.

* ``transform_if``: performs a transform on the input sequence(s) elements and stores the result into the
  corresponding position in the output sequence at each position for which the predicate applied to the
  element(s) evaluates to ``true``. If the predicate evaluates to ``false``, the transform is not applied for
  the elements(s), and the output sequence's corresponding position is left unmodified. There are two overloads
  of this function, one for a single input sequence with a unary transform and a unary predicate, and another
  for two input sequences and a binary transform and a binary predicate.

  Unary example::

    unary predicate: [](auto i){return i % 2 == 0;} // is even
    unary transform: [](auto i){return i * 2;}      // double element
    input sequence:           [0, 1, 2, 3, 3, 3, 4, 4, 7, 6]
    original output sequence: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    final output sequence:    [0, 8, 4, 6, 5, 4, 8, 8, 1, 12]


  Binary example::

    binary predicate: [](auto a, auto b){return a == b;} // are equal
    unary transform:  [](auto a, auto b){return a + b;}  // sum values
    input sequence1:           [0, 1, 2, 3, 3, 3, 4, 4, 7, 6]
    input sequence2:           [5, 1, 3, 4, 3, 3, 4, 4, 7, 9]
    original output sequence:  [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    final output sequence:     [9, 2, 9, 9, 6, 6, 8, 8, 14, 9]

* ``histogram``: performs a histogram on a sequence of of input elements. Histogram counts the number of
  elements which map to each of a defined set of bins. The algorithm has two overloads.

  The first overload takes as input the number of bins, range minimum, and range maximum, then evenly
  divides bins within that range. An input element ``a`` maps to a bin ``i`` such that
  ``i = floor((a - minimum) / ((maximum - minimum) / num_bins)))``.

  The other overload defines ``m`` bins from a sorted sequence of ``m + 1`` user-provided boundaries
  where an input element ``a`` maps to a bin ``i`` if and only if
  ``__boundary_first[i] <= a < __boundary_first[i + 1]``.

  Input values which do not map to a defined bin are skipped silently. The algorithm counts the number of
  input elements which map to each bin and outputs the result to a user-provided sequence of ``m`` output
  bin counts. The user must provide sufficient output data to store each bin, and the type of the output
  sequence must be sufficient to store the counts of the histogram without overflow. All input and output
  sequences must be ``RandomAccessIterators``. Histogram currently only supports execution with device
  policies.

  Evenly divided bins example::

    inputs:   [9, 9, 3, 8, 4, 4, 4, 5, 1, 99]
    num_bins: 5
    min:      0
    max:      10
    output:   [1, 1, 4, 0 3]

  Custom range bins example::

    inputs:     [9, 9, 3, 8, 4, 4, 4, 5, 1, 99]
    boundaries: [-1, 0, 8, 12]
    output:     [0, 6, 3]


