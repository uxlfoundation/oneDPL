Iterators
#########

The definitions of the iterators are available through the ``<oneapi/dpl/iterator>``
header.  All iterators are implemented in the ``oneapi::dpl`` namespace.

* ``counting_iterator``: a random-access iterator-like type whose dereferenced value is an integer
  counter. Instances of a ``counting_iterator`` provide read-only dereference operations. The counter of an
  ``counting_iterator`` instance changes according to the arithmetic of the random-access iterator type:

  .. code:: cpp

    dpl::counting_iterator<int> count_a(0);
    dpl::counting_iterator<int> count_b = count_a + 10;
    int init = count_a[0]; // OK: init == 0
    *count_b = 7; // ERROR: counting_iterator does not provide write operations
    auto sum = dpl::reduce(dpl::execution::dpcpp_default,
                           count_a, count_b, init); // sum is (0 + 0 + 1 + ... + 9) = 45

* ``zip_iterator``: an iterator constructed with one or more iterators as input. The result of
  ``zip_iterator`` dereferencing is a tuple-like object of an unspecified type that holds the values
  returned by dereferencing the member iterators, which the ``zip_iterator`` wraps. Arithmetic operations
  performed on a ``zip_iterator`` instance are also applied to each of the member iterators.

  The ``make_zip_iterator`` function is provided to simplify the construction of ``zip_iterator`` instances.
  The function returns ``zip_iterator`` instances with all the arguments held as member iterators.

  The ``zip_iterator`` is useful in defining by key algorithms where input iterators
  representing keys and values are processed as key-value pairs. The example below demonstrates a stable sort
  by key, where only the keys are compared but both keys and values are swapped:

  .. code:: cpp

    auto zipped_begin = dpl::make_zip_iterator(keys_begin, vals_begin);
    dpl::stable_sort(dpl::execution::dpcpp_default, zipped_begin, zipped_begin + n,
        [](auto lhs, auto rhs) { return get<0>(lhs) < get<0>(rhs); });

  The dereferenced object of ``zip_iterator`` supports the *structured binding* feature (`C++17 and above
  <https://en.cppreference.com/w/cpp/language/structured_binding>`_) for easier access to
  wrapped iterators values:

  .. code:: cpp

    auto zipped_begin = dpl::make_zip_iterator(sequence1.begin(), sequence2.begin(), sequence3.begin());
    auto found = dpl::find(dpl::execution::dpcpp_default, zipped_begin, zipped_begin + n,
        [](auto tuple_like_obj) {
          auto [e1, e2, e3] = tuple_like_obj;
          return e1 == e2 && e1 == e3;
        }
    );

  Since dereferencing ``zip_iterator`` is semantically a tuple of references, the copying of such an object
  is supposed to be cheap. In the example above ``e1``, ``e2`` and ``e3`` are references.

  For more examples with ``zip_iterator``, see the code snippet provided for ``discard_iterator`` below.

* ``discard_iterator``: a random-access iterator-like type that provides write-only dereference
  operations that discard values passed.

  The ``discard_iterator`` is useful in the implementation of stencil algorithms where the stencil is not part of the
  desired output. An example of this would be a ``copy_if`` algorithm that receives an input iterator range,
  a stencil iterator range, and copies the elements of the input whose corresponding stencil value is 1. Use
  ``discard_iterator`` so you do not declare a temporary allocation to store the copy of the stencil:

  .. code:: cpp

    auto zipped_first = dpl::make_zip_iterator(first, stencil);
    dpl::copy_if(dpl::execution::dpcpp_default,
                 zipped_first, zipped_first + (last - first),
                 dpl::make_zip_iterator(result, dpl::discard_iterator()),
                 [](auto t){return get<1>(t) == 1;}

* ``transform_iterator``: an iterator defined over another iterator whose dereferenced value is the result
  of a function applied to the corresponding element of the base iterator. Both the type of the base
  iterator and the unary function applied during dereference operations are required template parameters of
  the ``transform_iterator`` class. 

  The unary functor provided to a ``transform_iterator`` should have a ``const``-qualified call operator which accepts
  the reference type of the base iterator as argument. The functor's call operator should not have any side effects and
  should not modify the state of the functor object.
  
  The ``transform_iterator`` class provides the following constructors:

  * ``transform_iterator()``: instantiates the iterator using a default constructed base iterator and unary functor.
    This constructor participates in overload resolution only if the base iterator and unary functor are both default constructible.
  
  * ``transform_iterator(iter)``: instantiates the iterator using the base iterator provided and a default constructed
    unary functor. This constructor participates in overload resolution only if the unary functor is default constructible.
  
  * ``transform_iterator(iter, func)``: instantiates the iterator using the base iterator and unary functor provided.

  To simplify the construction of the iterator, ``oneapi::dpl::make_transform_iterator`` is provided. The
  function receives the base iterator and transform operation instance as arguments, and constructs the
  ``transform_iterator`` instance:

  .. code:: cpp

    dpl::counting_iterator<int> first(0);
    dpl::counting_iterator<int> last(10);
    auto transform_first = dpl::make_transform_iterator(first, std::negate<int>());
    auto transform_last = transform_first + (last - first);
    auto sum = dpl::reduce(dpl::execution::dpcpp_default,
                           transform_first, transform_last); // sum is (0 + -1 + ... + -9) = -45

* ``permutation_iterator``: an iterator whose dereferenced value set is defined by the source iterator
  provided, and whose iteration order over the dereferenced value set is defined by either another iterator or
  a functor whose index operator defines the mapping from the ``permutation_iterator`` index to the index of the
  source iterator. The ``permutation_iterator`` is useful in implementing applications where noncontiguous
  elements of data represented by an iterator need to be processed by an algorithm as though they were contiguous.
  An example is copying every other element to an output iterator. The source iterator cannot be a host-side iterator
  in cases where algorithms are executed with device policies.

  The ``make_permutation_iterator`` is provided to simplify construction of iterator instances. The function
  receives the source iterator and the iterator or function object representing the index map:

  .. code:: cpp

    struct multiply_index_by_two {
        template <typename Index>
        Index operator()(const Index& i) const
        { return i*2; }
    };

    // first and last are iterators that define a contiguous range of input elements
    // compute the number of elements in the range between the first and last that are accessed
    // by the permutation iterator
    size_t num_elements = std::distance(first, last) / 2 + std::distance(first, last) % 2;
    auto permutation_first = dpl::make_permutation_iterator(first, multiply_index_by_two());
    auto permutation_last = permutation_first + num_elements;
    dpl::copy(dpl::execution::dpcpp_default, permutation_first, permutation_last, result);
