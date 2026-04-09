.. SPDX-FileCopyrightText: Contributors to the oneAPI Specification project.
..
.. SPDX-License-Identifier: CC-BY-4.0

Parallel Range Algorithms
-------------------------

oneDPL provides variations of algorithms that work with ranges defined in the `C++ Standard`_, 6th edition (C++20)
and newer. These algorithms execute according to a oneDPL execution policy supplied as the first argument,
similarly to other oneDPL algorithms.
[*Note*: These algorithms mostly match the semantics of the
`parallel range algorithms <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3179r9.html>`_
in the working draft of the next C++ standard edition (C++26). -- *end note*]

The oneDPL parallel range algorithms rely on the functionality of C++20 and are not available in the code
compiled for earlier editions of the C++ standard.

The parallel range algorithms reside in ``namespace oneapi::dpl::ranges``. Same as the range algorithm functions
defined by the C++ standard in ``namespace std::ranges``, they cannot be found by argument-dependent name lookup
and cannot be called with explicitly specified template arguments. [*Note*: A typical implementation uses
predefined function objects which static function call operators have the required signatures. -- *end note*]

The following differences to the standard serial C++ range algorithms apply:

- List initialization of value parameters is enabled, as in the C++26 working draft.
- Parallel range algorithms cannot be used in constant expressions.
- The oneDPL execution policy parameter is added.
- Output data sequences are defined as ranges, not iterators.
- Both input and output ranges must support random access.
- As a rule, both input and output ranges must be sized.

  - Exceptions are binary ``transform``, ``equal``, and ``mismatch``, where at least one of the input ranges
    must be sized, and if a range is not sized it is supposed to be infinite.
    [*Note*: An example of an infinite range is ``std::views::repeat`` with no bound. -- *end note*]

- For algorithms with bounded output ranges, processing may not need to go over all the input data.
  In that case, the returned value contains iterators pointing to the positions past the last elements
  processed according to the algorithm semantics.
- ``for_each`` does not return its function object.
- The return type of ``reverse_copy`` and ``rotate_copy`` is ``std::ranges::in_in_out_result``
  rather than ``std::ranges::reverse_copy_result`` and ``std::ranges::rotate_copy_result``, respectively.
  The semantics of the returned value are as specified in
  `P3709R2 <https://isocpp.org/files/papers/P3709R2.html>`_.
- ``destroy`` is not marked with ``noexcept``.

Auxiliary Definitions
+++++++++++++++++++++

The following auxiliary entities are only defined for the purpose of exposition, to aid the specification
of parallel range algorithms.

.. code:: cpp

   // C++20 analogue of std::projected_value_t; exposition only
   template <typename I, typename Proj>
   using /*projected-value-type*/ =
       std::remove_cvref_t<std::invoke_result_t<Proj&, std::iter_value_t<I>&>>;

  // C++20 analogue of nothrow-random-access-range in the C++26 working draft; exposition only
  // Semantic requirements are listed further below
  template <typename R>
  concept /*nothrow-random-access-range*/ =
    std::ranges::random_access_range<R> &&
    std::is_lvalue_reference_v<std::iter_reference_t<std::ranges::iterator_t<R>>> &&
    std::same_as<std::remove_cvref_t<std::iter_reference_t<std::ranges::iterator_t<R>>>,
                 std::iter_value_t<std::ranges::iterator_t<R>>>;

A type ``R`` models ``nothrow-random-access-range`` if no exceptions are thrown from:

- any operation on an object of type ``std::ranges::iterator_t<R>``
  required by the ``std::random_access_iterator`` concept;
- calls to ``std::ranges::begin()``, ``std::ranges::end()`` and ``std::ranges::size()``
  on an object of type ``R``.

Whole Sequence Operations
+++++++++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // all_of
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      bool all_of (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // any_of
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      bool any_of (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // none_of
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      bool none_of (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // count
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              typename T = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirect_binary_predicate< std::ranges::equal_to,
                                               std::projected<std::ranges::iterator_t<R>, Proj>,
                                               const T* >
      std::ranges::range_difference_t<R>
        count (ExecutionPolicy&& pol, R&& r, const T& value, Proj proj = {});

    // count_if
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::range_difference_t<R>
        count_if (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // for_each
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirectly_unary_invocable< std::projected<std::ranges::iterator_t<R>, Proj> > Fn>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_iterator_t<R>
        for_each (ExecutionPolicy&& pol, R&& r, Fn f, Proj proj = {});

    // copy
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>>
      std::ranges::copy_result<std::ranges::borrowed_iterator_t<R>,
                               std::ranges::borrowed_iterator_t<OutR>>
        copy (ExecutionPolicy&& pol, R&& r, OutR&& result);

    // move
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_movable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>>
      std::ranges::move_result<std::ranges::borrowed_iterator_t<R>,
                               std::ranges::borrowed_iterator_t<OutR>>
        move (ExecutionPolicy&& pol, R&& r, OutR&& result);

    // fill
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename T = std::ranges::range_value_t<R>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirectly_writable<std::ranges::iterator_t<R>, const T&>
      std::ranges::borrowed_iterator_t<R>
        fill (ExecutionPolicy&& pol, R&& r, const T& value);

    // swap_ranges
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::indirectly_swappable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>>
      std::ranges::swap_ranges_result<std::ranges::borrowed_iterator_t<R1>,
                                      std::ranges::borrowed_iterator_t<R2>>
        swap_ranges (ExecutionPolicy&& pol, R1&& r1, R2&& r2);

  }

Minimum and Maximum
+++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // min
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirectly_copyable_storable< std::ranges::iterator_t<R>,
                                                  std::ranges::range_value_t<R>* >
      std::ranges::range_value_t<R>
        min (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // max
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirectly_copyable_storable< std::ranges::iterator_t<R>,
                                                  std::ranges::range_value_t<R>* >
      std::ranges::range_value_t<R>
        max (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // minmax
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirectly_copyable_storable< std::ranges::iterator_t<R>,
                                                  std::ranges::range_value_t<R>* >
      std::ranges::minmax_result<std::ranges::range_value_t<R>>
        minmax (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // min_element
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_iterator_t<R>
        min_element (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // max_element
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_iterator_t<R>
        max_element (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // minmax_element
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::minmax_element_result<std::ranges::borrowed_iterator_t<R>>
        minmax_element (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

  }

Element Search Operations
+++++++++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // find
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              typename T = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirect_binary_predicate< std::ranges::equal_to,
                                               std::projected<std::ranges::iterator_t<R>, Proj>,
                                               const T* >
      std::ranges::borrowed_iterator_t<R>
        find (ExecutionPolicy&& pol, R&& r, const T& value, Proj proj = {});

    // find_if
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_iterator_t<R>
        find_if (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // find_if_not
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_iterator_t<R>
        find_if_not (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // find_first_of
    template<typename ExecutionPolicy, std::ranges::random_access_range R1,
             std::ranges::random_access_range R2, typename Pred = std::ranges::equal_to,
             typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::indirectly_comparable< std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                                           Pred, Proj1, Proj2 >
      std::ranges::borrowed_iterator_t<R1>
        find_first_of (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Pred pred = {},
                       Proj1 proj1 = {}, Proj2 proj2 = {});

    // adjacent_find
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_binary_predicate< std::projected<std::ranges::iterator_t<R>, Proj>,
                                              std::projected<std::ranges::iterator_t<R>, Proj> >
                    Pred = std::ranges::equal_to>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_iterator_t<R>
        adjacent_find (ExecutionPolicy&& pol, R&& r, Pred pred = {}, Proj proj = {});

    // contains
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              typename T = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirect_binary_predicate< std::ranges::equal_to,
                                               std::projected<std::ranges::iterator_t<R>, Proj>,
                                               const T* >
      bool contains (ExecutionPolicy&& pol, R&& r, const T& value, Proj proj = {});

    // find_last
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              typename T = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirect_binary_predicate< std::ranges::equal_to,
                                               std::projected<std::ranges::iterator_t<R>, Proj>,
                                               const T* >
      std::ranges::borrowed_subrange_t<R>
        find_last (ExecutionPolicy&& pol, R&& r, const T& value, Proj proj = {});

    // find_last_if
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_subrange_t<R>
        find_last_if (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // find_last_if_not
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_subrange_t<R>
        find_last_if_not (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

  }

Sequence Search and Comparison
++++++++++++++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // equal
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
             std::ranges::random_access_range R2, typename Pred = std::ranges::equal_to,
             typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               (std::ranges::sized_range<R1> || std::ranges::sized_range<R2>) &&
               std::indirectly_comparable< std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                                           Pred, Proj1, Proj2 >
      bool equal (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Pred pred = {},
                  Proj1 proj1 = {}, Proj2 proj2 = {});

    // mismatch
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
             std::ranges::random_access_range R2, typename Pred = std::ranges::equal_to,
             typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               (std::ranges::sized_range<R1> || std::ranges::sized_range<R2>) &&
               std::indirectly_comparable< std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                                           Pred, Proj1, Proj2 >
      std::ranges::mismatch_result<std::ranges::borrowed_iterator_t<R1>,
                                   std::ranges::borrowed_iterator_t<R2>>
        mismatch (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Pred pred = {},
                  Proj1 proj1 = {}, Proj2 proj2 = {});

    // find_end
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
             std::ranges::random_access_range R2, typename Pred = std::ranges::equal_to,
             typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::indirectly_comparable< std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                                           Pred, Proj1, Proj2 >
      std::ranges::borrowed_subrange_t<R1>
        find_end (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Pred pred = {},
                  Proj1 proj1 = {}, Proj2 proj2 = {});

    // search
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
             std::ranges::random_access_range R2, typename Pred = std::ranges::equal_to,
             typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::indirectly_comparable< std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                                           Pred, Proj1, Proj2 >
      std::ranges::borrowed_subrange_t<R1>
        search (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Pred pred = {},
                Proj1 proj1 = {}, Proj2 proj2 = {});

    // search_n
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
             typename Pred = std::ranges::equal_to, typename Proj = std::identity,
             typename T = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirectly_comparable< std::ranges::iterator_t<R>, const T*, Pred, Proj >
      std::ranges::borrowed_subrange_t<R>
        search_n (ExecutionPolicy&& pol, R&& r, std::ranges::range_difference_t<R> count,
                  const T& value, Pred pred = {}, Proj proj = {});

    // lexicographical_compare
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, typename Proj1 = std::identity,
              typename Proj2 = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R1>, Proj1>,
                                               std::projected<std::ranges::iterator_t<R2>, Proj2> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2>
      bool lexicographical_compare (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Comp comp = {},
                                   Proj1 proj1 = {}, Proj2 proj2 = {});

    // contains_subrange
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, typename Pred = std::ranges::equal_to,
              typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::indirectly_comparable< std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                                           Pred, Proj1, Proj2 >
      bool contains_subrange (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Pred pred = {},
                              Proj1 proj1 = {}, Proj2 proj2 = {});

    // starts_with
    template < typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, typename Pred = std::ranges::equal_to,
              typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::indirectly_comparable< std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                                           Pred, Proj1, Proj2 >
      bool starts_with (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Pred pred = {},
                        Proj1 proj1 = {}, Proj2 proj2 = {});

    // ends_with
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, typename Pred = std::ranges::equal_to,
              typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::indirectly_comparable< std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                                           Pred, Proj1, Proj2 >
      bool ends_with (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Pred pred = {},
                      Proj1 proj1 = {}, Proj2 proj2 = {});

  }

Sequence Transformation
+++++++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // transform (unary)
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR, std::copy_constructible Fn,
              typename Proj = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_writable< std::ranges::iterator_t<OutR>,
                    std::indirect_result_t<Fn&, std::projected<std::ranges::iterator_t<R>, Proj>> >
      std::ranges::unary_transform_result<std::ranges::borrowed_iterator_t<R>,
                                          std::ranges::borrowed_iterator_t<OutR>>
        transform (ExecutionPolicy&& pol, R&& r, OutR&& result, Fn unary_op, Proj proj = {});

    // transform (binary)
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, std::ranges::random_access_range OutR,
              std::copy_constructible Fn, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               (std::ranges::sized_range<R1> || std::ranges::sized_range<R2>) &&
               std::ranges::sized_range<OutR> &&
               std::indirectly_writable< std::ranges::iterator_t<OutR>,
                    std::indirect_result_t<Fn&, std::projected<std::ranges::iterator_t<R1>, Proj1>,
                                                std::projected<std::ranges::iterator_t<R2>, Proj2>> >
      std::ranges::binary_transform_result<std::ranges::borrowed_iterator_t<R1>,
                                           std::ranges::borrowed_iterator_t<R2>,
                                           std::ranges::borrowed_iterator_t<OutR>>
        transform (ExecutionPolicy&& pol, R1&& r1, R2&& r2, OutR&& result, Fn binary_op,
                   Proj1 proj1 = {}, Proj2 proj2 = {});

    // replace
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              typename T1 = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>,
              typename T2 = std::ranges::range_value_t<R>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirectly_writable<std::ranges::iterator_t<R>, const T2&> &&
               std::indirect_binary_predicate< std::ranges::equal_to,
                                               std::projected<std::ranges::iterator_t<R>, Proj>,
                                               const T1* >
      std::ranges::borrowed_iterator_t<R>
        replace (ExecutionPolicy&& pol, R&& r, const T1& old_value, const T2& new_value,
                 Proj proj = {});

    // replace_if
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              typename T = std::ranges::range_value_t<R>,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::indirectly_writable<std::ranges::iterator_t<R>, const T&>
      std::ranges::borrowed_iterator_t<R>
        replace_if (ExecutionPolicy&& pol, R&& r, Pred pred, const T& new_value, Proj proj = {});

    // replace_copy
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR, typename Proj = std::identity,
              typename T1 = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>>,
              typename T2 = std::ranges::range_value_t<OutR>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>> &&
               std::indirect_binary_predicate< std::ranges::equal_to,
                                               std::projected<std::ranges::iterator_t<R>, Proj>,
                                               const T1* > &&
               std::indirectly_writable<std::ranges::iterator_t<OutR>, const T2&>
      std::ranges::replace_copy_result<std::ranges::borrowed_iterator_t<R>,
                                       std::ranges::borrowed_iterator_t<OutR>>
        replace_copy (ExecutionPolicy&& pol, R&& r, OutR&& result, const T1& old_value,
                      const T2& new_value, Proj proj = {});

    // replace_copy_if
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR,
              class T = std::ranges::range_value_t<OutR>, typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred,
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>> &&
               std::indirectly_writable<std::ranges::iterator_t<OutR>, const T&>
      std::ranges::replace_copy_if_result<std::ranges::borrowed_iterator_t<R>,
                                          std::ranges::borrowed_iterator_t<OutR>>
        replace_copy_if (ExecutionPolicy&& pol, R&& r, OutR&& result, Pred pred, const T& new_value,
                         Proj proj = {});

  }

Sequence Reordering
+++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // shift_left
    template <typename ExecutionPolicy, std::ranges::random_access_range R>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
      std::ranges::borrowed_subrange_t<R>
        shift_left (ExecutionPolicy&& pol, R&& r, std::ranges::range_difference_t<R> n);

    // shift_right
    template <typename ExecutionPolicy, std::ranges::random_access_range R>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
      std::ranges::borrowed_subrange_t<R>
        shift_right (ExecutionPolicy&& pol, R&& r, std::ranges::range_difference_t<R> n);

    // rotate
    template <typename ExecutionPolicy, std::ranges::random_access_range R>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
      std::ranges::borrowed_subrange_t<R>
        rotate (ExecutionPolicy&& pol, R&& r, std::ranges::iterator_t<R> middle);

    // rotate_copy
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>>
      std::ranges::in_in_out_result<std::ranges::borrowed_iterator_t<R>,
                                    std::ranges::borrowed_iterator_t<R>,
                                    std::ranges::borrowed_iterator_t<OutR>>
        rotate_copy (ExecutionPolicy&& pol, R&& r, std::ranges::iterator_t<R> middle, OutR&& result);

    // reverse
    template <typename ExecutionPolicy, std::ranges::random_access_range R>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
      std::ranges::borrowed_iterator_t<R>
        reverse (ExecutionPolicy&& pol, R&& r);

    // reverse_copy
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>>
      std::ranges::in_in_out_result<std::ranges::borrowed_iterator_t<R>,
                                    std::ranges::borrowed_iterator_t<R>,
                                    std::ranges::borrowed_iterator_t<OutR>>
        reverse_copy (ExecutionPolicy&& pol, R&& r, OutR&& result);

  }

Sequence Filtering
++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // copy_if
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR, typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>>
      std::ranges::copy_if_result<std::ranges::borrowed_iterator_t<R>,
                                  std::ranges::borrowed_iterator_t<OutR>>
        copy_if (ExecutionPolicy&& pol, R&& r, OutR&& result, Pred pred, Proj proj = {});

    // remove
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              typename T = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R> &&
               std::indirect_binary_predicate< std::ranges::equal_to,
                                               std::projected<std::ranges::iterator_t<R>, Proj>,
                                               const T* >
      std::ranges::borrowed_subrange_t<R>
        remove (ExecutionPolicy&& pol, R&& r, const T& value, Proj proj = {});

    // remove_if
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
      std::ranges::borrowed_subrange_t<R>
        remove_if (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // remove_copy
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR, typename Proj = std::identity,
              typename T = /*projected-value-type*/<std::ranges::iterator_t<R>, Proj>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>> &&
               std::indirect_binary_predicate< std::ranges::equal_to,
                                               std::projected<std::ranges::iterator_t<R>, Proj>,
                                               const T* >
      std::ranges::remove_copy_result<std::ranges::borrowed_iterator_t<R>,
                                      std::ranges::borrowed_iterator_t<OutR>>
        remove_copy (ExecutionPolicy&& pol, R&& r, OutR&& result, const T& value, Proj proj = {});

    // remove_copy_if
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR, typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>>
      std::ranges::remove_copy_if_result<std::ranges::borrowed_iterator_t<R>,
                                         std::ranges::borrowed_iterator_t<OutR>>
        remove_copy_if (ExecutionPolicy&& pol, R&& r, OutR&& result, Pred pred, Proj proj = {});

    // unique
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_equivalence_relation< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::equal_to>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
      std::ranges::borrowed_subrange_t<R>
        unique (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // unique_copy
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR, typename Proj = std::identity,
              std::indirect_equivalence_relation<std::projected<std::ranges::iterator_t<R>, Proj>>
                    Comp = std::ranges::equal_to>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>>
      std::ranges::unique_copy_result<std::ranges::borrowed_iterator_t<R>,
                                      std::ranges::borrowed_iterator_t<OutR>>
        unique_copy (ExecutionPolicy&& pol, R&& r, OutR&& result, Comp comp = {}, Proj proj = {});

  }

Sorting, Merge, and Heap Operations
+++++++++++++++++++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // sort
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Comp = std::ranges::less, typename Proj = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
      std::ranges::borrowed_iterator_t<R>
        sort (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // stable_sort
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Comp = std::ranges::less, typename Proj = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
      std::ranges::borrowed_iterator_t<R>
        stable_sort (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // partial_sort
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Comp = std::ranges::less, typename Proj = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
      std::ranges::borrowed_iterator_t<R>
        partial_sort (ExecutionPolicy&& pol, R&& r, std::ranges::iterator_t<R> middle,
                      Comp comp = {}, Proj proj = {});

    // partial_sort_copy
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR, typename Comp = std::ranges::less,
              typename Proj1 = std::identity, typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR>> &&
               std::sortable<std::ranges::iterator_t<OutR>, Comp, Proj2> &&
               std::indirect_strict_weak_order<Comp,
                                               std::projected<std::ranges::iterator_t<R>, Proj1>,
                                               std::projected<std::ranges::iterator_t<OutR>, Proj2> >
      std::ranges::partial_sort_copy_result<std::ranges::borrowed_iterator_t<R>,
                                            std::ranges::borrowed_iterator_t<OutR>>
        partial_sort_copy (ExecutionPolicy&& pol, R&& r, OutR&& result, Comp comp = {},
                           Proj1 proj1 = {}, Proj2 proj2 = {});

    // is_sorted
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      bool is_sorted (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // is_sorted_until
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_iterator_t<R>
        is_sorted_until (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // merge
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, std::ranges::random_access_range OutR,
              typename Comp = std::ranges::less, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::ranges::sized_range<OutR> &&
               std::mergeable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                              std::ranges::iterator_t<OutR>, Comp, Proj1, Proj2>
      std::ranges::merge_result<std::ranges::borrowed_iterator_t<R1>,
                                std::ranges::borrowed_iterator_t<R2>,
                                std::ranges::borrowed_iterator_t<OutR>>
        merge (ExecutionPolicy&& pol, R1&& r1, R2&& r2, OutR&& result, Comp comp = {},
               Proj1 proj1 = {}, Proj2 proj2 = {});

    // inplace_merge
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Comp = std::ranges::less, typename Proj = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
      std::ranges::borrowed_iterator_t<R>
        inplace_merge (ExecutionPolicy&& pol, R&& r, std::ranges::iterator_t<R> middle,
                       Comp comp = {}, Proj proj = {});

    // is_heap
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      bool is_heap (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

    // is_heap_until
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R>, Proj> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      std::ranges::borrowed_iterator_t<R>
        is_heap_until (ExecutionPolicy&& pol, R&& r, Comp comp = {}, Proj proj = {});

  }

Set operations
++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // includes
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2,
              typename Proj1 = std::identity, typename Proj2 = std::identity,
              std::indirect_strict_weak_order< std::projected<std::ranges::iterator_t<R1>, Proj1>,
                                               std::projected<std::ranges::iterator_t<R2>, Proj2> >
                    Comp = std::ranges::less>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
              std::ranges::sized_range<R1> && std::ranges::sized_range<R2>
      bool includes (ExecutionPolicy&& pol, R1&& r1, R2&& r2, Comp comp = {},
                     Proj1 proj1 = {}, Proj2 proj2 = {});

    // set_union
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, std::ranges::random_access_range OutR,
              typename Comp = std::ranges::less, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::ranges::sized_range<OutR> &&
               std::mergeable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                              std::ranges::iterator_t<OutR>, Comp, Proj1, Proj2>
      std::ranges::set_union_result<std::ranges::borrowed_iterator_t<R1>,
                                    std::ranges::borrowed_iterator_t<R2>,
                                    std::ranges::borrowed_iterator_t<OutR>>
        set_union (ExecutionPolicy&& pol, R1&& r1, R2&& r2, OutR&& result, Comp comp = {},
                   Proj1 proj1 = {}, Proj2 proj2 = {});

    // set_intersection
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, std::ranges::random_access_range OutR,
              typename Comp = std::ranges::less, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::ranges::sized_range<OutR> &&
               std::mergeable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                              std::ranges::iterator_t<OutR>, Comp, Proj1, Proj2>
      std::ranges::set_intersection_result<std::ranges::borrowed_iterator_t<R1>,
                                           std::ranges::borrowed_iterator_t<R2>,
                                           std::ranges::borrowed_iterator_t<OutR>>
        set_intersection (ExecutionPolicy&& pol, R1&& r1, R2&& r2, OutR&& result, Comp comp = {},
                          Proj1 proj1 = {}, Proj2 proj2 = {});

    // set_difference
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, std::ranges::random_access_range OutR,
              typename Comp = std::ranges::less, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::ranges::sized_range<OutR> &&
               std::mergeable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                              std::ranges::iterator_t<OutR>, Comp, Proj1, Proj2>
      std::ranges::set_difference_result<std::ranges::borrowed_iterator_t<R1>,
                                         std::ranges::borrowed_iterator_t<OutR>>
        set_difference (ExecutionPolicy&& pol, R1&& r1, R2&& r2, OutR&& result, Comp comp = {},
                        Proj1 proj1 = {}, Proj2 proj2 = {});

    // set_symmetric_difference
    template <typename ExecutionPolicy, std::ranges::random_access_range R1,
              std::ranges::random_access_range R2, std::ranges::random_access_range OutR,
              typename Comp = std::ranges::less, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
               std::ranges::sized_range<OutR> &&
               std::mergeable<std::ranges::iterator_t<R1>, std::ranges::iterator_t<R2>,
                              std::ranges::iterator_t<OutR>, Comp, Proj1, Proj2>
      std::ranges::set_symmetric_difference_result<std::ranges::borrowed_iterator_t<R1>,
                                                   std::ranges::borrowed_iterator_t<R2>,
                                                   std::ranges::borrowed_iterator_t<OutR>>
        set_symmetric_difference (ExecutionPolicy&& pol, R1&& r1, R2&& r2, OutR&& result,
                                 Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});

  }

Partition operations
++++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/algorithm>

  namespace oneapi::dpl::ranges {

    // is_partitioned
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R>
      bool is_partitioned (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // partition
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
      std::ranges::borrowed_subrange_t<R>
        partition (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // stable_partition
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
      std::ranges::borrowed_subrange_t<R>
        stable_partition (ExecutionPolicy&& pol, R&& r, Pred pred, Proj proj = {});

    // partition_copy
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              std::ranges::random_access_range OutR1, std::ranges::random_access_range OutR2,
              typename Proj = std::identity,
              std::indirect_unary_predicate< std::projected<std::ranges::iterator_t<R>, Proj> > Pred>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::ranges::sized_range<OutR1> &&
               std::ranges::sized_range<OutR2> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR1>> &&
               std::indirectly_copyable<std::ranges::iterator_t<R>, std::ranges::iterator_t<OutR2>>
      std::ranges::partition_copy_result<std::ranges::borrowed_iterator_t<R>,
                                         std::ranges::borrowed_iterator_t<OutR1>,
                                         std::ranges::borrowed_iterator_t<OutR2>>
        partition_copy (ExecutionPolicy&& pol, R&& r, OutR1&& out_true_r, OutR2&& out_false_r,
                        Pred pred, Proj proj = {});

    // nth_element
    template <typename ExecutionPolicy, std::ranges::random_access_range R,
              typename Comp = std::ranges::less, typename Proj = std::identity>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> && std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
      std::ranges::borrowed_iterator_t<R>
        nth_element (ExecutionPolicy&& pol, R&& r, std::ranges::iterator_t<R> nth,
                     Comp comp = {}, Proj proj = {});

  }

Uninitialized Memory Algorithms
+++++++++++++++++++++++++++++++

.. code:: cpp

  // Defined in <oneapi/dpl/memory>

  namespace oneapi::dpl::ranges {

    // uninitialized_default_construct
    template <typename ExecutionPolicy, /*nothrow-random-access-range*/ R>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::default_initializable<std::ranges::range_value_t<R>>
      std::ranges::borrowed_iterator_t<R>
        uninitialized_default_construct (ExecutionPolicy&& pol, R&& r);

    // uninitialized_value_construct
    template <typename ExecutionPolicy, /*nothrow-random-access-range*/ R>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::default_initializable<std::ranges::range_value_t<R>>
      std::ranges::borrowed_iterator_t<R>
        uninitialized_value_construct (ExecutionPolicy&& pol, R&& r);

    // uninitialized_copy
    template <typename ExecutionPolicy, std::random_access_range IR,
              /*nothrow-random-access-range*/ OR>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<IR> && std::ranges::sized_range<OR> &&
               std::constructible_from<std::ranges::range_value_t<OR>,
                                       std::ranges::range_reference_t<IR>>
      std::ranges::uninitialized_copy_result<std::ranges::borrowed_iterator_t<IR>,
                                             std::ranges::borrowed_iterator_t<OR>>
        uninitialized_copy (ExecutionPolicy&& pol, IR&& in_range, OR&& out_range);

    // uninitialized_move
    template <typename ExecutionPolicy, std::ranges::random_access_range IR,
              /*nothrow-random-access-range*/ OR>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<IR> && std::ranges::sized_range<OR> &&
               std::constructible_from<std::ranges::range_value_t<OR>,
                                       std::ranges::range_rvalue_reference_t<IR>>
      std::ranges::uninitialized_move_result<std::ranges::borrowed_iterator_t<IR>,
                                             std::ranges::borrowed_iterator_t<OR>>
        uninitialized_move (ExecutionPolicy&& pol, IR&& in_range, OR&& out_range);

    // uninitialized_fill
    template <typename ExecutionPolicy, /*nothrow-random-access-range*/ R,
              typename T = std::ranges::range_value_t<R>>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::constructible_from<std::ranges::range_value_t<R>, const T&>
      std::ranges::borrowed_iterator_t<R>
        uninitialized_fill (ExecutionPolicy&& pol, R&& r, const T& value);

    // destroy
    template <typename ExecutionPolicy, /*nothrow-random-access-range*/ R>
      requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
               std::ranges::sized_range<R> &&
               std::destructible<std::ranges::range_value_t<R>>
      std::ranges::borrowed_iterator_t<R>
        destroy (ExecutionPolicy&& pol, R&& r);

  }

.. _`C++ Standard`: https://isocpp.org/std/the-standard
.. _`SYCL`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
