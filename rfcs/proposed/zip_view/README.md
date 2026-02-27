# zip_view Support for the oneDPL Range APIs with C++20

We propose to add `ranges::zip_view` to oneDPL, with at least the same API and functionality as `std::ranges::zip_view`.

## Motivation
`std::ranges::zip_view` is a convenient way to combine multiple ranges into a single view, where each element of
the resulting range is a tuple referring to corresponding elements from each of the input ranges. This can be particularly
useful for iterating over multiple collections in parallel. `std::ranges::zip_view` is added in C++23,
but many developers are still using the C++20 standard, and it is also the required standard for oneDPL parallel range algorithms.

In case of C++23 having a `zip_view` in oneDPL also makes sense at least for the device policies, because
`std::ranges::zip_view` still is not device copyable (any wrapper over `std::tuple` is not device copyable, see https://godbolt.org/z/brfvcMeM6)
There are other technical issues with `std::tuple` (see below for the details).

## Key Requirements
`ranges::zip_view` should be:
- compilable with C++20 version (minimum);
- API-compliant with `std::ranges::zip_view`;
- in case of device usage: a device copyable view if all underlying views are device copyable views;
- The implementation may be based on tuple-like type underhood, but it must provide transitive device copyability
  for view pipelines created over `ranges::zip_view`.
  
`ranges::zip_view::iterator` should be:
- value-swappable (https://en.cppreference.com/w/cpp/named_req/ValueSwappable)
- indirectly writable (https://en.cppreference.com/w/cpp/iterator/indirectly_writable.html)
- able to be used with the non-range algorithms, including C++ and oneDPL parallel algorithms:
  depending on the algorithm, stricter requirements may apply. For example, `std::sortable` concept must be satisfied in order to call `std::ranges::sort`.

### Discrepancies with std::zip_view C++23
- `ranges::zip_view` may use a oneDPL tuple-like type instead of `std::tuple` 
- `ranges::zip_view::iterator::value_type` should not be defined as `std::tuple` (see a known technical issue below)

### Other technical reasons not to use std::zip_view C++23 (and std::tuple) with oneDPL algorithms in the future:
- There is an issue with `std::ranges::sort(zip_view)` with clang 19.0 and older. (https://godbolt.org/z/jKvG9rY5M)
- There is an issue with `std::ranges::stable_sort(zip_view)` with gcc library 
- Passing `std::zip_view::iterator` instances to the iterator-based algorithms works only for gcc 14.1 and newer, clang 19.1 and newer or
  starting 17.01 with libc++ lib (https://godbolt.org/z/To6Mjr9M6)
- Consideration `std::tuple` as `ranges::zip_view::iterator::value_type`. There are issues, at least, with `sortable`, `permutable`
  and `indirectly_writable` concepts: `const_cast<const std::iter_reference_t<Out>&&>(*o) = std::forward<T>(t)` is not compiled till C++23.  (https://godbolt.org/z/zT9qqnjWq)

## Implementation proposal (C++20)
- `ranges::zip_view` is as a C++ class representing a range adaptor (see C++ Range Library).
- The implementation derives from the C++ `std::ranges::view_interface`.
This class encapsulates a tuple-like type to keep a combination of two or more ranges.
- To ensure device copyability, `oneapi::dpl::__internal::tuple` is proposed as a tuple-like type for underlying elements.
- To provide a value-swappable requirement `oneapi::dpl::__internal::tuple` is proposed as a dereferenced value for
`ranges::zip_view::iterator` due to `std::tuple` not satisfying the value-swappable requirement in C++20.
- To provide an indirectly writable requirement `oneapi::dpl::__internal::tuple` is proposed as the public type for `ranges::zip_view::iterator::value_type`.
- Usage of C++ concepts is desirable to write type requirements for types, methods and members of the class.
- C++20 is minimum supported version for the class. It allows using modern C++ features such as concepts and others.
- A range adapter `zip_view` is available in the `oneapi::dpl::experimental::ranges` namespace.
- A customization point object `zip` is available in the `oneapi::dpl::experimental::ranges::views` and `oneapi::dpl::experimental::views` namespaces.

### Test coverage
- `ranges::zip_view` is tested itself, base functionality (the API that is used for a range in the oneDPL algorithm implementations)
- the base functionality test coverage may be extended by the adapted LLVM `std::ranges::zip_view` (C++23) tests.
- should be tested with range based algorithms.
- should be tested with iterator based algorithms.

### Experimental stage
Until the API is fully specified, it should go into `namespace oneapi::dpl::experimental`.
For C++20 and higher, it replaces the current implementation in that namespace, extending applicability
beyond experimental range algorithms but still supporting those.
