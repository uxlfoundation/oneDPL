# zip_view Support for the oneDPL Range APIs with C++20

## Introduction
`std::ranges::zip_view` is a powerful utility that enables developers to combine two or more ranges into a single view,
where each element is represented as a tuple containing corresponding elements from each input range.

## Motivations
`std::ranges::zip_view` is a convenient way to combine multiple ranges into a single view, where each element of
the resulting range is a tuple containing one element from each of the input ranges. This can be particularly
useful for iterating over multiple collections in parallel. `std::ranges::zip_view` was introduced in C++23,
but many developers are still using C++20 standard. So, oneDPL introduces `oneapi::dpl::ranges::zip_view`,
with the same API and functionality as `std::ranges::zip_view`.

In case of C++23 `oneapi::dpl::ranges::zip_view` using also makes sense at least for the device policies, because
`std::ranges::zip_view` C++23 still is not device copyable. Any wrapper over `std::tuple` C++23 is not device copyable. (https://godbolt.org/z/brfvcMeM6)


### Key Requirements
`oneapi::dpl::ranges::zip_view` should be:
- compilable with C++20 version (minimum)
- API-compliant with `std::ranges::zip_view`
- in case of a device usage: a device copyable view if the all "underlying" views are device copyable views.
- To provide a transitive device copyability oneDPL tuple-like type underhood is proposed.
  So, oneDPL tuple-like concept  should be proposed into the oneDPL spec.
  
`oneapi::dpl::ranges::zip_view::iterator` should be:
- value-swappable (https://en.cppreference.com/w/cpp/named_req/ValueSwappable)
- able to be used with the non-range algorithms

### Discrepancies with std::zip_view C++23
- `oneapi::dpl::ranges::zip_view` is based on oneDPL tuple-like type oneapi::dpl::__internal::tuple instead of std::tuple.
- `oneapi::dpl::ranges::zip_view::iterator::value_type` is oneDPL tuple-like type oneapi::dpl::__internal::tuple instead of std::tuple.

### Other technical reasons not to use std::zip_view C++23 with oneDPL algorithms in the future:
- There is an issue with `std::ranges::sort(zip_view)` with clang 19.0 and older. (https://godbolt.org/z/jKvG9rY5M)
- There is an issue with `std::ranges::stable_sort(zip_view)` with gcc library 
- Passing `std::zip_view::iterator` intances to the iterator-based algorithms works only for gcc 14.1 and newer, clang 19.1 and newer or
  starting 17.01 with libc++ lib (https://godbolt.org/z/To6Mjr9M6)

### Implementation proposal (C++20)
- `oneapi::dpl::ranges::zip_view` is designed as a C++ class which represents a range adaptor (see C++ Range Library).
This class encapsulates a tuple-like type to keep a combination of two or more ranges.
- To ensure device copyability, `oneapi::dpl::__internal::tuple` is proposed as a tuple-like type for underlying elements.
- To provide a value-swappable requirement `oneapi::dpl::__internal::tuple` is proposed as a dereferenced value for
`oneapi::dpl::ranges::zip_view::iterator` due to `std::tuple` not satisfying the value-swappable requirement in C++20.
- Usage of C++ concepts is desirable to write type requirements for types, methods and members of the class.
- C++20 is minimum supported version for the class. It allows using modern C++ features such as concepts and others.
- Considiration `std::tuple` as `oneapi::dpl::ranges::zip_view::iterator`. There are issues, at least, with `sortable`, `permutable`
  and `indirectly_writable` concepts: const_cast<const std::iter_reference_t<Out>&&>(*o) = std::forward<T>(t) is not compiled till C++23.  (https://godbolt.org/z/zT9qqnjWq)

### Test coverage

- `oneapi::dpl::ranges::zip_view` is tested itself, base functionality (the API that is used for a range in the oneDPL algorithm implementations)
- the base functionality test coverage may be extended by the adapted LLVM `std::ranges::zip_view` tests.
- should be tested with at least one oneDPL range based algorithm.
- should be tested with at least one oneDPL iterator based algorithm.
