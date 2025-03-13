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

### Key Requirements
`oneapi::dpl::ranges::zip_view` should be:
- compilable with C++20 version (minimum)
- API-compliant with `std::ranges::zip_view`
- random accessible view; the "underlying" views also should be random accessible
- in case of a device usage: a device copyable view if the all "underlying" views are device copyable views.
- To provide a transitive device copyability oneapi::dpl::__internal::tuple is proposed as tuple-like type underhood.
  
`oneapi::dpl::ranges::zip_view::iterator` should be:
- value-swappable (https://en.cppreference.com/w/cpp/named_req/ValueSwappable)
- able to be used with the non-range algorithms

### Discrepancies with std::zip_view C++23
- `oneapi::dpl::ranges::zip_view` is based on oneDPL tuple-like type oneapi::dpl::__internal::tuple instead of std::tuple.
- `oneapi::dpl::ranges::zip_view::iterator::value_type` is oneDPL tuple-like type oneapi::dpl::__internal::tuple instead of std::tuple.

### Implementation proposal
- `oneapi::dpl::ranges::zip_view` is designed as a C++ class which represents a range adaptor (see C++ Range Library).
This class encapsulates a tuple-like type to keep a combination of two or more ranges.
- The implementation provides all necessary operators to satisfy 'random accessible view' requirement.
- To ensure device copyability, `oneapi::dpl::__internal::tuple` is proposed as a tuple-like type for underlying elements.
- To provide a value-swappable requirement `oneapi::dpl::__internal::tuple` is proposed as a dereferenced value for
`oneapi::dpl::ranges::zip_view::iterator` due to `std::tuple` not satisfying the value-swappable requirement in C++20.
- Usage of C++ concepts is desirable to write type requirements for types, methods and members of the class.
- C++20 is minimum supported version for the class. It allows using modern C++ features such as concepts and others.

### Test coverage

- `oneapi::dpl::ranges::zip_view` is tested itself, base functionality (the API that is used for a range in the oneDPL algorithm implementations)
- the base functionality test coverage may be extended by the adapted LLVM `std::ranges::zip_view` tests.
- should be tested with at least one oneDPL range based algorithm.
- should be tested with at least one oneDPL iterator based algorithm.
