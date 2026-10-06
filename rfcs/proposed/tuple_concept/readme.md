# tuple like concept for the oneDPL C++20

## Introduction
A tuple like type in C++ is a powerful utility that enables developers to combine two or more types into a single one - a tuple.
There are very useful other types based on a tuple like type. For example, zip iterator or zip view types.
C++ standard library indroduces `std::tuple` type and `std::ranges::zip_view` is based on `std::tuple`.

## Motivation
In spite of `std::tuple` is present in C++ standard library, there are unresolvable issues in usage of zip iterator or zip view types
(based on `std::tuple`) with parallel algorithms and oneDPL parallel algorithms in particular.
- Dereferenced zip_view iterator is not value-swappable in C++20 (https://en.cppreference.com/w/cpp/named_req/ValueSwappable)
So, it is not able to be used with the non-range algorithms
- Usage of `std::tuple` as  `value_type` for zip_view iterator leads to the issues, at least, with `sortable`, `permutable`
  and `indirectly_writable` concepts: const_cast<const std::iter_reference_t<Out>&&>(*o) = std::forward<T>(t) is not compiled till C++23.  (https://godbolt.org/z/zT9qqnjWq)
- There is an issue with `std::ranges::sort(zip_view)` with clang 19.0 and older. (https://godbolt.org/z/jKvG9rY5M)
- There is an issue with `std::ranges::stable_sort(zip_view)` with gcc library 
- Passing `std::zip_view::iterator` instances to the iterator-based algorithms works only for gcc 14.1 and newer, clang 19.1 and newer or
  starting 17.01 with libc++ lib (https://godbolt.org/z/To6Mjr9M6)
- Any wrapper over `std::tuple` is not device copyable. (https://godbolt.org/z/brfvcMeM6). A device copyability of the `std::tuple` is not transitive property.

oneDPL has internal tuple like type without the issues mentioned above.
But, we would not specify that internal type in oneDPL specification. Rather we would have a `tuple like concept` for oneDPL.

### Key Requirements
A tuple type, which is satisfying a `tuple like concept`, should be:
- Explicitly convertible to `std::tuple`.
- A SYCL device copyable type.
- Elements of such typel type can be accessed using structured bindings.
- The type can be used with `std::get`, `std::tuple_element` and `std::tuple_size`, `std::apply`.
- The type is lexicographically comparable the values in the tuple. Other words, the following operators should be defined for the type:
  `operator==`, `operator!=`, `operator<`, `operator<=`, `operator>`, `operator>=`, `operator<=>`.
- The standard typrs `std::common_type` and `std::basic_common_reference` should be specified for the type.
  
### Implementation proposal (C++20)
- For C++20 oneDPL introduces a C++ concept, for example, `oneapi::dpl::tuple`. A proper name may be discussed.
- This C++ concept is defined via a list of C++ requirements, corresponding to the `Key Requirements` list above.

### Test coverage
- The test cases should cover as the whole concept and as each require from the `Key Requirements` list for `oneapi::dpl::__internal::tuple` type.
