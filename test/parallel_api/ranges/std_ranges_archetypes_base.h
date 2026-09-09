// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _STD_RANGES_ARCHETYPES_BASE_H
#define _STD_RANGES_ARCHETYPES_BASE_H

// test_config.h defines both _ENABLE_STD_RANGES_TESTING and TEST_DPCPP_BACKEND_PRESENT, so it has to
// come before the checks below: without it the whole header would silently compile to nothing.
#include "support/test_config.h"

#if _ENABLE_STD_RANGES_TESTING

#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl_defs.h"
#endif

// The types below are "archetypes": each of them satisfies exactly the constraints written in the
// requires-clause of the corresponding oneapi::dpl::ranges algorithm and nothing more. Every
// operation which is not implied by those constraints is explicitly deleted. If an algorithm
// compiles and works with an archetype, the implementation does not silently require more from a
// user type than it declares; otherwise the extra requirement shows up as a compilation error.
//
// Each archetype keeps two observable fields, val1 and val2, so that a test can check which part of
// the raw memory has been written, exactly as the pre-existing Elem/Elem_0 types do.

// Unary operator& is not required by any constraint, so a conforming implementation has to use
// std::addressof instead of taking the address directly. Define this macro to 0 to relax the
// archetypes if the deleted operator& hides other findings.
#ifndef TEST_ARCHETYPE_DELETE_ADDRESSOF
#    define TEST_ARCHETYPE_DELETE_ADDRESSOF 1
#endif

#if TEST_ARCHETYPE_DELETE_ADDRESSOF
#    define TEST_ARCHETYPE_DELETED_ADDRESSOF void operator&() const = delete;
#else
#    define TEST_ARCHETYPE_DELETED_ADDRESSOF
#endif

// Deletes everything a "regular" type would provide but no constraint of the tested algorithms asks
// for: copying, moving, assignment and taking the address.
#define TEST_ARCHETYPE_DELETED_OPERATIONS(_Name)                                                                       \
    _Name(const _Name&) = delete;                                                                                      \
    _Name(_Name&&) = delete;                                                                                           \
    _Name& operator=(const _Name&) = delete;                                                                           \
    _Name& operator=(_Name&&) = delete;                                                                                \
    TEST_ARCHETYPE_DELETED_ADDRESSOF

// The device copyable counterpart of TEST_ARCHETYPE_DELETED_OPERATIONS: the copy and the move
// operations are trivial, which makes the type trivially copyable and thus device copyable by
// default, while everything else stays exactly as restricted as in the host only archetype.
#define TEST_ARCHETYPE_DEFAULTED_OPERATIONS(_Name)                                                                     \
    _Name(const _Name&) = default;                                                                                     \
    _Name(_Name&&) = default;                                                                                          \
    _Name& operator=(const _Name&) = default;                                                                          \
    _Name& operator=(_Name&&) = default;                                                                               \
    TEST_ARCHETYPE_DELETED_ADDRESSOF

// Checks that a device copyable archetype really is accepted by SYCL without an explicit
// sycl::is_device_copyable specialization.
#if TEST_DPCPP_BACKEND_PRESENT
#    define TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(_Name)                                                                 \
        static_assert(std::is_trivially_copyable_v<_Name>);                                                             \
        static_assert(sycl::is_device_copyable_v<_Name>);
#else
#    define TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(_Name) static_assert(std::is_trivially_copyable_v<_Name>);
#endif

namespace test_std_ranges
{
namespace archetypes
{

// A random access iterator which is deliberately not a contiguous one. Unlike a pointer, a span
// iterator or a subrange over pointers, it gives the implementation no way to fall back to raw
// pointer arithmetic on the underlying storage.
template <typename T>
class archetype_iterator
{
    T* ptr = nullptr;

  public:
    using iterator_concept = std::random_access_iterator_tag;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using pointer = T*;

    archetype_iterator() = default;
    explicit archetype_iterator(T* p) : ptr(p) {}

    T* base() const { return ptr; }

    reference operator*() const { return *ptr; }
    pointer operator->() const { return ptr; }
    reference operator[](difference_type n) const { return ptr[n]; }

    archetype_iterator& operator++() { ++ptr; return *this; }
    archetype_iterator operator++(int) { auto tmp = *this; ++ptr; return tmp; }
    archetype_iterator& operator--() { --ptr; return *this; }
    archetype_iterator operator--(int) { auto tmp = *this; --ptr; return tmp; }

    archetype_iterator& operator+=(difference_type n) { ptr += n; return *this; }
    archetype_iterator& operator-=(difference_type n) { ptr -= n; return *this; }

    friend archetype_iterator operator+(archetype_iterator i, difference_type n) { return i += n; }
    friend archetype_iterator operator+(difference_type n, archetype_iterator i) { return i += n; }
    friend archetype_iterator operator-(archetype_iterator i, difference_type n) { return i -= n; }
    friend difference_type operator-(archetype_iterator i, archetype_iterator j) { return i.ptr - j.ptr; }

    friend bool operator==(archetype_iterator i, archetype_iterator j) { return i.ptr == j.ptr; }
    friend auto operator<=>(archetype_iterator i, archetype_iterator j) { return i.ptr <=> j.ptr; }
};

// A sentinel type distinct from the iterator, which makes the range non-common while keeping it
// sized via the sized_sentinel_for requirement.
template <typename T>
class archetype_sentinel
{
    T* ptr = nullptr;

  public:
    archetype_sentinel() = default;
    explicit archetype_sentinel(T* p) : ptr(p) {}

    T* base() const { return ptr; }

    friend bool operator==(archetype_iterator<T> i, archetype_sentinel s) { return i.base() == s.ptr; }
    friend std::ptrdiff_t operator-(archetype_iterator<T> i, archetype_sentinel s) { return i.base() - s.ptr; }
    friend std::ptrdiff_t operator-(archetype_sentinel s, archetype_iterator<T> i) { return s.ptr - i.base(); }
};

// A view over raw storage which satisfies __nothrow_random_access_range and sized_range, but is
// neither contiguous nor common. It is marked as a borrowed range so that the algorithms keep
// returning a real iterator rather than std::ranges::dangling.
template <typename T>
class archetype_view : public std::ranges::view_interface<archetype_view<T>>
{
    T* first = nullptr;
    T* last = nullptr;

  public:
    archetype_view() = default;
    archetype_view(T* p, std::size_t n) : first(p), last(p + n) {}

    archetype_iterator<T> begin() const { return archetype_iterator<T>(first); }
    archetype_sentinel<T> end() const { return archetype_sentinel<T>(last); }
};

} // namespace archetypes
} // namespace test_std_ranges

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<test_std_ranges::archetypes::archetype_view<T>> = true;

namespace test_std_ranges
{
namespace archetypes
{

static_assert(std::random_access_iterator<archetype_iterator<int>>);
static_assert(!std::contiguous_iterator<archetype_iterator<int>>);
static_assert(std::sized_sentinel_for<archetype_sentinel<int>, archetype_iterator<int>>);

static_assert(std::ranges::random_access_range<archetype_view<int>>);
static_assert(std::ranges::sized_range<archetype_view<int>>);
static_assert(std::ranges::borrowed_range<archetype_view<int>>);
static_assert(!std::ranges::contiguous_range<archetype_view<int>>);
static_assert(!std::ranges::common_range<archetype_view<int>>);

// The two extra requirements of __nothrow_random_access_range beyond random_access_range.
static_assert(std::is_lvalue_reference_v<std::ranges::range_reference_t<archetype_view<int>>>);
static_assert(std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<archetype_view<int>>>,
                           std::ranges::range_value_t<archetype_view<int>>>);

// Owns raw storage and constructs the elements in place. The archetypes are neither copyable nor
// movable, so they cannot be kept in a standard container; the allocator is a template parameter so
// that the very same storage works with std::allocator on the host and with sycl::usm_allocator on
// a device.
template <typename T, typename Alloc>
class archetype_storage
{
    Alloc alloc;
    std::size_t count = 0;
    T* data = nullptr;

  public:
    // _Factory is called as __factory(i) for every index and has to return the arguments of the
    // element constructor.
    template <typename _Factory>
    archetype_storage(Alloc __alloc, std::size_t __n, _Factory __factory) : alloc(__alloc), count(__n)
    {
        data = alloc.allocate(count);
        for (std::size_t __i = 0; __i < count; ++__i)
            std::construct_at(data + __i, __factory(__i));
    }

    archetype_storage(const archetype_storage&) = delete;
    archetype_storage& operator=(const archetype_storage&) = delete;

    ~archetype_storage()
    {
        for (std::size_t __i = 0; __i < count; ++__i)
            std::destroy_at(data + __i);
        alloc.deallocate(data, count);
    }

    std::size_t size() const { return count; }
    T* begin_ptr() const { return data; }

    archetype_view<T> view() const { return archetype_view<T>(data, count); }
};


//------------------------------------------------------------------------------------------------
// Callables taking their arguments by non-const reference.
//
// std::indirectly_unary_invocable, std::indirect_unary_predicate, std::indirect_binary_predicate,
// std::indirect_strict_weak_order and std::projected are all spelled in terms of iter_value_t<_It>&,
// iter_reference_t<_It> and iter_common_reference_t<_It>. For archetype_view<_T> all three of them
// are _T&, i.e. a non-const lvalue reference, so a callable which accepts nothing but _T& satisfies
// those concepts. The requires-clauses of the algorithms therefore allow such a callable, and an
// implementation which hands a const lvalue, an rvalue or a copy of the element to the user callable
// does not compile with the types below.
//
// The _mut counterparts only add the non-const parameter list; the element archetypes and the
// expected results stay exactly the ones of the corresponding family above.
//------------------------------------------------------------------------------------------------

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_BASE_H
