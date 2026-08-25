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

#ifndef _STD_RANGES_ARCHETYPES_H
#define _STD_RANGES_ARCHETYPES_H

#if _ENABLE_STD_RANGES_TESTING

#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>

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

namespace test_std_ranges
{
namespace archetypes
{

// std::default_initializable, required by uninitialized_default_construct.
// The default constructor is user-provided, so default- and value-initialization are the same and
// val2 is left untouched by the algorithm.
struct default_construct_archetype
{
    int val1;
    int val2;

    default_construct_archetype() { val1 = 1; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(default_construct_archetype)
};

static_assert(std::default_initializable<default_construct_archetype>);
static_assert(std::destructible<default_construct_archetype>);
static_assert(!std::copy_constructible<default_construct_archetype>);
static_assert(!std::move_constructible<default_construct_archetype>);
static_assert(!std::equality_comparable<default_construct_archetype>);
static_assert(!std::swappable<default_construct_archetype>);

// std::default_initializable, required by uninitialized_value_construct.
// The default constructor is defaulted on its first declaration and therefore is not user-provided:
// value-initialization zero-initializes the whole object, which lets the test tell value
// construction apart from default construction.
struct value_construct_archetype
{
    int val1;
    int val2;

    value_construct_archetype() = default;

    TEST_ARCHETYPE_DELETED_OPERATIONS(value_construct_archetype)
};

static_assert(std::default_initializable<value_construct_archetype>);
static_assert(std::destructible<value_construct_archetype>);
static_assert(!std::copy_constructible<value_construct_archetype>);
static_assert(!std::move_constructible<value_construct_archetype>);
static_assert(!std::equality_comparable<value_construct_archetype>);
static_assert(!std::swappable<value_construct_archetype>);

// The _T template parameter of uninitialized_fill is deduced from the value argument, so the filler
// type is deliberately different from the range value type: the only required conversion is
// std::constructible_from<range_value_t<_R>, const fill_source&>.
struct fill_source
{
    int val;
};

struct fill_archetype
{
    int val1;
    int val2;

    explicit fill_archetype(const fill_source& src) { val2 = src.val; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(fill_archetype)
};

static_assert(std::constructible_from<fill_archetype, const fill_source&>);
static_assert(std::destructible<fill_archetype>);
static_assert(!std::default_initializable<fill_archetype>);
static_assert(!std::copy_constructible<fill_archetype>);
static_assert(!std::move_constructible<fill_archetype>);

// Input element type of uninitialized_copy and uninitialized_move. No constraint is imposed on it
// besides forming a random access range, so it is only constructible from an int, which is what the
// test harness uses to prepare the input data.
struct transfer_source
{
    int val1;
    int val2;

    explicit transfer_source(int v) { val2 = v; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(transfer_source)
};

static_assert(std::destructible<transfer_source>);
static_assert(!std::default_initializable<transfer_source>);
static_assert(!std::copy_constructible<transfer_source>);
static_assert(!std::move_constructible<transfer_source>);

// std::constructible_from<range_value_t<_OutRange>, range_reference_t<_InRange>>, required by
// uninitialized_copy. range_reference_t of a range of transfer_source is exactly transfer_source&,
// so the implementation must not pass a const lvalue or an rvalue instead.
struct copy_archetype
{
    int val1;
    int val2;

    explicit copy_archetype(transfer_source& src) { val2 = src.val2; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(copy_archetype)
};

static_assert(std::constructible_from<copy_archetype, transfer_source&>);
static_assert(std::destructible<copy_archetype>);
static_assert(!std::constructible_from<copy_archetype, const transfer_source&>);
static_assert(!std::constructible_from<copy_archetype, transfer_source&&>);
static_assert(!std::default_initializable<copy_archetype>);
static_assert(!std::copy_constructible<copy_archetype>);

// std::constructible_from<range_value_t<_OutRange>, range_rvalue_reference_t<_InRange>>, required by
// uninitialized_move. Only an rvalue is accepted, so the implementation has to move the source
// element (std::ranges::iter_move) rather than copy it.
struct move_archetype
{
    int val1;
    int val2;

    explicit move_archetype(transfer_source&& src) { val2 = src.val2; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(move_archetype)
};

static_assert(std::constructible_from<move_archetype, transfer_source&&>);
static_assert(std::destructible<move_archetype>);
static_assert(!std::constructible_from<move_archetype, transfer_source&>);
static_assert(!std::default_initializable<move_archetype>);
static_assert(!std::copy_constructible<move_archetype>);

// std::destructible, required by destroy. No constructor at all is declared, which is enough for the
// test: the harness works on raw memory and only observes the effect of the destructor.
struct destroy_archetype
{
    int val1;
    volatile int val2; // volatile prevents optimization of the destructor observed with g++

    ~destroy_archetype() { val2 = 3; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(destroy_archetype)
};

static_assert(std::destructible<destroy_archetype>);
static_assert(!std::default_initializable<destroy_archetype>);
static_assert(!std::copy_constructible<destroy_archetype>);
static_assert(!std::move_constructible<destroy_archetype>);

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
// Archetypes for the algorithms of glue_algorithm_ranges_impl.h
//
// Every algorithm there constrains its range parameters with std::ranges::random_access_range and
// std::ranges::sized_range only; all the remaining requirements are expressed as indirect concepts
// on the iterators. The element archetypes below therefore drop everything a "regular" type would
// have and add back exactly the operations one concept family needs. archetype_view is reused as the
// range, so the ranges are random access and sized but neither contiguous nor common.
//------------------------------------------------------------------------------------------------

// Family 1: read-only algorithms parameterized by a callable.
// std::indirectly_unary_invocable / std::indirect_unary_predicate / std::indirect_strict_weak_order /
// std::indirect_equivalence_relation only require the callable to be invocable with the projected
// value; they impose nothing at all on the element type itself.
// Used by: for_each, find_if, find_if_not, find_last_if, find_last_if_not, any_of, all_of, none_of,
// count_if, is_partitioned, adjacent_find, is_sorted, is_sorted_until, is_heap, is_heap_until,
// min_element, max_element, minmax_element, lexicographical_compare, includes.
struct read_archetype
{
    int val;

    explicit read_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(read_archetype)
};

static_assert(std::destructible<read_archetype>);
static_assert(!std::default_initializable<read_archetype>);
static_assert(!std::copy_constructible<read_archetype>);
static_assert(!std::move_constructible<read_archetype>);
static_assert(!std::equality_comparable<read_archetype>);
static_assert(!std::totally_ordered<read_archetype>);

// The callables take exactly const _T& and return exactly the required type, so an implementation
// cannot pass an rvalue, a copy, or expect a wider return type.
struct read_unary_fun
{
    void operator()(const read_archetype&) const {}
};

struct read_unary_pred
{
    bool operator()(const read_archetype& __v) const { return __v.val % 3 == 0; }
};

struct read_binary_pred
{
    bool operator()(const read_archetype& __v1, const read_archetype& __v2) const { return __v1.val == __v2.val; }
};

struct read_comp
{
    bool operator()(const read_archetype& __v1, const read_archetype& __v2) const { return __v1.val < __v2.val; }
};

// A projection which returns a prvalue of an unrelated type, so nothing links the projected type
// back to the element type.
struct read_proj_result
{
    int val;
};

struct read_proj
{
    read_proj_result operator()(const read_archetype& __v) const { return read_proj_result{__v.val}; }
};

struct read_proj_pred
{
    bool operator()(const read_proj_result& __v) const { return __v.val % 3 == 0; }
};

using read_iterator_t = std::ranges::iterator_t<archetype_view<read_archetype>>;

static_assert(std::indirectly_unary_invocable<read_unary_fun, read_iterator_t>);
static_assert(std::indirect_unary_predicate<read_unary_pred, read_iterator_t>);
static_assert(std::indirect_binary_predicate<read_binary_pred, read_iterator_t, read_iterator_t>);
static_assert(std::indirect_strict_weak_order<read_comp, read_iterator_t>);
static_assert(std::indirect_unary_predicate<read_proj_pred, std::projected<read_iterator_t, read_proj>>);

// Family 2: algorithms taking a search value.
// The constraint is
//   std::indirect_binary_predicate<std::ranges::equal_to, std::projected<iterator_t<_R>, _Proj>,
//                                  const _T*>
// std::ranges::equal_to is itself constrained by std::equality_comparable_with, which is much
// stronger than a bare `element == value`: both types have to be equality comparable with
// themselves and to share a common reference type. The archetypes below provide exactly that and
// nothing else, in particular they are still neither copyable nor movable.
// Used by: find, find_last, count, contains, remove, remove_copy, replace, replace_copy.
// The value is passed to a device kernel by copy, so, unlike the other archetypes, it has to be
// trivially copyable and thus device copyable. Everything else a "regular" type provides is still
// missing: no default constructor, no ordering, no relation to the element type but equality.
struct nocopy_search_value;

struct search_value
{
    int val;

    explicit search_value(int __v) : val(__v) {}

    search_value(const search_value&) = default;
    search_value& operator=(const search_value&) = default;

    friend bool operator==(const search_value& __v1, const search_value& __v2) { return __v1.val == __v2.val; }
};

struct searchable_archetype
{
    int val;

    explicit searchable_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(searchable_archetype)

    friend bool operator==(const searchable_archetype& __e1, const searchable_archetype& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const searchable_archetype& __e, const search_value& __v) { return __e.val == __v.val; }

    friend bool operator==(const searchable_archetype& __e, const nocopy_search_value& __v);
};

// Family 2b: the very same constraint, but the search value is neither copyable nor movable.
// std::indirect_binary_predicate<std::ranges::equal_to, std::projected<iterator_t<_R>, _Proj>,
// const _T*> says nothing about copying _T, so a host policy must keep a reference to the value
// instead of storing a copy of it. A device policy legitimately copies the value into the kernel,
// so this archetype is only ever used with the host policies.
struct nocopy_search_value
{
    int val;

    explicit nocopy_search_value(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(nocopy_search_value)

    friend bool operator==(const nocopy_search_value& __v1, const nocopy_search_value& __v2)
    {
        return __v1.val == __v2.val;
    }
};

inline bool
operator==(const searchable_archetype& __e, const nocopy_search_value& __v)
{
    return __e.val == __v.val;
}

// The element archetype of the removing algorithms. remove() requires
//   std::permutable<iterator_t<_R>> && indirect_binary_predicate<std::ranges::equal_to, ...>
// so the element has to be movable, but still not copyable and not default constructible.
struct removable_archetype
{
    int val;

    explicit removable_archetype(int __v) : val(__v) {}

    removable_archetype(removable_archetype&& __other) : val(__other.val) {}

    removable_archetype&
    operator=(removable_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    removable_archetype(const removable_archetype&) = delete;
    removable_archetype& operator=(const removable_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    friend bool operator==(const removable_archetype& __e1, const removable_archetype& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const removable_archetype& __e, const nocopy_search_value& __v)
    {
        return __e.val == __v.val;
    }

    friend bool operator==(const removable_archetype& __e, const search_value& __v) { return __e.val == __v.val; }
};

// The common reference required by std::equality_comparable_with. It is only ever formed as a
// reference by the concept machinery, so a minimal type which both archetypes convert to is enough.
struct search_common
{
    int val;

    search_common(const searchable_archetype& __e) : val(__e.val) {}
    search_common(const removable_archetype& __e) : val(__e.val) {}
    search_common(const search_value& __v) : val(__v.val) {}
    search_common(const nocopy_search_value& __v) : val(__v.val) {}

    friend bool operator==(const search_common& __v1, const search_common& __v2) { return __v1.val == __v2.val; }
};

} // namespace archetypes
} // namespace test_std_ranges

namespace std
{
template <>
struct common_type<test_std_ranges::archetypes::searchable_archetype, test_std_ranges::archetypes::search_value>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::search_value, test_std_ranges::archetypes::searchable_archetype>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::searchable_archetype, test_std_ranges::archetypes::nocopy_search_value>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::nocopy_search_value, test_std_ranges::archetypes::searchable_archetype>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::removable_archetype, test_std_ranges::archetypes::search_value>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::search_value, test_std_ranges::archetypes::removable_archetype>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::removable_archetype, test_std_ranges::archetypes::nocopy_search_value>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::nocopy_search_value, test_std_ranges::archetypes::removable_archetype>
{
    using type = test_std_ranges::archetypes::search_common;
};
} // namespace std

namespace test_std_ranges
{
namespace archetypes
{

using searchable_iterator_t = std::ranges::iterator_t<archetype_view<searchable_archetype>>;

static_assert(std::indirect_binary_predicate<std::ranges::equal_to, searchable_iterator_t, const search_value*>);
static_assert(
    std::indirect_binary_predicate<std::ranges::equal_to, searchable_iterator_t, const nocopy_search_value*>);
static_assert(!std::copy_constructible<nocopy_search_value>);
static_assert(!std::move_constructible<nocopy_search_value>);
static_assert(!std::default_initializable<nocopy_search_value>);

using removable_iterator_t = std::ranges::iterator_t<archetype_view<removable_archetype>>;

static_assert(std::permutable<removable_iterator_t>);
static_assert(std::indirect_binary_predicate<std::ranges::equal_to, removable_iterator_t, const search_value*>);
static_assert(std::indirect_binary_predicate<std::ranges::equal_to, removable_iterator_t, const nocopy_search_value*>);
static_assert(!std::copy_constructible<removable_archetype>);
static_assert(!std::default_initializable<removable_archetype>);
static_assert(!std::totally_ordered<removable_archetype>);
static_assert(!std::copy_constructible<searchable_archetype>);
static_assert(!std::move_constructible<searchable_archetype>);
static_assert(std::is_trivially_copyable_v<search_value>);
static_assert(!std::default_initializable<search_value>);
static_assert(!std::totally_ordered<search_value>);
static_assert(!std::default_initializable<searchable_archetype>);
static_assert(!std::totally_ordered<searchable_archetype>);

// Family 3: two-range algorithms constrained by std::indirectly_comparable.
// std::indirectly_comparable<It1, It2, _Pred, _Proj1, _Proj2> only asks for the predicate to be
// invocable on the two projected references, so the two element types stay unrelated and neither of
// them is comparable with itself.
// Used by: equal, mismatch, search, find_end, find_first_of, contains_subrange, starts_with,
// ends_with.
struct lhs_archetype
{
    int val;

    explicit lhs_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(lhs_archetype)
};

struct rhs_archetype
{
    int val;

    explicit rhs_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(rhs_archetype)
};

struct cross_pred
{
    bool operator()(const lhs_archetype& __v1, const rhs_archetype& __v2) const { return __v1.val == __v2.val; }
};

using lhs_iterator_t = std::ranges::iterator_t<archetype_view<lhs_archetype>>;
using rhs_iterator_t = std::ranges::iterator_t<archetype_view<rhs_archetype>>;

static_assert(std::indirectly_comparable<lhs_iterator_t, rhs_iterator_t, cross_pred>);
static_assert(!std::equality_comparable<lhs_archetype>);
static_assert(!std::equality_comparable<rhs_archetype>);
static_assert(!std::copy_constructible<lhs_archetype>);
static_assert(!std::copy_constructible<rhs_archetype>);

// Family 4: algorithms writing a value into the range itself.
// The constraint is std::indirectly_writable<iterator_t<_R>, const _T&>, which needs `*it = value`
// for a const lvalue value and nothing else: the element still does not have to be copyable,
// movable or default constructible, and _T stays an unrelated type.
// Used by: fill, replace_if, replace (new value), replace_copy_if / replace_copy (new value).
struct write_value
{
    int val;

    explicit write_value(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(write_value)
};

struct writable_archetype
{
    int val;

    explicit writable_archetype(int __v) : val(__v) {}

    writable_archetype(const writable_archetype&) = delete;
    writable_archetype(writable_archetype&&) = delete;
    writable_archetype& operator=(const writable_archetype&) = delete;
    writable_archetype& operator=(writable_archetype&&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    writable_archetype& operator=(const write_value& __v)
    {
        val = __v.val;
        return *this;
    }
};

using writable_iterator_t = std::ranges::iterator_t<archetype_view<writable_archetype>>;

static_assert(std::indirectly_writable<writable_iterator_t, const write_value&>);
static_assert(!std::copyable<writable_archetype>);
static_assert(!std::movable<writable_archetype>);
static_assert(!std::default_initializable<writable_archetype>);

// Family 5: copying algorithms.
// std::indirectly_copyable<In, Out> == indirectly_readable<In> && indirectly_writable<Out,
// iter_reference_t<In>>, so the output element only has to be assignable from a non-const lvalue of
// the input element type. Neither element type has to be copyable, movable or default
// constructible, and the two types are deliberately different.
// Used by: copy, copy_if, reverse_copy, rotate_copy, remove_copy, remove_copy_if, unique_copy,
// replace_copy, replace_copy_if, partition_copy, partial_sort_copy.
struct copy_in_archetype
{
    int val;

    explicit copy_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(copy_in_archetype)
};

struct copy_out_archetype
{
    int val;

    explicit copy_out_archetype(int __v) : val(__v) {}

    copy_out_archetype(const copy_out_archetype&) = delete;
    copy_out_archetype(copy_out_archetype&&) = delete;
    copy_out_archetype& operator=(const copy_out_archetype&) = delete;
    copy_out_archetype& operator=(copy_out_archetype&&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    copy_out_archetype& operator=(copy_in_archetype& __v)
    {
        val = __v.val;
        return *this;
    }
};

using copy_in_iterator_t = std::ranges::iterator_t<archetype_view<copy_in_archetype>>;
using copy_out_iterator_t = std::ranges::iterator_t<archetype_view<copy_out_archetype>>;

static_assert(std::indirectly_copyable<copy_in_iterator_t, copy_out_iterator_t>);
static_assert(!std::copyable<copy_in_archetype>);
static_assert(!std::copyable<copy_out_archetype>);
static_assert(!std::default_initializable<copy_out_archetype>);

// Family 6: the move algorithm.
// std::indirectly_movable<In, Out> asks for indirectly_writable<Out, iter_rvalue_reference_t<In>>,
// so the output element is only assignable from an rvalue of the input element type: an
// implementation which copies instead of moving does not compile.
struct move_in_archetype
{
    int val;

    explicit move_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(move_in_archetype)
};

struct move_out_archetype
{
    int val;

    explicit move_out_archetype(int __v) : val(__v) {}

    move_out_archetype(const move_out_archetype&) = delete;
    move_out_archetype(move_out_archetype&&) = delete;
    move_out_archetype& operator=(const move_out_archetype&) = delete;
    move_out_archetype& operator=(move_out_archetype&&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    move_out_archetype& operator=(move_in_archetype&& __v)
    {
        val = __v.val;
        return *this;
    }
};

using move_in_iterator_t = std::ranges::iterator_t<archetype_view<move_in_archetype>>;
using move_out_iterator_t = std::ranges::iterator_t<archetype_view<move_out_archetype>>;

static_assert(std::indirectly_movable<move_in_iterator_t, move_out_iterator_t>);
// An lvalue is explicitly rejected, so copying instead of moving is a compilation error.
static_assert(!std::indirectly_copyable<move_in_iterator_t, move_out_iterator_t>);
static_assert(!std::movable<move_out_archetype>);

// Family 7: swap_ranges.
// std::indirectly_swappable<It1, It2> needs std::ranges::swap on the two references, both ways. A
// dedicated hidden-friend swap is provided, so the element does not have to be move constructible
// or move assignable, which is what the fallback std::swap would require.
struct swap_archetype
{
    int val;

    explicit swap_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(swap_archetype)

    friend void swap(swap_archetype& __v1, swap_archetype& __v2)
    {
        const int __tmp = __v1.val;
        __v1.val = __v2.val;
        __v2.val = __tmp;
    }
};

using swap_iterator_t = std::ranges::iterator_t<archetype_view<swap_archetype>>;

static_assert(std::indirectly_swappable<swap_iterator_t, swap_iterator_t>);
static_assert(!std::movable<swap_archetype>);
static_assert(!std::move_constructible<swap_archetype>);
static_assert(!std::default_initializable<swap_archetype>);

// Family 8: transform.
// The output constraint is
//   std::indirectly_writable<iterator_t<_OutRange>, std::indirect_result_t<_F&, projected...>>
// so the output element is only assignable from the result of the functor, which is a third,
// unrelated type. _F itself is only required to be std::copy_constructible.
struct transform_in_archetype
{
    int val;

    explicit transform_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(transform_in_archetype)
};

// The result of the functor. indirectly_writable requires the assignment to work for the prvalue,
// the const lvalue and the const rvalue forms of the result type, which a prvalue-returning functor
// naturally provides.
struct transform_result
{
    int val;
};

struct transform_out_archetype
{
    int val;

    explicit transform_out_archetype(int __v) : val(__v) {}

    transform_out_archetype(const transform_out_archetype&) = delete;
    transform_out_archetype(transform_out_archetype&&) = delete;
    transform_out_archetype& operator=(const transform_out_archetype&) = delete;
    transform_out_archetype& operator=(transform_out_archetype&&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    transform_out_archetype& operator=(const transform_result& __v)
    {
        val = __v.val;
        return *this;
    }
};

struct transform_unary_op
{
    transform_result operator()(const transform_in_archetype& __v) const { return transform_result{__v.val * 2}; }
};

struct transform_binary_op
{
    transform_result operator()(const transform_in_archetype& __v1, const transform_in_archetype& __v2) const
    {
        return transform_result{__v1.val + __v2.val};
    }
};

using transform_in_iterator_t = std::ranges::iterator_t<archetype_view<transform_in_archetype>>;
using transform_out_iterator_t = std::ranges::iterator_t<archetype_view<transform_out_archetype>>;

static_assert(std::copy_constructible<transform_unary_op>);
static_assert(std::copy_constructible<transform_binary_op>);
static_assert(std::indirectly_writable<transform_out_iterator_t,
                                       std::indirect_result_t<transform_unary_op&, transform_in_iterator_t>>);
static_assert(std::indirectly_writable<
              transform_out_iterator_t,
              std::indirect_result_t<transform_binary_op&, transform_in_iterator_t, transform_in_iterator_t>>);
static_assert(!std::copyable<transform_out_archetype>);
static_assert(!std::default_initializable<transform_out_archetype>);

// Family 9: permuting algorithms.
// std::permutable<It> == forward_iterator<It> && indirectly_movable_storable<It, It> &&
// indirectly_swappable<It, It>, which does require the element to be movable and move
// constructible, but still not copyable, not default constructible and not comparable.
// Used by: reverse, rotate, shift_left, shift_right, remove_if, remove, unique, partition,
// stable_partition.
struct permutable_archetype
{
    int val;

    explicit permutable_archetype(int __v) : val(__v) {}

    permutable_archetype(permutable_archetype&& __other) : val(__other.val) {}

    permutable_archetype& operator=(permutable_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    permutable_archetype(const permutable_archetype&) = delete;
    permutable_archetype& operator=(const permutable_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF
};

using permutable_iterator_t = std::ranges::iterator_t<archetype_view<permutable_archetype>>;

static_assert(std::permutable<permutable_iterator_t>);
static_assert(!std::copy_constructible<permutable_archetype>);
static_assert(!std::default_initializable<permutable_archetype>);
static_assert(!std::equality_comparable<permutable_archetype>);
static_assert(!std::totally_ordered<permutable_archetype>);

// The predicate and the comparator of the permuting algorithms only see the projected reference.
struct permutable_pred
{
    bool operator()(const permutable_archetype& __v) const { return __v.val % 3 == 0; }
};

struct permutable_equiv
{
    bool operator()(const permutable_archetype& __v1, const permutable_archetype& __v2) const
    {
        return __v1.val == __v2.val;
    }
};

// std::sortable<It, _Comp, _Proj> == permutable<It> && indirect_strict_weak_order<_Comp,
// projected<It, _Proj>>, so the very same element archetype works and the ordering has to come from
// the comparator, never from an operator< on the element.
// Used by: sort, stable_sort, partial_sort, inplace_merge, nth_element, partial_sort_copy.
struct permutable_comp
{
    bool operator()(const permutable_archetype& __v1, const permutable_archetype& __v2) const
    {
        return __v1.val < __v2.val;
    }
};

static_assert(std::sortable<permutable_iterator_t, permutable_comp>);

// The merge family additionally needs std::indirectly_copyable from both inputs into the output.
// The output element is therefore assignable from a non-const lvalue of either input element type,
// while remaining non-copyable itself.
// Used by: merge, set_union, set_intersection, set_difference, set_symmetric_difference.
struct merge_in_archetype
{
    int val;

    explicit merge_in_archetype(int __v) : val(__v) {}

    merge_in_archetype(merge_in_archetype&& __other) : val(__other.val) {}

    merge_in_archetype& operator=(merge_in_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    merge_in_archetype(const merge_in_archetype&) = delete;
    merge_in_archetype& operator=(const merge_in_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF
};

struct merge_out_archetype
{
    int val;

    explicit merge_out_archetype(int __v) : val(__v) {}

    merge_out_archetype(merge_out_archetype&& __other) : val(__other.val) {}

    merge_out_archetype& operator=(merge_out_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    merge_out_archetype(const merge_out_archetype&) = delete;
    merge_out_archetype& operator=(const merge_out_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    merge_out_archetype& operator=(merge_in_archetype& __v)
    {
        val = __v.val;
        return *this;
    }
};

struct merge_comp
{
    bool operator()(const merge_in_archetype& __v1, const merge_in_archetype& __v2) const
    {
        return __v1.val < __v2.val;
    }
};

using merge_in_iterator_t = std::ranges::iterator_t<archetype_view<merge_in_archetype>>;
using merge_out_iterator_t = std::ranges::iterator_t<archetype_view<merge_out_archetype>>;

static_assert(std::mergeable<merge_in_iterator_t, merge_in_iterator_t, merge_out_iterator_t, merge_comp>);
static_assert(!std::copy_constructible<merge_in_archetype>);
static_assert(!std::copy_constructible<merge_out_archetype>);
static_assert(!std::default_initializable<merge_out_archetype>);

// min / max / minmax additionally require
// std::indirectly_copyable_storable<iterator_t<_R>, range_value_t<_R>*>, which does need a copy
// constructor and copy assignment, but still no default constructor and no ordering operator.
struct storable_archetype
{
    int val;

    explicit storable_archetype(int __v) : val(__v) {}

    storable_archetype(const storable_archetype& __other) : val(__other.val) {}

    storable_archetype& operator=(const storable_archetype& __other)
    {
        val = __other.val;
        return *this;
    }

    TEST_ARCHETYPE_DELETED_ADDRESSOF
};

struct storable_comp
{
    bool operator()(const storable_archetype& __v1, const storable_archetype& __v2) const
    {
        return __v1.val < __v2.val;
    }
};

using storable_iterator_t = std::ranges::iterator_t<archetype_view<storable_archetype>>;

static_assert(std::indirectly_copyable_storable<storable_iterator_t, storable_archetype*>);
static_assert(std::indirect_strict_weak_order<storable_comp, storable_iterator_t>);
static_assert(!std::default_initializable<storable_archetype>);
static_assert(!std::equality_comparable<storable_archetype>);
static_assert(!std::totally_ordered<storable_archetype>);

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_H
