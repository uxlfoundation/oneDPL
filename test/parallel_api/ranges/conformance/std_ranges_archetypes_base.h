// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _STD_RANGES_ARCHETYPES_BASE_H
#define _STD_RANGES_ARCHETYPES_BASE_H

#include "support/test_config.h"

#if _ENABLE_STD_RANGES_TESTING

#include <compare>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl_defs.h"
#endif

#ifndef TEST_ARCHETYPE_DELETE_ADDRESSOF
#    define TEST_ARCHETYPE_DELETE_ADDRESSOF 1
#endif

#if TEST_ARCHETYPE_DELETE_ADDRESSOF
#    define TEST_ARCHETYPE_DELETED_ADDRESSOF void operator&() const = delete;
#else
#    define TEST_ARCHETYPE_DELETED_ADDRESSOF
#endif

#define TEST_ARCHETYPE_DELETED_OPERATIONS(_Name)                                                                       \
    _Name(const _Name&) = delete;                                                                                      \
    _Name(_Name&&) = delete;                                                                                           \
    _Name& operator=(const _Name&) = delete;                                                                           \
    _Name& operator=(_Name&&) = delete;                                                                                \
    TEST_ARCHETYPE_DELETED_ADDRESSOF

#define TEST_ARCHETYPE_DEFAULTED_OPERATIONS(_Name)                                                                     \
    _Name(const _Name&) = default;                                                                                     \
    _Name(_Name&&) = default;                                                                                          \
    _Name& operator=(const _Name&) = default;                                                                          \
    _Name& operator=(_Name&&) = default;                                                                               \
    TEST_ARCHETYPE_DELETED_ADDRESSOF

#if TEST_DPCPP_BACKEND_PRESENT
#    define TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(_Name)                                                                \
        static_assert(std::is_trivially_copyable_v<_Name>);                                                            \
        static_assert(sycl::is_device_copyable_v<_Name>);
#else
#    define TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(_Name) static_assert(std::is_trivially_copyable_v<_Name>);
#endif

namespace test_std_ranges
{
namespace archetypes
{

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

    archetype_iterator() = default;
    explicit archetype_iterator(T* p) : ptr(p) {}

    T* base() const { return ptr; }

    reference operator*() const { return *ptr; }
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

template <typename T>
class plain_archetype_view
{
    T* first = nullptr;
    T* last = nullptr;

  public:
    plain_archetype_view() = default;
    plain_archetype_view(T* p, std::size_t n) : first(p), last(p + n) {}

    archetype_iterator<T> begin() const { return archetype_iterator<T>(first); }
    archetype_sentinel<T> end() const { return archetype_sentinel<T>(last); }
};

} // namespace archetypes
} // namespace test_std_ranges

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<test_std_ranges::archetypes::archetype_view<T>> = true;

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<test_std_ranges::archetypes::plain_archetype_view<T>> = true;
template <typename T>
inline constexpr bool std::ranges::enable_view<test_std_ranges::archetypes::plain_archetype_view<T>> = true;

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

static_assert(std::ranges::view<plain_archetype_view<int>>);
static_assert(std::ranges::random_access_range<plain_archetype_view<int>>);
static_assert(std::ranges::sized_range<plain_archetype_view<int>>);
static_assert(std::ranges::borrowed_range<plain_archetype_view<int>>);
static_assert(!std::ranges::contiguous_range<plain_archetype_view<int>>);
static_assert(!std::ranges::common_range<plain_archetype_view<int>>);
static_assert(std::same_as<std::ranges::range_reference_t<plain_archetype_view<int>>, int&>);

template <typename _R>
concept has_view_interface_members = requires(_R& __r) {
    __r.size();
    __r[0];
    __r.empty();
    __r.front();
};

static_assert(has_view_interface_members<archetype_view<int>>);
static_assert(!has_view_interface_members<plain_archetype_view<int>>);

static_assert(std::is_lvalue_reference_v<std::ranges::range_reference_t<archetype_view<int>>>);
static_assert(std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<archetype_view<int>>>,
                           std::ranges::range_value_t<archetype_view<int>>>);

template <typename T, typename Alloc>
class archetype_storage
{
    Alloc alloc;
    std::size_t count = 0;
    T* data = nullptr;

  public:
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

    template <template <typename> class _View = archetype_view>
    _View<T> view() const
    {
        return _View<T>(data, count);
    }
};


} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_BASE_H
