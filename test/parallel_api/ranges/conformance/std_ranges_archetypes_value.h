// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _STD_RANGES_ARCHETYPES_VALUE_H
#define _STD_RANGES_ARCHETYPES_VALUE_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"

namespace test_std_ranges
{
namespace archetypes
{

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

    using nocopy_value_type = nocopy_search_value;

    explicit searchable_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(searchable_archetype)

    friend bool operator==(const searchable_archetype& __e1, const searchable_archetype& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const searchable_archetype& __e, const search_value& __v) { return __e.val == __v.val; }

    friend bool operator==(const searchable_archetype& __e, const nocopy_search_value& __v);
};

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

struct nocopy_search_value_dc
{
    int val;

    explicit nocopy_search_value_dc(int __v) : val(__v) {}

    nocopy_search_value_dc(const nocopy_search_value_dc&) = default;
    nocopy_search_value_dc& operator=(const nocopy_search_value_dc&) = default;

    friend bool operator==(const nocopy_search_value_dc& __v1, const nocopy_search_value_dc& __v2)
    {
        return __v1.val == __v2.val;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(nocopy_search_value_dc)
static_assert(!std::default_initializable<nocopy_search_value_dc>);
static_assert(!std::totally_ordered<nocopy_search_value_dc>);

struct removable_archetype
{
    int val;

    using nocopy_value_type = nocopy_search_value;

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

struct searchable_archetype_dc
{
    int val;

    using nocopy_value_type = nocopy_search_value_dc;

    explicit searchable_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(searchable_archetype_dc)

    friend bool operator==(const searchable_archetype_dc& __e1, const searchable_archetype_dc& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const searchable_archetype_dc& __e, const search_value& __v) { return __e.val == __v.val; }

    friend bool operator==(const searchable_archetype_dc& __e, const nocopy_search_value& __v)
    {
        return __e.val == __v.val;
    }

    friend bool operator==(const searchable_archetype_dc& __e, const nocopy_search_value_dc& __v)
    {
        return __e.val == __v.val;
    }
};

struct removable_archetype_dc
{
    int val;

    using nocopy_value_type = nocopy_search_value_dc;

    explicit removable_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(removable_archetype_dc)

    friend bool operator==(const removable_archetype_dc& __e1, const removable_archetype_dc& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const removable_archetype_dc& __e, const search_value& __v) { return __e.val == __v.val; }

    friend bool operator==(const removable_archetype_dc& __e, const nocopy_search_value& __v)
    {
        return __e.val == __v.val;
    }

    friend bool operator==(const removable_archetype_dc& __e, const nocopy_search_value_dc& __v)
    {
        return __e.val == __v.val;
    }
};

struct search_common
{
    int val;

    search_common(const searchable_archetype& __e) : val(__e.val) {}
    search_common(const removable_archetype& __e) : val(__e.val) {}
    search_common(const searchable_archetype_dc& __e) : val(__e.val) {}
    search_common(const removable_archetype_dc& __e) : val(__e.val) {}
    search_common(const search_value& __v) : val(__v.val) {}
    search_common(const nocopy_search_value& __v) : val(__v.val) {}
    search_common(const nocopy_search_value_dc& __v) : val(__v.val) {}

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

template <>
struct common_type<test_std_ranges::archetypes::searchable_archetype_dc, test_std_ranges::archetypes::search_value>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::search_value, test_std_ranges::archetypes::searchable_archetype_dc>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::searchable_archetype_dc,
                   test_std_ranges::archetypes::nocopy_search_value>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::nocopy_search_value,
                   test_std_ranges::archetypes::searchable_archetype_dc>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::removable_archetype_dc, test_std_ranges::archetypes::search_value>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::search_value, test_std_ranges::archetypes::removable_archetype_dc>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::removable_archetype_dc,
                   test_std_ranges::archetypes::nocopy_search_value>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::nocopy_search_value,
                   test_std_ranges::archetypes::removable_archetype_dc>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::searchable_archetype_dc,
                   test_std_ranges::archetypes::nocopy_search_value_dc>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::nocopy_search_value_dc,
                   test_std_ranges::archetypes::searchable_archetype_dc>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::removable_archetype_dc,
                   test_std_ranges::archetypes::nocopy_search_value_dc>
{
    using type = test_std_ranges::archetypes::search_common;
};

template <>
struct common_type<test_std_ranges::archetypes::nocopy_search_value_dc,
                   test_std_ranges::archetypes::removable_archetype_dc>
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

using searchable_dc_iterator_t = std::ranges::iterator_t<archetype_view<searchable_archetype_dc>>;
using removable_dc_iterator_t = std::ranges::iterator_t<archetype_view<removable_archetype_dc>>;

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(searchable_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(removable_archetype_dc)
static_assert(std::indirect_binary_predicate<std::ranges::equal_to, searchable_dc_iterator_t, const search_value*>);
static_assert(
    std::indirect_binary_predicate<std::ranges::equal_to, searchable_dc_iterator_t, const nocopy_search_value*>);
static_assert(
    std::indirect_binary_predicate<std::ranges::equal_to, searchable_dc_iterator_t, const nocopy_search_value_dc*>);
static_assert(std::permutable<removable_dc_iterator_t>);
static_assert(std::indirect_binary_predicate<std::ranges::equal_to, removable_dc_iterator_t, const search_value*>);
static_assert(
    std::indirect_binary_predicate<std::ranges::equal_to, removable_dc_iterator_t, const nocopy_search_value_dc*>);
static_assert(!std::default_initializable<searchable_archetype_dc>);
static_assert(!std::default_initializable<removable_archetype_dc>);
static_assert(!std::totally_ordered<searchable_archetype_dc>);
static_assert(!std::totally_ordered<removable_archetype_dc>);

struct search_proj_mut
{
    searchable_archetype& operator()(searchable_archetype& __v) const { return __v; }
    searchable_archetype_dc& operator()(searchable_archetype_dc& __v) const { return __v; }
    removable_archetype& operator()(removable_archetype& __v) const { return __v; }
    removable_archetype_dc& operator()(removable_archetype_dc& __v) const { return __v; }
};

static_assert(std::indirect_binary_predicate<std::ranges::equal_to,
                                             std::projected<searchable_iterator_t, search_proj_mut>,
                                             const search_value*>);
static_assert(std::indirect_binary_predicate<std::ranges::equal_to,
                                             std::projected<removable_iterator_t, search_proj_mut>,
                                             const search_value*>);
static_assert(!std::invocable<const search_proj_mut&, const searchable_archetype&>);

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_VALUE_H
