// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _STD_RANGES_ARCHETYPES_MERGE_H
#define _STD_RANGES_ARCHETYPES_MERGE_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"

namespace test_std_ranges
{
namespace archetypes
{

struct merge_out_archetype;

struct merge_in_archetype
{
    int val;

    using out_type = merge_out_archetype;

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

struct merge_out_archetype_dc;

struct merge_in_archetype_dc
{
    int val;

    using out_type = merge_out_archetype_dc;

    explicit merge_in_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(merge_in_archetype_dc)
};

struct merge_out_archetype_dc
{
    int val;

    explicit merge_out_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(merge_out_archetype_dc)

    merge_out_archetype_dc& operator=(merge_in_archetype_dc& __v)
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
    bool operator()(const merge_in_archetype_dc& __v1, const merge_in_archetype_dc& __v2) const
    {
        return __v1.val < __v2.val;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(merge_in_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(merge_out_archetype_dc)
static_assert(std::mergeable<std::ranges::iterator_t<archetype_view<merge_in_archetype_dc>>,
                             std::ranges::iterator_t<archetype_view<merge_in_archetype_dc>>,
                             std::ranges::iterator_t<archetype_view<merge_out_archetype_dc>>, merge_comp>);
static_assert(!std::default_initializable<merge_out_archetype_dc>);

using merge_in_iterator_t = std::ranges::iterator_t<archetype_view<merge_in_archetype>>;
using merge_out_iterator_t = std::ranges::iterator_t<archetype_view<merge_out_archetype>>;

static_assert(std::mergeable<merge_in_iterator_t, merge_in_iterator_t, merge_out_iterator_t, merge_comp>);
static_assert(!std::copy_constructible<merge_in_archetype>);
static_assert(!std::copy_constructible<merge_out_archetype>);
static_assert(!std::default_initializable<merge_out_archetype>);

struct merge_ordered_out_archetype;

struct merge_ordered_in_archetype
{
    int val;

    using out_type = merge_ordered_out_archetype;

    explicit merge_ordered_in_archetype(int __v) : val(__v) {}

    merge_ordered_in_archetype(merge_ordered_in_archetype&& __other) : val(__other.val) {}

    merge_ordered_in_archetype& operator=(merge_ordered_in_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    bool operator==(const merge_ordered_in_archetype& __other) const { return val == __other.val; }

    std::strong_ordering operator<=>(const merge_ordered_in_archetype& __other) const
    {
        return val <=> __other.val;
    }

    merge_ordered_in_archetype(const merge_ordered_in_archetype&) = delete;
    merge_ordered_in_archetype& operator=(const merge_ordered_in_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF
};

struct merge_ordered_out_archetype
{
    int val;

    explicit merge_ordered_out_archetype(int __v) : val(__v) {}

    merge_ordered_out_archetype(merge_ordered_out_archetype&& __other) : val(__other.val) {}

    merge_ordered_out_archetype& operator=(merge_ordered_out_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    merge_ordered_out_archetype(const merge_ordered_out_archetype&) = delete;
    merge_ordered_out_archetype& operator=(const merge_ordered_out_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    merge_ordered_out_archetype& operator=(merge_ordered_in_archetype& __v)
    {
        val = __v.val;
        return *this;
    }
};

struct merge_ordered_out_archetype_dc;

struct merge_ordered_in_archetype_dc
{
    int val;

    using out_type = merge_ordered_out_archetype_dc;

    explicit merge_ordered_in_archetype_dc(int __v) : val(__v) {}

    bool operator==(const merge_ordered_in_archetype_dc& __other) const { return val == __other.val; }

    std::strong_ordering operator<=>(const merge_ordered_in_archetype_dc& __other) const
    {
        return val <=> __other.val;
    }

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(merge_ordered_in_archetype_dc)
};

struct merge_ordered_out_archetype_dc
{
    int val;

    explicit merge_ordered_out_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(merge_ordered_out_archetype_dc)

    merge_ordered_out_archetype_dc& operator=(merge_ordered_in_archetype_dc& __v)
    {
        val = __v.val;
        return *this;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(merge_ordered_in_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(merge_ordered_out_archetype_dc)

using merge_ordered_in_iterator_t = std::ranges::iterator_t<archetype_view<merge_ordered_in_archetype>>;
using merge_ordered_out_iterator_t = std::ranges::iterator_t<archetype_view<merge_ordered_out_archetype>>;
using merge_ordered_in_dc_iterator_t = std::ranges::iterator_t<archetype_view<merge_ordered_in_archetype_dc>>;
using merge_ordered_out_dc_iterator_t = std::ranges::iterator_t<archetype_view<merge_ordered_out_archetype_dc>>;

static_assert(std::mergeable<merge_ordered_in_iterator_t, merge_ordered_in_iterator_t, merge_ordered_out_iterator_t,
                             std::ranges::less>);
static_assert(std::mergeable<merge_ordered_in_dc_iterator_t, merge_ordered_in_dc_iterator_t,
                             merge_ordered_out_dc_iterator_t, std::ranges::less>);
static_assert(std::totally_ordered<merge_ordered_in_archetype>);
static_assert(!std::totally_ordered<merge_ordered_out_archetype>);
static_assert(!std::copy_constructible<merge_ordered_in_archetype>);
static_assert(!std::copy_constructible<merge_ordered_out_archetype>);
static_assert(!std::default_initializable<merge_ordered_out_archetype>);

struct merge_comp_mut
{
    bool operator()(merge_in_archetype& __v1, merge_in_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(merge_in_archetype_dc& __v1, merge_in_archetype_dc& __v2) const { return __v1.val < __v2.val; }
};

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_MERGE_H
