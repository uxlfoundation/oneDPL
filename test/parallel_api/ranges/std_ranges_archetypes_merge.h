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

#ifndef _STD_RANGES_ARCHETYPES_MERGE_H
#define _STD_RANGES_ARCHETYPES_MERGE_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"

namespace test_std_ranges
{
namespace archetypes
{

// The merge family additionally needs std::indirectly_copyable from both inputs into the output.
// The output element is therefore assignable from a non-const lvalue of either input element type,
// while remaining non-copyable itself.
// Used by: merge, set_union, set_intersection, set_difference, set_symmetric_difference.
struct merge_out_archetype;

struct merge_in_archetype
{
    int val;

    // The output element type the algorithm has to be called with, so that a generic test body may
    // pick the right one for the input element type it works on.
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

// The device copyable counterparts of the two archetypes above, used with the hetero policies.
struct merge_out_archetype_dc;

struct merge_in_archetype_dc
{
    int val;

    // The matching output element type, see merge_in_archetype::out_type: one and the same generic test
    // body serves the host and the hetero policies, so it derives the output type from the input one.
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

// The merge family and min / max / minmax, whose comparators are constrained the very same way.
struct merge_comp_mut
{
    bool operator()(merge_in_archetype& __v1, merge_in_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(merge_in_archetype_dc& __v1, merge_in_archetype_dc& __v2) const { return __v1.val < __v2.val; }
};

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_MERGE_H
