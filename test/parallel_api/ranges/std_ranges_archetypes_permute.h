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

#ifndef _STD_RANGES_ARCHETYPES_PERMUTE_H
#define _STD_RANGES_ARCHETYPES_PERMUTE_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"

namespace test_std_ranges
{
namespace archetypes
{

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

// The device copyable counterpart of the archetype above, used with the hetero policies.
struct permutable_archetype_dc
{
    int val;

    explicit permutable_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(permutable_archetype_dc)
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(permutable_archetype_dc)
static_assert(!std::default_initializable<permutable_archetype_dc>);
static_assert(!std::equality_comparable<permutable_archetype_dc>);
static_assert(!std::totally_ordered<permutable_archetype_dc>);

using permutable_iterator_t = std::ranges::iterator_t<archetype_view<permutable_archetype>>;
using permutable_dc_iterator_t = std::ranges::iterator_t<archetype_view<permutable_archetype_dc>>;

static_assert(std::permutable<permutable_iterator_t>);
static_assert(!std::copy_constructible<permutable_archetype>);
static_assert(!std::default_initializable<permutable_archetype>);
static_assert(!std::equality_comparable<permutable_archetype>);
static_assert(!std::totally_ordered<permutable_archetype>);

// The predicate and the comparator of the permuting algorithms only see the projected reference.
struct permutable_pred
{
    bool operator()(const permutable_archetype& __v) const { return __v.val % 3 == 0; }
    bool operator()(const permutable_archetype_dc& __v) const { return __v.val % 3 == 0; }
};

struct permutable_equiv
{
    bool operator()(const permutable_archetype& __v1, const permutable_archetype& __v2) const
    {
        return __v1.val == __v2.val;
    }
    bool operator()(const permutable_archetype_dc& __v1, const permutable_archetype_dc& __v2) const
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
    bool operator()(const permutable_archetype_dc& __v1, const permutable_archetype_dc& __v2) const
    {
        return __v1.val < __v2.val;
    }
};

static_assert(std::sortable<permutable_iterator_t, permutable_comp>);
static_assert(std::permutable<permutable_dc_iterator_t>);
static_assert(std::sortable<permutable_dc_iterator_t, permutable_comp>);

// Family 9: permuting and sorting algorithms. The element is mutable by definition here, so the
// predicate and the comparator may take it by non-const reference as well.
struct permutable_pred_mut
{
    bool operator()(permutable_archetype& __v) const { return __v.val % 3 == 0; }
    bool operator()(permutable_archetype_dc& __v) const { return __v.val % 3 == 0; }
};

struct permutable_equiv_mut
{
    bool operator()(permutable_archetype& __v1, permutable_archetype& __v2) const { return __v1.val == __v2.val; }
    bool operator()(permutable_archetype_dc& __v1, permutable_archetype_dc& __v2) const
    {
        return __v1.val == __v2.val;
    }
};

struct permutable_comp_mut
{
    bool operator()(permutable_archetype& __v1, permutable_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(permutable_archetype_dc& __v1, permutable_archetype_dc& __v2) const { return __v1.val < __v2.val; }
};

static_assert(std::indirect_unary_predicate<permutable_pred_mut, permutable_iterator_t>);
static_assert(std::indirect_binary_predicate<permutable_equiv_mut, permutable_iterator_t, permutable_iterator_t>);
static_assert(std::sortable<permutable_iterator_t, permutable_comp_mut>);
static_assert(std::sortable<permutable_dc_iterator_t, permutable_comp_mut>);
static_assert(!std::invocable<const permutable_comp_mut&, const permutable_archetype&, const permutable_archetype&>);

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_PERMUTE_H
