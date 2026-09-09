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

#ifndef _STD_RANGES_ARCHETYPES_STORABLE_H
#define _STD_RANGES_ARCHETYPES_STORABLE_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"
// storable_comp_mut is constrained over merge_in_iterator_t as well, see the assert below.
#include "std_ranges_archetypes_merge.h"

namespace test_std_ranges
{
namespace archetypes
{

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

// The device copyable counterpart of the archetype above, used with the hetero policies.
struct storable_archetype_dc
{
    int val;

    explicit storable_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(storable_archetype_dc)
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(storable_archetype_dc)
static_assert(!std::default_initializable<storable_archetype_dc>);
static_assert(!std::equality_comparable<storable_archetype_dc>);
static_assert(!std::totally_ordered<storable_archetype_dc>);

struct storable_comp
{
    bool operator()(const storable_archetype& __v1, const storable_archetype& __v2) const
    {
        return __v1.val < __v2.val;
    }
    bool operator()(const storable_archetype_dc& __v1, const storable_archetype_dc& __v2) const
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

static_assert(std::indirectly_copyable_storable<std::ranges::iterator_t<archetype_view<storable_archetype_dc>>,
                                                storable_archetype_dc*>);

struct storable_comp_mut
{
    bool operator()(storable_archetype& __v1, storable_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(storable_archetype_dc& __v1, storable_archetype_dc& __v2) const { return __v1.val < __v2.val; }
};

static_assert(std::mergeable<merge_in_iterator_t, merge_in_iterator_t, merge_out_iterator_t, merge_comp_mut>);
static_assert(std::indirect_strict_weak_order<storable_comp_mut, storable_iterator_t>);
static_assert(!std::invocable<const merge_comp_mut&, const merge_in_archetype&, const merge_in_archetype&>);
static_assert(!std::invocable<const storable_comp_mut&, const storable_archetype&, const storable_archetype&>);

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_STORABLE_H
