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

// Family 12: the sorting algorithms called without a comparator, i.e. with the default
// std::ranges::less. std::sortable<_It, _Comp, _Proj> then requires the element type itself to be
// std::totally_ordered on top of std::permutable, so this archetype is permutable_archetype plus
// exactly the two comparison operators that concept asks for; the comparisons are const members,
// because that is how std::equality_comparable and partially-ordered-with are spelled.
// Used by: sort, stable_sort, partial_sort, nth_element, inplace_merge.
struct permutable_ordered_archetype
{
    int val;

    explicit permutable_ordered_archetype(int __v) : val(__v) {}

    permutable_ordered_archetype(permutable_ordered_archetype&& __other) : val(__other.val) {}

    permutable_ordered_archetype& operator=(permutable_ordered_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    bool operator==(const permutable_ordered_archetype& __other) const { return val == __other.val; }

    std::strong_ordering operator<=>(const permutable_ordered_archetype& __other) const
    {
        return val <=> __other.val;
    }

    permutable_ordered_archetype(const permutable_ordered_archetype&) = delete;
    permutable_ordered_archetype& operator=(const permutable_ordered_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF
};

// The device copyable counterpart of the archetype above, used with the hetero policies.
struct permutable_ordered_archetype_dc
{
    int val;

    explicit permutable_ordered_archetype_dc(int __v) : val(__v) {}

    bool operator==(const permutable_ordered_archetype_dc& __other) const { return val == __other.val; }

    std::strong_ordering operator<=>(const permutable_ordered_archetype_dc& __other) const
    {
        return val <=> __other.val;
    }

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(permutable_ordered_archetype_dc)
};

// Family 13: unique called without its equivalence relation, i.e. with the default
// std::ranges::equal_to, which asks the element type for std::equality_comparable and for no ordering
// at all.
struct permutable_equality_archetype
{
    int val;

    explicit permutable_equality_archetype(int __v) : val(__v) {}

    permutable_equality_archetype(permutable_equality_archetype&& __other) : val(__other.val) {}

    permutable_equality_archetype& operator=(permutable_equality_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    bool operator==(const permutable_equality_archetype& __other) const { return val == __other.val; }

    permutable_equality_archetype(const permutable_equality_archetype&) = delete;
    permutable_equality_archetype& operator=(const permutable_equality_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF
};

// The device copyable counterpart of the archetype above, used with the hetero policies.
struct permutable_equality_archetype_dc
{
    int val;

    explicit permutable_equality_archetype_dc(int __v) : val(__v) {}

    bool operator==(const permutable_equality_archetype_dc& __other) const { return val == __other.val; }

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(permutable_equality_archetype_dc)
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(permutable_ordered_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(permutable_equality_archetype_dc)

using permutable_ordered_iterator_t = std::ranges::iterator_t<archetype_view<permutable_ordered_archetype>>;
using permutable_ordered_dc_iterator_t = std::ranges::iterator_t<archetype_view<permutable_ordered_archetype_dc>>;
using permutable_equality_iterator_t = std::ranges::iterator_t<archetype_view<permutable_equality_archetype>>;
using permutable_equality_dc_iterator_t = std::ranges::iterator_t<archetype_view<permutable_equality_archetype_dc>>;

// The default comparator of the sorting algorithms is accepted, and the element is still neither
// copyable nor default constructible.
static_assert(std::sortable<permutable_ordered_iterator_t, std::ranges::less>);
static_assert(std::sortable<permutable_ordered_dc_iterator_t, std::ranges::less>);
static_assert(!std::copy_constructible<permutable_ordered_archetype>);
static_assert(!std::default_initializable<permutable_ordered_archetype>);
static_assert(!std::default_initializable<permutable_ordered_archetype_dc>);

// unique only ever needs the equality, so the ordering is deliberately missing here.
static_assert(std::permutable<permutable_equality_iterator_t>);
static_assert(std::permutable<permutable_equality_dc_iterator_t>);
static_assert(std::equality_comparable<permutable_equality_archetype>);
static_assert(!std::totally_ordered<permutable_equality_archetype>);
static_assert(!std::totally_ordered<permutable_equality_archetype_dc>);
static_assert(!std::sortable<permutable_equality_iterator_t, std::ranges::less>);
static_assert(!std::copy_constructible<permutable_equality_archetype>);
static_assert(!std::default_initializable<permutable_equality_archetype>);

// Family 14: partial_sort_copy, which is the sorting family and a copying algorithm at once. Beyond
// std::sortable<iterator_t<_OutR>, _Comp, _Proj2> for the output range it requires
//   std::indirectly_copyable<iterator_t<_R>, iterator_t<_OutR>> &&
//   std::indirect_strict_weak_order<_Comp, projected<iterator_t<_R>, _Proj1>,
//                                          projected<iterator_t<_OutR>, _Proj2>>
// so the input element is only assignable into the output one and comparable with it, which leaves it
// neither copyable, movable nor default constructible; the output element is the permutable one of
// family 9 plus that assignment. The comparator has to accept all four combinations of the two element
// types, because std::strict_weak_order is symmetric in its two argument types.
struct psort_copy_out_archetype;

struct psort_copy_in_archetype
{
    int val;

    // The output element type the algorithm has to be called with, see merge_in_archetype::out_type.
    using out_type = psort_copy_out_archetype;

    explicit psort_copy_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(psort_copy_in_archetype)
};

struct psort_copy_out_archetype
{
    int val;

    explicit psort_copy_out_archetype(int __v) : val(__v) {}

    psort_copy_out_archetype(psort_copy_out_archetype&& __other) : val(__other.val) {}

    psort_copy_out_archetype& operator=(psort_copy_out_archetype&& __other)
    {
        val = __other.val;
        return *this;
    }

    psort_copy_out_archetype(const psort_copy_out_archetype&) = delete;
    psort_copy_out_archetype& operator=(const psort_copy_out_archetype&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    psort_copy_out_archetype& operator=(psort_copy_in_archetype& __v)
    {
        val = __v.val;
        return *this;
    }
};

// The device copyable counterparts of the two archetypes above, used with the hetero policies.
struct psort_copy_out_archetype_dc;

struct psort_copy_in_archetype_dc
{
    int val;

    // The matching output element type, see merge_in_archetype_dc::out_type.
    using out_type = psort_copy_out_archetype_dc;

    explicit psort_copy_in_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(psort_copy_in_archetype_dc)
};

struct psort_copy_out_archetype_dc
{
    int val;

    explicit psort_copy_out_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(psort_copy_out_archetype_dc)

    psort_copy_out_archetype_dc& operator=(psort_copy_in_archetype_dc& __v)
    {
        val = __v.val;
        return *this;
    }
};

// The comparator orders descending, so that the values partial_sort_copy writes into the output range
// differ from the zeros it starts out with as well as from the ascending input.
struct psort_copy_comp
{
    bool operator()(const psort_copy_in_archetype& __v1, const psort_copy_in_archetype& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(const psort_copy_in_archetype& __v1, const psort_copy_out_archetype& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(const psort_copy_out_archetype& __v1, const psort_copy_in_archetype& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(const psort_copy_out_archetype& __v1, const psort_copy_out_archetype& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(const psort_copy_in_archetype_dc& __v1, const psort_copy_in_archetype_dc& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(const psort_copy_in_archetype_dc& __v1, const psort_copy_out_archetype_dc& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(const psort_copy_out_archetype_dc& __v1, const psort_copy_in_archetype_dc& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(const psort_copy_out_archetype_dc& __v1, const psort_copy_out_archetype_dc& __v2) const
    {
        return __v1.val > __v2.val;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(psort_copy_in_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(psort_copy_out_archetype_dc)

using psort_copy_in_iterator_t = std::ranges::iterator_t<archetype_view<psort_copy_in_archetype>>;
using psort_copy_out_iterator_t = std::ranges::iterator_t<archetype_view<psort_copy_out_archetype>>;
using psort_copy_in_dc_iterator_t = std::ranges::iterator_t<archetype_view<psort_copy_in_archetype_dc>>;
using psort_copy_out_dc_iterator_t = std::ranges::iterator_t<archetype_view<psort_copy_out_archetype_dc>>;

static_assert(std::indirectly_copyable<psort_copy_in_iterator_t, psort_copy_out_iterator_t>);
static_assert(std::indirectly_copyable<psort_copy_in_dc_iterator_t, psort_copy_out_dc_iterator_t>);
static_assert(std::sortable<psort_copy_out_iterator_t, psort_copy_comp>);
static_assert(std::sortable<psort_copy_out_dc_iterator_t, psort_copy_comp>);
static_assert(std::indirect_strict_weak_order<psort_copy_comp, psort_copy_in_iterator_t, psort_copy_out_iterator_t>);
static_assert(std::indirect_strict_weak_order<psort_copy_comp, psort_copy_in_dc_iterator_t,
                                              psort_copy_out_dc_iterator_t>);
// The input element is not even movable, and neither element type is ordered by itself.
static_assert(!std::movable<psort_copy_in_archetype>);
static_assert(!std::copy_constructible<psort_copy_in_archetype>);
static_assert(!std::default_initializable<psort_copy_in_archetype>);
static_assert(!std::default_initializable<psort_copy_in_archetype_dc>);
static_assert(!std::copy_constructible<psort_copy_out_archetype>);
static_assert(!std::default_initializable<psort_copy_out_archetype>);
static_assert(!std::default_initializable<psort_copy_out_archetype_dc>);
static_assert(!std::totally_ordered<psort_copy_in_archetype>);
static_assert(!std::totally_ordered<psort_copy_out_archetype>);

// Family 15: the sorting algorithms with a projection mapping the element to an integer key. The
// element stays the permutable archetype of family 9, because std::sortable asks it for nothing but
// moving and swapping once the ordering comes from std::ranges::less on the projected key; the key is
// a prvalue, so the element is not even required to be comparable.
// This is also the only way to reach the radix sort of the device backend at all:
// __is_radix_sort_usable_for_type (parallel_backend_sycl.h:1434) is instantiated with
// __key_t<_Proj, _Range> (utils_ranges.h:204), i.e. with the return type of the projection and never
// with the element type, which no archetype can make integral.
struct permutable_proj_key
{
    int operator()(const permutable_archetype& __v) const { return __v.val; }
    int operator()(const permutable_archetype_dc& __v) const { return __v.val; }
};

static_assert(std::sortable<permutable_iterator_t, std::ranges::less, permutable_proj_key>);
static_assert(std::sortable<permutable_dc_iterator_t, std::ranges::less, permutable_proj_key>);
static_assert(std::same_as<int, std::invoke_result_t<permutable_proj_key&, permutable_archetype&>>);

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

// Family 14: the comparator and the projections of partial_sort_copy taking the elements by non-const
// reference. Both element types are mutable through their ranges, so nothing in the requires-clause
// asks for a const lvalue anywhere.
struct psort_copy_comp_mut
{
    bool operator()(psort_copy_in_archetype& __v1, psort_copy_in_archetype& __v2) const { return __v1.val > __v2.val; }
    bool operator()(psort_copy_in_archetype& __v1, psort_copy_out_archetype& __v2) const { return __v1.val > __v2.val; }
    bool operator()(psort_copy_out_archetype& __v1, psort_copy_in_archetype& __v2) const { return __v1.val > __v2.val; }
    bool operator()(psort_copy_out_archetype& __v1, psort_copy_out_archetype& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(psort_copy_in_archetype_dc& __v1, psort_copy_in_archetype_dc& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(psort_copy_in_archetype_dc& __v1, psort_copy_out_archetype_dc& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(psort_copy_out_archetype_dc& __v1, psort_copy_in_archetype_dc& __v2) const
    {
        return __v1.val > __v2.val;
    }
    bool operator()(psort_copy_out_archetype_dc& __v1, psort_copy_out_archetype_dc& __v2) const
    {
        return __v1.val > __v2.val;
    }
};

// The two projections are separate types, because partial_sort_copy projects the input range and the
// output range with two distinct callables, see search_proj_mut for the same construct in family 2.
struct psort_copy_in_proj_mut
{
    psort_copy_in_archetype& operator()(psort_copy_in_archetype& __v) const { return __v; }
    psort_copy_in_archetype_dc& operator()(psort_copy_in_archetype_dc& __v) const { return __v; }
};

struct psort_copy_out_proj_mut
{
    psort_copy_out_archetype& operator()(psort_copy_out_archetype& __v) const { return __v; }
    psort_copy_out_archetype_dc& operator()(psort_copy_out_archetype_dc& __v) const { return __v; }
};

static_assert(std::sortable<psort_copy_out_iterator_t, psort_copy_comp_mut, psort_copy_out_proj_mut>);
static_assert(std::sortable<psort_copy_out_dc_iterator_t, psort_copy_comp_mut, psort_copy_out_proj_mut>);
static_assert(!std::invocable<const psort_copy_comp_mut&, const psort_copy_in_archetype&,
                              const psort_copy_out_archetype&>);

static_assert(std::indirect_unary_predicate<permutable_pred_mut, permutable_iterator_t>);
static_assert(std::indirect_binary_predicate<permutable_equiv_mut, permutable_iterator_t, permutable_iterator_t>);
static_assert(std::sortable<permutable_iterator_t, permutable_comp_mut>);
static_assert(std::sortable<permutable_dc_iterator_t, permutable_comp_mut>);
static_assert(!std::invocable<const permutable_comp_mut&, const permutable_archetype&, const permutable_archetype&>);

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_PERMUTE_H
