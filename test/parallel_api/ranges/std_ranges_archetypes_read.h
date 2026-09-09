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

#ifndef _STD_RANGES_ARCHETYPES_READ_H
#define _STD_RANGES_ARCHETYPES_READ_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"

namespace test_std_ranges
{
namespace archetypes
{

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

// The device copyable counterpart of read_archetype, used with the hetero policies: it is trivially
// copyable, so a device kernel may take it by value, but it is still not default constructible and
// not comparable.
struct read_archetype_dc
{
    int val;

    explicit read_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(read_archetype_dc)
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(read_archetype_dc)
static_assert(!std::default_initializable<read_archetype_dc>);
static_assert(!std::equality_comparable<read_archetype_dc>);
static_assert(!std::totally_ordered<read_archetype_dc>);

// The callables take exactly const _T& and return exactly the required type, so an implementation
// cannot pass an rvalue, a copy, or expect a wider return type.
struct read_unary_fun
{
    void operator()(const read_archetype&) const {}
    void operator()(const read_archetype_dc&) const {}
};

struct read_unary_pred
{
    bool operator()(const read_archetype& __v) const { return __v.val % 3 == 0; }
    bool operator()(const read_archetype_dc& __v) const { return __v.val % 3 == 0; }
};

struct read_binary_pred
{
    bool operator()(const read_archetype& __v1, const read_archetype& __v2) const { return __v1.val == __v2.val; }
    bool operator()(const read_archetype_dc& __v1, const read_archetype_dc& __v2) const
    {
        return __v1.val == __v2.val;
    }
};

struct read_comp
{
    bool operator()(const read_archetype& __v1, const read_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(const read_archetype_dc& __v1, const read_archetype_dc& __v2) const { return __v1.val < __v2.val; }
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
    read_proj_result operator()(const read_archetype_dc& __v) const { return read_proj_result{__v.val}; }
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

// The device copyable counterparts of the two archetypes above, used with the hetero policies.
struct lhs_archetype_dc
{
    int val;

    explicit lhs_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(lhs_archetype_dc)
};

struct rhs_archetype_dc
{
    int val;

    explicit rhs_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(rhs_archetype_dc)
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(lhs_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(rhs_archetype_dc)
static_assert(!std::equality_comparable<lhs_archetype_dc>);
static_assert(!std::equality_comparable<rhs_archetype_dc>);

struct cross_pred
{
    bool operator()(const lhs_archetype& __v1, const rhs_archetype& __v2) const { return __v1.val == __v2.val; }
    bool operator()(const lhs_archetype_dc& __v1, const rhs_archetype_dc& __v2) const { return __v1.val == __v2.val; }
};

// includes is constrained by std::indirect_strict_weak_order over the two projected iterators, which
// subsumes std::relation and therefore asks for the two element types in all four combinations, not
// only for (lhs, rhs) the way std::indirectly_comparable does for cross_pred above.
struct cross_comp
{
    bool operator()(const lhs_archetype& __v1, const lhs_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(const lhs_archetype& __v1, const rhs_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(const rhs_archetype& __v1, const lhs_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(const rhs_archetype& __v1, const rhs_archetype& __v2) const { return __v1.val < __v2.val; }

    bool operator()(const lhs_archetype_dc& __v1, const lhs_archetype_dc& __v2) const { return __v1.val < __v2.val; }
    bool operator()(const lhs_archetype_dc& __v1, const rhs_archetype_dc& __v2) const { return __v1.val < __v2.val; }
    bool operator()(const rhs_archetype_dc& __v1, const lhs_archetype_dc& __v2) const { return __v1.val < __v2.val; }
    bool operator()(const rhs_archetype_dc& __v1, const rhs_archetype_dc& __v2) const { return __v1.val < __v2.val; }
};

using lhs_iterator_t = std::ranges::iterator_t<archetype_view<lhs_archetype>>;
using rhs_iterator_t = std::ranges::iterator_t<archetype_view<rhs_archetype>>;

static_assert(std::indirectly_comparable<lhs_iterator_t, rhs_iterator_t, cross_pred>);
static_assert(std::indirect_strict_weak_order<cross_comp, lhs_iterator_t, rhs_iterator_t>);
static_assert(std::indirect_strict_weak_order<cross_comp, std::ranges::iterator_t<archetype_view<lhs_archetype_dc>>,
                                              std::ranges::iterator_t<archetype_view<rhs_archetype_dc>>>);
static_assert(std::indirectly_comparable<std::ranges::iterator_t<archetype_view<lhs_archetype_dc>>,
                                        std::ranges::iterator_t<archetype_view<rhs_archetype_dc>>, cross_pred>);
static_assert(!std::equality_comparable<lhs_archetype>);
static_assert(!std::equality_comparable<rhs_archetype>);
static_assert(!std::copy_constructible<lhs_archetype>);
static_assert(!std::copy_constructible<rhs_archetype>);

// Family 1: read-only algorithms parameterized by a callable.
struct read_unary_fun_mut
{
    void operator()(read_archetype&) const {}
    void operator()(read_archetype_dc&) const {}
};

struct read_unary_pred_mut
{
    bool operator()(read_archetype& __v) const { return __v.val % 3 == 0; }
    bool operator()(read_archetype_dc& __v) const { return __v.val % 3 == 0; }
};

struct read_binary_pred_mut
{
    bool operator()(read_archetype& __v1, read_archetype& __v2) const { return __v1.val == __v2.val; }
    bool operator()(read_archetype_dc& __v1, read_archetype_dc& __v2) const { return __v1.val == __v2.val; }
};

struct read_comp_mut
{
    bool operator()(read_archetype& __v1, read_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(read_archetype_dc& __v1, read_archetype_dc& __v2) const { return __v1.val < __v2.val; }
};

// A projection taking the element by non-const reference. Its result is a prvalue of an unrelated
// type, which the pre-existing read_proj_pred consumes: a predicate over a projection cannot take a
// non-const reference itself, because indirect_unary_predicate also requires it to be invocable with
// iter_reference_t of the projected iterator, which is that prvalue.
struct read_proj_mut
{
    read_proj_result operator()(read_archetype& __v) const { return read_proj_result{__v.val}; }
    read_proj_result operator()(read_archetype_dc& __v) const { return read_proj_result{__v.val}; }
};

static_assert(std::indirectly_unary_invocable<read_unary_fun_mut, read_iterator_t>);
static_assert(std::indirect_unary_predicate<read_unary_pred_mut, read_iterator_t>);
static_assert(std::indirect_binary_predicate<read_binary_pred_mut, read_iterator_t, read_iterator_t>);
static_assert(std::indirect_strict_weak_order<read_comp_mut, read_iterator_t>);
static_assert(std::indirect_unary_predicate<read_proj_pred, std::projected<read_iterator_t, read_proj_mut>>);
// The callables really do reject anything but a non-const lvalue of the element type.
static_assert(!std::invocable<const read_unary_pred_mut&, const read_archetype&>);
static_assert(!std::invocable<const read_unary_pred_mut&, read_archetype&&>);
static_assert(!std::invocable<const read_comp_mut&, const read_archetype&, const read_archetype&>);
static_assert(!std::invocable<const read_proj_mut&, const read_archetype&>);

// Family 3: two-range algorithms constrained by std::indirectly_comparable. Both references are
// non-const lvalues, so the predicate may take both of its arguments that way.
struct cross_pred_mut
{
    bool operator()(lhs_archetype& __v1, rhs_archetype& __v2) const { return __v1.val == __v2.val; }
    bool operator()(lhs_archetype_dc& __v1, rhs_archetype_dc& __v2) const { return __v1.val == __v2.val; }
};

static_assert(std::indirectly_comparable<lhs_iterator_t, rhs_iterator_t, cross_pred_mut>);
static_assert(!std::invocable<const cross_pred_mut&, const lhs_archetype&, const rhs_archetype&>);

// The four-combination comparator of includes, see cross_comp: every reference it is handed by
// std::indirect_strict_weak_order is a non-const lvalue as well.
struct cross_comp_mut
{
    bool operator()(lhs_archetype& __v1, lhs_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(lhs_archetype& __v1, rhs_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(rhs_archetype& __v1, lhs_archetype& __v2) const { return __v1.val < __v2.val; }
    bool operator()(rhs_archetype& __v1, rhs_archetype& __v2) const { return __v1.val < __v2.val; }

    bool operator()(lhs_archetype_dc& __v1, lhs_archetype_dc& __v2) const { return __v1.val < __v2.val; }
    bool operator()(lhs_archetype_dc& __v1, rhs_archetype_dc& __v2) const { return __v1.val < __v2.val; }
    bool operator()(rhs_archetype_dc& __v1, lhs_archetype_dc& __v2) const { return __v1.val < __v2.val; }
    bool operator()(rhs_archetype_dc& __v1, rhs_archetype_dc& __v2) const { return __v1.val < __v2.val; }
};

static_assert(std::indirect_strict_weak_order<cross_comp_mut, lhs_iterator_t, rhs_iterator_t>);
static_assert(!std::invocable<const cross_comp_mut&, const lhs_archetype&, const rhs_archetype&>);

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_READ_H
