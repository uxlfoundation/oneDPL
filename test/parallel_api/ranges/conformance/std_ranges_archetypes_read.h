// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

struct ordered_archetype
{
    int val;

    explicit ordered_archetype(int __v) : val(__v) {}

    bool operator==(const ordered_archetype& __other) const { return val == __other.val; }
    std::strong_ordering operator<=>(const ordered_archetype& __other) const { return val <=> __other.val; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(ordered_archetype)
};

struct ordered_archetype_dc
{
    int val;

    explicit ordered_archetype_dc(int __v) : val(__v) {}

    bool operator==(const ordered_archetype_dc& __other) const { return val == __other.val; }
    std::strong_ordering operator<=>(const ordered_archetype_dc& __other) const { return val <=> __other.val; }

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(ordered_archetype_dc)
};

struct equality_archetype
{
    int val;

    explicit equality_archetype(int __v) : val(__v) {}

    bool operator==(const equality_archetype& __other) const { return val == __other.val; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(equality_archetype)
};

struct equality_archetype_dc
{
    int val;

    explicit equality_archetype_dc(int __v) : val(__v) {}

    bool operator==(const equality_archetype_dc& __other) const { return val == __other.val; }

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(equality_archetype_dc)
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(ordered_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(equality_archetype_dc)

using ordered_iterator_t = std::ranges::iterator_t<archetype_view<ordered_archetype>>;
using ordered_dc_iterator_t = std::ranges::iterator_t<archetype_view<ordered_archetype_dc>>;
using equality_iterator_t = std::ranges::iterator_t<archetype_view<equality_archetype>>;
using equality_dc_iterator_t = std::ranges::iterator_t<archetype_view<equality_archetype_dc>>;

static_assert(std::totally_ordered<ordered_archetype>);
static_assert(std::totally_ordered<ordered_archetype_dc>);
static_assert(std::indirect_strict_weak_order<std::ranges::less, ordered_iterator_t>);
static_assert(std::indirect_strict_weak_order<std::ranges::less, ordered_dc_iterator_t>);
static_assert(!std::copy_constructible<ordered_archetype>);
static_assert(!std::move_constructible<ordered_archetype>);
static_assert(!std::default_initializable<ordered_archetype>);
static_assert(!std::default_initializable<ordered_archetype_dc>);

static_assert(std::equality_comparable<equality_archetype>);
static_assert(std::equality_comparable<equality_archetype_dc>);
static_assert(!std::totally_ordered<equality_archetype>);
static_assert(!std::totally_ordered<equality_archetype_dc>);
static_assert(std::indirectly_comparable<equality_iterator_t, equality_iterator_t, std::ranges::equal_to>);
static_assert(std::indirectly_comparable<equality_dc_iterator_t, equality_dc_iterator_t, std::ranges::equal_to>);
static_assert(!std::indirect_strict_weak_order<std::ranges::less, equality_iterator_t>);
static_assert(!std::copy_constructible<equality_archetype>);
static_assert(!std::default_initializable<equality_archetype>);
static_assert(!std::default_initializable<equality_archetype_dc>);

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
static_assert(!std::invocable<const read_unary_pred_mut&, const read_archetype&>);
static_assert(!std::invocable<const read_unary_pred_mut&, read_archetype&&>);
static_assert(!std::invocable<const read_comp_mut&, const read_archetype&, const read_archetype&>);
static_assert(!std::invocable<const read_proj_mut&, const read_archetype&>);

struct cross_pred_mut
{
    bool operator()(lhs_archetype& __v1, rhs_archetype& __v2) const { return __v1.val == __v2.val; }
    bool operator()(lhs_archetype_dc& __v1, rhs_archetype_dc& __v2) const { return __v1.val == __v2.val; }
};

static_assert(std::indirectly_comparable<lhs_iterator_t, rhs_iterator_t, cross_pred_mut>);
static_assert(!std::invocable<const cross_pred_mut&, const lhs_archetype&, const rhs_archetype&>);

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
