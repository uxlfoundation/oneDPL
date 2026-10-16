// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

static_assert(std::sortable<permutable_ordered_iterator_t, std::ranges::less>);
static_assert(std::sortable<permutable_ordered_dc_iterator_t, std::ranges::less>);
static_assert(!std::copy_constructible<permutable_ordered_archetype>);
static_assert(!std::default_initializable<permutable_ordered_archetype>);
static_assert(!std::default_initializable<permutable_ordered_archetype_dc>);

static_assert(std::permutable<permutable_equality_iterator_t>);
static_assert(std::permutable<permutable_equality_dc_iterator_t>);
static_assert(std::equality_comparable<permutable_equality_archetype>);
static_assert(!std::totally_ordered<permutable_equality_archetype>);
static_assert(!std::totally_ordered<permutable_equality_archetype_dc>);
static_assert(!std::sortable<permutable_equality_iterator_t, std::ranges::less>);
static_assert(!std::copy_constructible<permutable_equality_archetype>);
static_assert(!std::default_initializable<permutable_equality_archetype>);

struct psort_copy_out_archetype;

struct psort_copy_in_archetype
{
    int val;

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

struct psort_copy_out_archetype_dc;

struct psort_copy_in_archetype_dc
{
    int val;

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
static_assert(!std::movable<psort_copy_in_archetype>);
static_assert(!std::copy_constructible<psort_copy_in_archetype>);
static_assert(!std::default_initializable<psort_copy_in_archetype>);
static_assert(!std::default_initializable<psort_copy_in_archetype_dc>);
static_assert(!std::copy_constructible<psort_copy_out_archetype>);
static_assert(!std::default_initializable<psort_copy_out_archetype>);
static_assert(!std::default_initializable<psort_copy_out_archetype_dc>);
static_assert(!std::totally_ordered<psort_copy_in_archetype>);
static_assert(!std::totally_ordered<psort_copy_out_archetype>);

struct permutable_proj_key
{
    int operator()(const permutable_archetype& __v) const { return __v.val; }
    int operator()(const permutable_archetype_dc& __v) const { return __v.val; }
};

static_assert(std::sortable<permutable_iterator_t, std::ranges::less, permutable_proj_key>);
static_assert(std::sortable<permutable_dc_iterator_t, std::ranges::less, permutable_proj_key>);
static_assert(std::same_as<int, std::invoke_result_t<permutable_proj_key&, permutable_archetype&>>);

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
