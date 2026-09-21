// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _STD_RANGES_ARCHETYPES_WRITE_H
#define _STD_RANGES_ARCHETYPES_WRITE_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"
#include "std_ranges_archetypes_value.h"

namespace test_std_ranges
{
namespace archetypes
{

struct write_value
{
    int val;

    explicit write_value(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(write_value)
};

struct write_value_dc
{
    int val;

    explicit write_value_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(write_value_dc)
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(write_value_dc)

struct writable_archetype
{
    int val;

    using value_arg = write_value;

    explicit writable_archetype(int __v) : val(__v) {}

    writable_archetype(const writable_archetype&) = delete;
    writable_archetype(writable_archetype&&) = delete;
    writable_archetype& operator=(const writable_archetype&) = delete;
    writable_archetype& operator=(writable_archetype&&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    writable_archetype& operator=(const write_value& __v)
    {
        val = __v.val;
        return *this;
    }
};

struct writable_archetype_dc
{
    int val;

    using value_arg = write_value_dc;

    explicit writable_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(writable_archetype_dc)

    writable_archetype_dc& operator=(const write_value_dc& __v)
    {
        val = __v.val;
        return *this;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(writable_archetype_dc)

struct write_pred
{
    bool operator()(const writable_archetype& __v) const { return __v.val % 3 == 0; }
    bool operator()(const writable_archetype_dc& __v) const { return __v.val % 3 == 0; }
};

using writable_iterator_t = std::ranges::iterator_t<archetype_view<writable_archetype>>;

static_assert(std::indirect_unary_predicate<write_pred, writable_iterator_t>);
static_assert(
    std::indirect_unary_predicate<write_pred, std::ranges::iterator_t<archetype_view<writable_archetype_dc>>>);
static_assert(std::indirectly_writable<writable_iterator_t, const write_value&>);
static_assert(std::indirectly_writable<std::ranges::iterator_t<archetype_view<writable_archetype_dc>>,
                                       const write_value_dc&>);
static_assert(!std::default_initializable<writable_archetype_dc>);
static_assert(!std::copyable<writable_archetype>);
static_assert(!std::movable<writable_archetype>);
static_assert(!std::default_initializable<writable_archetype>);

struct replaceable_archetype
{
    int val;

    using value_arg = write_value;

    explicit replaceable_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(replaceable_archetype)

    replaceable_archetype&
    operator=(const write_value& __v)
    {
        val = __v.val;
        return *this;
    }

    friend bool operator==(const replaceable_archetype& __e1, const replaceable_archetype& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const replaceable_archetype& __e, const search_value& __v) { return __e.val == __v.val; }
};

struct replaceable_archetype_dc
{
    int val;

    using value_arg = write_value_dc;

    explicit replaceable_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(replaceable_archetype_dc)

    replaceable_archetype_dc&
    operator=(const write_value_dc& __v)
    {
        val = __v.val;
        return *this;
    }

    friend bool operator==(const replaceable_archetype_dc& __e1, const replaceable_archetype_dc& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const replaceable_archetype_dc& __e, const search_value& __v)
    {
        return __e.val == __v.val;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(replaceable_archetype_dc)

struct copy_out_archetype;

struct copy_in_archetype
{
    int val;

    using out_type = copy_out_archetype;

    explicit copy_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(copy_in_archetype)
};

struct remove_copy_in_archetype
{
    int val;

    using out_type = copy_out_archetype;

    explicit remove_copy_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(remove_copy_in_archetype)

    friend bool operator==(const remove_copy_in_archetype& __e1, const remove_copy_in_archetype& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const remove_copy_in_archetype& __e, const search_value& __v)
    {
        return __e.val == __v.val;
    }
};

struct copy_out_archetype
{
    int val;

    using value_arg = write_value;

    explicit copy_out_archetype(int __v) : val(__v) {}

    copy_out_archetype(const copy_out_archetype&) = delete;
    copy_out_archetype(copy_out_archetype&&) = delete;
    copy_out_archetype& operator=(const copy_out_archetype&) = delete;
    copy_out_archetype& operator=(copy_out_archetype&&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    copy_out_archetype& operator=(copy_in_archetype& __v)
    {
        val = __v.val;
        return *this;
    }

    copy_out_archetype&
    operator=(remove_copy_in_archetype& __v)
    {
        val = __v.val;
        return *this;
    }

    copy_out_archetype&
    operator=(const write_value& __v)
    {
        val = __v.val;
        return *this;
    }
};

struct copy_out_archetype_dc;

struct copy_in_archetype_dc
{
    int val;

    using out_type = copy_out_archetype_dc;

    explicit copy_in_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(copy_in_archetype_dc)
};

struct remove_copy_in_archetype_dc
{
    int val;

    using out_type = copy_out_archetype_dc;

    explicit remove_copy_in_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(remove_copy_in_archetype_dc)

    friend bool operator==(const remove_copy_in_archetype_dc& __e1, const remove_copy_in_archetype_dc& __e2)
    {
        return __e1.val == __e2.val;
    }

    friend bool operator==(const remove_copy_in_archetype_dc& __e, const search_value& __v)
    {
        return __e.val == __v.val;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(remove_copy_in_archetype_dc)

struct copy_out_archetype_dc
{
    int val;

    using value_arg = write_value_dc;

    explicit copy_out_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(copy_out_archetype_dc)

    copy_out_archetype_dc& operator=(copy_in_archetype_dc& __v)
    {
        val = __v.val;
        return *this;
    }

    copy_out_archetype_dc&
    operator=(remove_copy_in_archetype_dc& __v)
    {
        val = __v.val;
        return *this;
    }

    copy_out_archetype_dc&
    operator=(const write_value_dc& __v)
    {
        val = __v.val;
        return *this;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(copy_in_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(copy_out_archetype_dc)
static_assert(std::indirectly_copyable<std::ranges::iterator_t<archetype_view<copy_in_archetype_dc>>,
                                       std::ranges::iterator_t<archetype_view<copy_out_archetype_dc>>>);
static_assert(!std::default_initializable<copy_out_archetype_dc>);

using copy_in_iterator_t = std::ranges::iterator_t<archetype_view<copy_in_archetype>>;
using copy_out_iterator_t = std::ranges::iterator_t<archetype_view<copy_out_archetype>>;

static_assert(std::indirectly_copyable<copy_in_iterator_t, copy_out_iterator_t>);
static_assert(std::indirectly_writable<copy_out_iterator_t, const write_value&>);
static_assert(std::indirectly_writable<std::ranges::iterator_t<archetype_view<copy_out_archetype_dc>>,
                                       const write_value_dc&>);
static_assert(!std::copyable<copy_in_archetype>);
static_assert(!std::copyable<copy_out_archetype>);
static_assert(!std::default_initializable<copy_out_archetype>);

struct copy_pred
{
    bool operator()(const copy_in_archetype& __v) const { return __v.val % 3 == 0; }
    bool operator()(const copy_in_archetype_dc& __v) const { return __v.val % 3 == 0; }
};

struct copy_equiv
{
    bool operator()(const copy_in_archetype& __v1, const copy_in_archetype& __v2) const
    {
        return __v1.val / 3 == __v2.val / 3;
    }
    bool operator()(const copy_in_archetype_dc& __v1, const copy_in_archetype_dc& __v2) const
    {
        return __v1.val / 3 == __v2.val / 3;
    }
};

static_assert(std::indirect_unary_predicate<copy_pred, copy_in_iterator_t>);
static_assert(std::indirect_equivalence_relation<copy_equiv, copy_in_iterator_t>);
static_assert(std::indirect_unary_predicate<copy_pred, std::ranges::iterator_t<archetype_view<copy_in_archetype_dc>>>);
static_assert(
    std::indirect_equivalence_relation<copy_equiv, std::ranges::iterator_t<archetype_view<copy_in_archetype_dc>>>);

struct move_in_archetype
{
    int val;

    explicit move_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(move_in_archetype)
};

struct move_out_archetype
{
    int val;

    explicit move_out_archetype(int __v) : val(__v) {}

    move_out_archetype(const move_out_archetype&) = delete;
    move_out_archetype(move_out_archetype&&) = delete;
    move_out_archetype& operator=(const move_out_archetype&) = delete;
    move_out_archetype& operator=(move_out_archetype&&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    move_out_archetype& operator=(move_in_archetype&& __v)
    {
        val = __v.val;
        return *this;
    }
};

struct move_in_archetype_dc
{
    int val;

    explicit move_in_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(move_in_archetype_dc)
};

struct move_out_archetype_dc
{
    int val;

    explicit move_out_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(move_out_archetype_dc)

    move_out_archetype_dc& operator=(move_in_archetype_dc&& __v)
    {
        val = __v.val;
        return *this;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(move_in_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(move_out_archetype_dc)
static_assert(std::indirectly_movable<std::ranges::iterator_t<archetype_view<move_in_archetype_dc>>,
                                      std::ranges::iterator_t<archetype_view<move_out_archetype_dc>>>);
static_assert(!std::indirectly_copyable<std::ranges::iterator_t<archetype_view<move_in_archetype_dc>>,
                                        std::ranges::iterator_t<archetype_view<move_out_archetype_dc>>>);

using move_in_iterator_t = std::ranges::iterator_t<archetype_view<move_in_archetype>>;
using move_out_iterator_t = std::ranges::iterator_t<archetype_view<move_out_archetype>>;

static_assert(std::indirectly_movable<move_in_iterator_t, move_out_iterator_t>);
static_assert(!std::indirectly_copyable<move_in_iterator_t, move_out_iterator_t>);
static_assert(!std::movable<move_out_archetype>);

struct swap_archetype
{
    int val;

    explicit swap_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(swap_archetype)

    friend void swap(swap_archetype& __v1, swap_archetype& __v2)
    {
        const int __tmp = __v1.val;
        __v1.val = __v2.val;
        __v2.val = __tmp;
    }
};

struct swap_archetype_dc
{
    int val;

    explicit swap_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(swap_archetype_dc)

    friend void swap(swap_archetype_dc& __v1, swap_archetype_dc& __v2)
    {
        const int __tmp = __v1.val;
        __v1.val = __v2.val;
        __v2.val = __tmp;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(swap_archetype_dc)
static_assert(!std::default_initializable<swap_archetype_dc>);

using swap_iterator_t = std::ranges::iterator_t<archetype_view<swap_archetype>>;
static_assert(std::indirectly_swappable<swap_iterator_t, swap_iterator_t>);
static_assert(!std::movable<swap_archetype>);
static_assert(!std::move_constructible<swap_archetype>);
static_assert(!std::default_initializable<swap_archetype>);

struct transform_out_archetype;

struct transform_in_archetype
{
    int val;

    using out_type = transform_out_archetype;

    explicit transform_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(transform_in_archetype)
};

struct transform_result
{
    int val;
};

struct transform_out_archetype
{
    int val;

    explicit transform_out_archetype(int __v) : val(__v) {}

    transform_out_archetype(const transform_out_archetype&) = delete;
    transform_out_archetype(transform_out_archetype&&) = delete;
    transform_out_archetype& operator=(const transform_out_archetype&) = delete;
    transform_out_archetype& operator=(transform_out_archetype&&) = delete;
    TEST_ARCHETYPE_DELETED_ADDRESSOF

    transform_out_archetype& operator=(const transform_result& __v)
    {
        val = __v.val;
        return *this;
    }
};

struct transform_out_archetype_dc;

struct transform_in_archetype_dc
{
    int val;

    using out_type = transform_out_archetype_dc;

    explicit transform_in_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(transform_in_archetype_dc)
};

struct transform_out_archetype_dc
{
    int val;

    explicit transform_out_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(transform_out_archetype_dc)

    transform_out_archetype_dc& operator=(const transform_result& __v)
    {
        val = __v.val;
        return *this;
    }
};

TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(transform_in_archetype_dc)
TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE(transform_out_archetype_dc)
static_assert(!std::default_initializable<transform_out_archetype_dc>);

struct transform_unary_op
{
    transform_result operator()(const transform_in_archetype& __v) const { return transform_result{__v.val * 2}; }
    transform_result operator()(const transform_in_archetype_dc& __v) const { return transform_result{__v.val * 2}; }
};

struct transform_binary_op
{
    transform_result operator()(const transform_in_archetype& __v1, const transform_in_archetype& __v2) const
    {
        return transform_result{__v1.val + __v2.val};
    }
    transform_result operator()(const transform_in_archetype_dc& __v1, const transform_in_archetype_dc& __v2) const
    {
        return transform_result{__v1.val + __v2.val};
    }
};

using transform_in_iterator_t = std::ranges::iterator_t<archetype_view<transform_in_archetype>>;
using transform_out_iterator_t = std::ranges::iterator_t<archetype_view<transform_out_archetype>>;

static_assert(std::copy_constructible<transform_unary_op>);
static_assert(std::copy_constructible<transform_binary_op>);
static_assert(std::indirectly_writable<transform_out_iterator_t,
                                       std::indirect_result_t<transform_unary_op&, transform_in_iterator_t>>);
static_assert(std::indirectly_writable<
              transform_out_iterator_t,
              std::indirect_result_t<transform_binary_op&, transform_in_iterator_t, transform_in_iterator_t>>);
static_assert(!std::copyable<transform_out_archetype>);
static_assert(!std::default_initializable<transform_out_archetype>);

struct transform_proj_result
{
    int val;
};

struct transform_proj
{
    transform_proj_result operator()(const transform_in_archetype& __v) const
    {
        return transform_proj_result{__v.val + 1};
    }
    transform_proj_result operator()(const transform_in_archetype_dc& __v) const
    {
        return transform_proj_result{__v.val + 1};
    }
};

struct transform_projected_unary_op
{
    transform_result operator()(const transform_proj_result& __v) const { return transform_result{__v.val * 2}; }
};

struct transform_projected_binary_op
{
    transform_result operator()(const transform_proj_result& __v1, const transform_proj_result& __v2) const
    {
        return transform_result{__v1.val + __v2.val};
    }
};

using transform_projected_iterator_t = std::projected<transform_in_iterator_t, transform_proj>;

static_assert(std::copy_constructible<transform_proj>);
static_assert(std::indirectly_regular_unary_invocable<transform_proj, transform_in_iterator_t>);
static_assert(std::indirectly_writable<
              transform_out_iterator_t,
              std::indirect_result_t<transform_projected_unary_op&, transform_projected_iterator_t>>);
static_assert(std::indirectly_writable<transform_out_iterator_t,
                                       std::indirect_result_t<transform_projected_binary_op&,
                                                              transform_projected_iterator_t,
                                                              transform_projected_iterator_t>>);
static_assert(!std::invocable<transform_projected_unary_op&, transform_in_archetype&>);
static_assert(!std::invocable<transform_projected_binary_op&, transform_in_archetype&, transform_in_archetype&>);
static_assert(!std::indirectly_writable<transform_out_iterator_t, transform_proj_result>);

struct write_pred_mut
{
    bool operator()(writable_archetype& __v) const { return __v.val % 3 == 0; }
    bool operator()(writable_archetype_dc& __v) const { return __v.val % 3 == 0; }
};

static_assert(std::indirect_unary_predicate<write_pred_mut, writable_iterator_t>);
static_assert(!std::invocable<const write_pred_mut&, const writable_archetype&>);

struct copy_pred_mut
{
    bool operator()(copy_in_archetype& __v) const { return __v.val % 3 == 0; }
    bool operator()(copy_in_archetype_dc& __v) const { return __v.val % 3 == 0; }
};

struct copy_equiv_mut
{
    bool operator()(copy_in_archetype& __v1, copy_in_archetype& __v2) const { return __v1.val / 3 == __v2.val / 3; }
    bool operator()(copy_in_archetype_dc& __v1, copy_in_archetype_dc& __v2) const
    {
        return __v1.val / 3 == __v2.val / 3;
    }
};

static_assert(std::indirect_unary_predicate<copy_pred_mut, copy_in_iterator_t>);
static_assert(std::indirect_equivalence_relation<copy_equiv_mut, copy_in_iterator_t>);
static_assert(!std::invocable<const copy_pred_mut&, const copy_in_archetype&>);
static_assert(!std::invocable<const copy_equiv_mut&, const copy_in_archetype&, const copy_in_archetype&>);

struct transform_unary_op_mut
{
    transform_result operator()(transform_in_archetype& __v) const { return transform_result{__v.val * 2}; }
    transform_result operator()(transform_in_archetype_dc& __v) const { return transform_result{__v.val * 2}; }
};

struct transform_binary_op_mut
{
    transform_result operator()(transform_in_archetype& __v1, transform_in_archetype& __v2) const
    {
        return transform_result{__v1.val + __v2.val};
    }
    transform_result operator()(transform_in_archetype_dc& __v1, transform_in_archetype_dc& __v2) const
    {
        return transform_result{__v1.val + __v2.val};
    }
};

struct transform_proj_mut
{
    transform_proj_result operator()(transform_in_archetype& __v) const { return transform_proj_result{__v.val + 1}; }
    transform_proj_result operator()(transform_in_archetype_dc& __v) const
    {
        return transform_proj_result{__v.val + 1};
    }
};

static_assert(std::indirectly_writable<transform_out_iterator_t,
                                       std::indirect_result_t<transform_unary_op_mut&, transform_in_iterator_t>>);
static_assert(!std::invocable<const transform_unary_op_mut&, const transform_in_archetype&>);
static_assert(std::indirectly_writable<
              transform_out_iterator_t,
              std::indirect_result_t<transform_binary_op_mut&, transform_in_iterator_t, transform_in_iterator_t>>);
static_assert(
    !std::invocable<const transform_binary_op_mut&, const transform_in_archetype&, const transform_in_archetype&>);
static_assert(std::indirectly_regular_unary_invocable<transform_proj_mut, transform_in_iterator_t>);
static_assert(!std::invocable<const transform_proj_mut&, const transform_in_archetype&>);
static_assert(std::indirectly_writable<
              transform_out_iterator_t,
              std::indirect_result_t<transform_projected_unary_op&,
                                     std::projected<transform_in_iterator_t, transform_proj_mut>>>);

struct replace_common
{
    int val;

    replace_common(const replaceable_archetype& __e) : val(__e.val) {}
    replace_common(const replaceable_archetype_dc& __e) : val(__e.val) {}
    replace_common(const remove_copy_in_archetype& __e) : val(__e.val) {}
    replace_common(const remove_copy_in_archetype_dc& __e) : val(__e.val) {}
    replace_common(const search_value& __v) : val(__v.val) {}

    friend bool operator==(const replace_common& __v1, const replace_common& __v2) { return __v1.val == __v2.val; }
};

} // namespace archetypes
} // namespace test_std_ranges

namespace std
{
template <>
struct common_type<test_std_ranges::archetypes::replaceable_archetype, test_std_ranges::archetypes::search_value>
{
    using type = test_std_ranges::archetypes::replace_common;
};

template <>
struct common_type<test_std_ranges::archetypes::search_value, test_std_ranges::archetypes::replaceable_archetype>
{
    using type = test_std_ranges::archetypes::replace_common;
};

template <>
struct common_type<test_std_ranges::archetypes::replaceable_archetype_dc, test_std_ranges::archetypes::search_value>
{
    using type = test_std_ranges::archetypes::replace_common;
};

template <>
struct common_type<test_std_ranges::archetypes::search_value, test_std_ranges::archetypes::replaceable_archetype_dc>
{
    using type = test_std_ranges::archetypes::replace_common;
};

template <>
struct common_type<test_std_ranges::archetypes::remove_copy_in_archetype, test_std_ranges::archetypes::search_value>
{
    using type = test_std_ranges::archetypes::replace_common;
};

template <>
struct common_type<test_std_ranges::archetypes::search_value, test_std_ranges::archetypes::remove_copy_in_archetype>
{
    using type = test_std_ranges::archetypes::replace_common;
};

template <>
struct common_type<test_std_ranges::archetypes::remove_copy_in_archetype_dc, test_std_ranges::archetypes::search_value>
{
    using type = test_std_ranges::archetypes::replace_common;
};

template <>
struct common_type<test_std_ranges::archetypes::search_value, test_std_ranges::archetypes::remove_copy_in_archetype_dc>
{
    using type = test_std_ranges::archetypes::replace_common;
};
} // namespace std

namespace test_std_ranges
{
namespace archetypes
{

using replaceable_iterator_t = std::ranges::iterator_t<archetype_view<replaceable_archetype>>;
using replaceable_dc_iterator_t = std::ranges::iterator_t<archetype_view<replaceable_archetype_dc>>;
using remove_copy_in_iterator_t = std::ranges::iterator_t<archetype_view<remove_copy_in_archetype>>;
using remove_copy_in_dc_iterator_t = std::ranges::iterator_t<archetype_view<remove_copy_in_archetype_dc>>;

static_assert(std::indirectly_writable<replaceable_iterator_t, const write_value&>);
static_assert(std::indirect_binary_predicate<std::ranges::equal_to, replaceable_iterator_t, const search_value*>);
static_assert(std::indirectly_writable<replaceable_dc_iterator_t, const write_value_dc&>);
static_assert(std::indirect_binary_predicate<std::ranges::equal_to, replaceable_dc_iterator_t, const search_value*>);
static_assert(std::indirectly_copyable<remove_copy_in_iterator_t, copy_out_iterator_t>);
static_assert(std::indirect_binary_predicate<std::ranges::equal_to, remove_copy_in_iterator_t, const search_value*>);
static_assert(std::indirectly_copyable<remove_copy_in_dc_iterator_t,
                                       std::ranges::iterator_t<archetype_view<copy_out_archetype_dc>>>);
static_assert(
    std::indirect_binary_predicate<std::ranges::equal_to, remove_copy_in_dc_iterator_t, const search_value*>);
static_assert(!std::indirectly_writable<replaceable_iterator_t, const search_value&>);
static_assert(!std::copyable<replaceable_archetype>);
static_assert(!std::movable<replaceable_archetype>);
static_assert(!std::default_initializable<replaceable_archetype>);
static_assert(!std::totally_ordered<replaceable_archetype>);
static_assert(!std::copyable<remove_copy_in_archetype>);
static_assert(!std::default_initializable<remove_copy_in_archetype>);
static_assert(!std::totally_ordered<remove_copy_in_archetype>);
static_assert(!std::default_initializable<replaceable_archetype_dc>);
static_assert(!std::totally_ordered<replaceable_archetype_dc>);
static_assert(!std::default_initializable<remove_copy_in_archetype_dc>);
static_assert(!std::totally_ordered<remove_copy_in_archetype_dc>);

struct replace_proj_mut
{
    replaceable_archetype& operator()(replaceable_archetype& __v) const { return __v; }
    replaceable_archetype_dc& operator()(replaceable_archetype_dc& __v) const { return __v; }
    remove_copy_in_archetype& operator()(remove_copy_in_archetype& __v) const { return __v; }
    remove_copy_in_archetype_dc& operator()(remove_copy_in_archetype_dc& __v) const { return __v; }
};

static_assert(std::indirect_binary_predicate<std::ranges::equal_to,
                                             std::projected<replaceable_iterator_t, replace_proj_mut>,
                                             const search_value*>);
static_assert(std::indirect_binary_predicate<std::ranges::equal_to,
                                             std::projected<remove_copy_in_iterator_t, replace_proj_mut>,
                                             const search_value*>);
static_assert(!std::invocable<const replace_proj_mut&, const replaceable_archetype&>);
static_assert(!std::invocable<const replace_proj_mut&, const remove_copy_in_archetype&>);

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_WRITE_H
