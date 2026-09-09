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

#ifndef _STD_RANGES_ARCHETYPES_WRITE_H
#define _STD_RANGES_ARCHETYPES_WRITE_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"

namespace test_std_ranges
{
namespace archetypes
{

// Family 4: algorithms writing a value into the range itself.
// The constraint is std::indirectly_writable<iterator_t<_R>, const _T&>, which needs `*it = value`
// for a const lvalue value and nothing else: the element still does not have to be copyable,
// movable or default constructible, and _T stays an unrelated type.
// Used by: fill, replace_if, replace (new value), replace_copy_if / replace_copy (new value).
struct write_value
{
    int val;

    explicit write_value(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(write_value)
};

// The device copyable counterpart of write_value: a value argument is passed to a device kernel by
// copy, so the hetero policies need a trivially copyable one.
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

    // The value type the algorithm has to be called with, so that a generic test body may pick the
    // right one for the element type it works on.
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

using writable_iterator_t = std::ranges::iterator_t<archetype_view<writable_archetype>>;

static_assert(std::indirectly_writable<writable_iterator_t, const write_value&>);
static_assert(std::indirectly_writable<std::ranges::iterator_t<archetype_view<writable_archetype_dc>>,
                                       const write_value_dc&>);
static_assert(!std::default_initializable<writable_archetype_dc>);
static_assert(!std::copyable<writable_archetype>);
static_assert(!std::movable<writable_archetype>);
static_assert(!std::default_initializable<writable_archetype>);

// Family 5: copying algorithms.
// std::indirectly_copyable<In, Out> == indirectly_readable<In> && indirectly_writable<Out,
// iter_reference_t<In>>, so the output element only has to be assignable from a non-const lvalue of
// the input element type. Neither element type has to be copyable, movable or default
// constructible, and the two types are deliberately different.
// Used by: copy, copy_if, reverse_copy, rotate_copy, remove_copy, remove_copy_if, unique_copy,
// replace_copy, replace_copy_if, partition_copy, partial_sort_copy.
struct copy_in_archetype
{
    int val;

    explicit copy_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(copy_in_archetype)
};

struct copy_out_archetype
{
    int val;

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
};

// The device copyable counterparts of the two archetypes above, used with the hetero policies.
struct copy_in_archetype_dc
{
    int val;

    explicit copy_in_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(copy_in_archetype_dc)
};

struct copy_out_archetype_dc
{
    int val;

    explicit copy_out_archetype_dc(int __v) : val(__v) {}

    TEST_ARCHETYPE_DEFAULTED_OPERATIONS(copy_out_archetype_dc)

    copy_out_archetype_dc& operator=(copy_in_archetype_dc& __v)
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
static_assert(!std::copyable<copy_in_archetype>);
static_assert(!std::copyable<copy_out_archetype>);
static_assert(!std::default_initializable<copy_out_archetype>);

// Family 6: the move algorithm.
// std::indirectly_movable<In, Out> asks for indirectly_writable<Out, iter_rvalue_reference_t<In>>,
// so the output element is only assignable from an rvalue of the input element type: an
// implementation which copies instead of moving does not compile.
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

// The device copyable counterparts of the two archetypes above, used with the hetero policies. The
// assignment from a non-const lvalue of the input type is still missing, so an implementation which
// copies instead of moving does not compile either.
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
// An lvalue is explicitly rejected, so copying instead of moving is a compilation error.
static_assert(!std::indirectly_copyable<move_in_iterator_t, move_out_iterator_t>);
static_assert(!std::movable<move_out_archetype>);

// Family 7: swap_ranges.
// std::indirectly_swappable<It1, It2> needs std::ranges::swap on the two references, both ways. A
// dedicated hidden-friend swap is provided, so the element does not have to be move constructible
// or move assignable, which is what the fallback std::swap would require.
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

// The device copyable counterpart of the archetype above, used with the hetero policies. The
// dedicated swap is kept, so the algorithm still has to go through std::ranges::swap.
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

// Family 8: transform.
// The output constraint is
//   std::indirectly_writable<iterator_t<_OutRange>, std::indirect_result_t<_F&, projected...>>
// so the output element is only assignable from the result of the functor, which is a third,
// unrelated type. _F itself is only required to be std::copy_constructible.
struct transform_out_archetype;

struct transform_in_archetype
{
    int val;

    // The output element type the algorithm has to be called with, so that a generic test body which
    // allocates the output range itself may pick the right one for the input element type it works on.
    using out_type = transform_out_archetype;

    explicit transform_in_archetype(int __v) : val(__v) {}

    TEST_ARCHETYPE_DELETED_OPERATIONS(transform_in_archetype)
};

// The result of the functor. indirectly_writable requires the assignment to work for the prvalue,
// the const lvalue and the const rvalue forms of the result type, which a prvalue-returning functor
// naturally provides.
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

// The device copyable counterparts of the two archetypes above, used with the hetero policies.
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

// Both transform overloads project their input before invoking the functor, and the requires-clause
// spells the functor over std::projected, so the functor never sees the element itself. The
// projection returns yet another unrelated type: an implementation which applies the functor to the
// element, or writes the projected value into the output, does not compile.
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
// The projected functors reject the element type, and the output element rejects the projected
// value, so neither the projection nor the functor can be skipped by the implementation.
static_assert(!std::invocable<transform_projected_unary_op&, transform_in_archetype&>);
static_assert(!std::invocable<transform_projected_binary_op&, transform_in_archetype&, transform_in_archetype&>);
static_assert(!std::indirectly_writable<transform_out_iterator_t, transform_proj_result>);

// Family 8: transform. The functor is only required to be std::copy_constructible and invocable with
// the projected reference, which is a non-const lvalue.
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

// A projection taking its argument by non-const reference. The functor invoked with the projected
// value cannot do the same: the projection returns a prvalue, which does not bind to a non-const
// lvalue reference, so the projected functors of the const section are reused with this projection.
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

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_WRITE_H
