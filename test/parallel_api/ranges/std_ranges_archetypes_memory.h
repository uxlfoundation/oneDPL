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

#ifndef _STD_RANGES_ARCHETYPES_MEMORY_H
#define _STD_RANGES_ARCHETYPES_MEMORY_H

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes_base.h"

namespace test_std_ranges
{
namespace archetypes
{

// std::default_initializable, required by uninitialized_default_construct.
// The default constructor is user-provided, so default- and value-initialization are the same and
// val2 is left untouched by the algorithm.
struct default_construct_archetype
{
    int val1;
    int val2;

    default_construct_archetype() { val1 = 1; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(default_construct_archetype)
};

static_assert(std::default_initializable<default_construct_archetype>);
static_assert(std::destructible<default_construct_archetype>);
static_assert(!std::copy_constructible<default_construct_archetype>);
static_assert(!std::move_constructible<default_construct_archetype>);
static_assert(!std::equality_comparable<default_construct_archetype>);
static_assert(!std::swappable<default_construct_archetype>);

// std::default_initializable, required by uninitialized_value_construct.
// The default constructor is defaulted on its first declaration and therefore is not user-provided:
// value-initialization zero-initializes the whole object, which lets the test tell value
// construction apart from default construction.
struct value_construct_archetype
{
    int val1;
    int val2;

    value_construct_archetype() = default;

    TEST_ARCHETYPE_DELETED_OPERATIONS(value_construct_archetype)
};

static_assert(std::default_initializable<value_construct_archetype>);
static_assert(std::destructible<value_construct_archetype>);
static_assert(!std::copy_constructible<value_construct_archetype>);
static_assert(!std::move_constructible<value_construct_archetype>);
static_assert(!std::equality_comparable<value_construct_archetype>);
static_assert(!std::swappable<value_construct_archetype>);

// The _T template parameter of uninitialized_fill is deduced from the value argument, so the filler
// type is deliberately different from the range value type: the only required conversion is
// std::constructible_from<range_value_t<_R>, const fill_source&>.
struct fill_source
{
    int val;
};

struct fill_archetype
{
    int val1;
    int val2;

    explicit fill_archetype(const fill_source& src) { val2 = src.val; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(fill_archetype)
};

static_assert(std::constructible_from<fill_archetype, const fill_source&>);
static_assert(std::destructible<fill_archetype>);
static_assert(!std::default_initializable<fill_archetype>);
static_assert(!std::copy_constructible<fill_archetype>);
static_assert(!std::move_constructible<fill_archetype>);

// Input element type of uninitialized_copy and uninitialized_move. No constraint is imposed on it
// besides forming a random access range, so it is only constructible from an int, which is what the
// test harness uses to prepare the input data.
struct transfer_source
{
    int val1;
    int val2;

    explicit transfer_source(int v) { val2 = v; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(transfer_source)
};

static_assert(std::destructible<transfer_source>);
static_assert(!std::default_initializable<transfer_source>);
static_assert(!std::copy_constructible<transfer_source>);
static_assert(!std::move_constructible<transfer_source>);

// std::constructible_from<range_value_t<_OutRange>, range_reference_t<_InRange>>, required by
// uninitialized_copy. range_reference_t of a range of transfer_source is exactly transfer_source&,
// so the implementation must not pass a const lvalue or an rvalue instead.
struct copy_archetype
{
    int val1;
    int val2;

    explicit copy_archetype(transfer_source& src) { val2 = src.val2; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(copy_archetype)
};

static_assert(std::constructible_from<copy_archetype, transfer_source&>);
static_assert(std::destructible<copy_archetype>);
static_assert(!std::constructible_from<copy_archetype, const transfer_source&>);
static_assert(!std::constructible_from<copy_archetype, transfer_source&&>);
static_assert(!std::default_initializable<copy_archetype>);
static_assert(!std::copy_constructible<copy_archetype>);

// std::constructible_from<range_value_t<_OutRange>, range_rvalue_reference_t<_InRange>>, required by
// uninitialized_move. Only an rvalue is accepted, so the implementation has to move the source
// element (std::ranges::iter_move) rather than copy it.
struct move_archetype
{
    int val1;
    int val2;

    explicit move_archetype(transfer_source&& src) { val2 = src.val2; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(move_archetype)
};

static_assert(std::constructible_from<move_archetype, transfer_source&&>);
static_assert(std::destructible<move_archetype>);
static_assert(!std::constructible_from<move_archetype, transfer_source&>);
static_assert(!std::default_initializable<move_archetype>);
static_assert(!std::copy_constructible<move_archetype>);

// std::destructible, required by destroy. No constructor at all is declared, which is enough for the
// test: the harness works on raw memory and only observes the effect of the destructor.
struct destroy_archetype
{
    int val1;
    volatile int val2; // volatile prevents optimization of the destructor observed with g++

    ~destroy_archetype() { val2 = 3; }

    TEST_ARCHETYPE_DELETED_OPERATIONS(destroy_archetype)
};

static_assert(std::destructible<destroy_archetype>);
static_assert(!std::default_initializable<destroy_archetype>);
static_assert(!std::copy_constructible<destroy_archetype>);
static_assert(!std::move_constructible<destroy_archetype>);

} // namespace archetypes
} // namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_MEMORY_H
