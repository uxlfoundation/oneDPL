// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

struct destroy_archetype
{
    int val1;
    volatile int val2;

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
