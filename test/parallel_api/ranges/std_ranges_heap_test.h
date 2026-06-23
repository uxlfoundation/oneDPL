// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _STD_RANGES_HEAP_TEST_H
#define _STD_RANGES_HEAP_TEST_H

#if _ENABLE_STD_RANGES_TESTING

#include <limits> // std::numeric_limits
#include <bit>    // std::bit_width

namespace test_std_ranges
{
using element_t = int;

template <typename T>
struct CustomLess
{
    bool
    operator()(const T& a, const T& b) const
    {
        return std::less<T>{}(a, b);
    }
};

template <typename T>
struct CustomGreat
{
    bool
    operator()(const T& a, const T& b) const
    {
        return std::greater<T>{}(a, b);
    }
};

struct MaxHeapGenerator
{
    element_t
    operator()(std::size_t idx) const
    {
        return static_cast<element_t>(-idx);
    }
};

struct ThroughParentHeapGenerator
{
    element_t
    operator()(std::size_t idx) const
    {
        if (idx == 0)
            return element_t{0};

        return std::numeric_limits<element_t>::max() - static_cast<element_t>(std::bit_width(idx + 1u));
    }
};

struct NonHeapGenerator
{
    element_t
    operator()(std::size_t idx) const
    {
        return static_cast<element_t>(idx);
    }
};

template <typename GeneratorT>
struct CorruptedHeapGenerator
{
    element_t
    operator()(std::size_t idx) const
    {
        if (idx == corrupted_element_idx)
            return 1;
        return generator(idx);
    }

    const GeneratorT generator;
    const std::size_t corrupted_element_idx = {};
};

} //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING
#endif //_STD_RANGES_HEAP_TEST_H
