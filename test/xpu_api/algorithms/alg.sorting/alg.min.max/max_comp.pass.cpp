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

// <algorithm>

// template<class T, StrictWeakOrder<auto, T> Compare>
//   requires !SameType<T, Compare> && CopyConstructible<Compare>
//   const T&
//   max(const T& a, const T& b, Compare comp);

#include "support/test_complex.h"
#include "support/test_macros.h"

#include <oneapi/dpl/algorithm>
#include <functional>
#include <cassert>

template <class T, class C>
void
test(const T& a, const T& b, C c, const T& x)
{
    assert(&dpl::max(a, b, c) == &x);
}

ONEDPL_TEST_NUM_MAIN
{
    {
    int x = 0;
    int y = 0;
    test(x, y, dpl::greater<int>(), x);
    test(y, x, dpl::greater<int>(), y);
    }
    {
    int x = 0;
    int y = 1;
    test(x, y, dpl::greater<int>(), x);
    test(y, x, dpl::greater<int>(), x);
    }
    {
    int x = 1;
    int y = 0;
    test(x, y, dpl::greater<int>(), y);
    test(y, x, dpl::greater<int>(), y);
    }
#if TEST_STD_VER >= 14
    {
    constexpr int x = 1;
    constexpr int y = 0;
    static_assert(dpl::max(x, y, dpl::greater<int>()) == y);
    static_assert(dpl::max(y, x, dpl::greater<int>()) == y);
    }
#endif

  return 0;
}
