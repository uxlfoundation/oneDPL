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

// template<LessThanComparable T>
//   const T&
//   min(const T& a, const T& b);

#include "support/test_complex.h"
#include "support/test_macros.h"

#include <oneapi/dpl/algorithm>
#include <cassert>

template <class T>
void
test(const T& a, const T& b, const T& x)
{
    assert(&dpl::min(a, b) == &x);
}

ONEDPL_TEST_NUM_MAIN
{
    {
    int x = 0;
    int y = 0;
    test(x, y, x);
    test(y, x, y);
    }
    {
    int x = 0;
    int y = 1;
    test(x, y, x);
    test(y, x, x);
    }
    {
    int x = 1;
    int y = 0;
    test(x, y, y);
    test(y, x, y);
    }
#if TEST_STD_VER >= 14
    {
    constexpr int x = 1;
    constexpr int y = 0;
    static_assert(dpl::min(x, y) == y);
    static_assert(dpl::min(y, x) == y);
    }
#endif

  return 0;
}
