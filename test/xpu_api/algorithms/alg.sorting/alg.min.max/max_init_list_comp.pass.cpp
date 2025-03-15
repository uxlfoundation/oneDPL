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

// UNSUPPORTED: c++03

// <algorithm>

// template<class T, class Compare>
//   T
//   max(initializer_list<T> t, Compare comp);

#include "support/test_complex.h"
#include "support/test_macros.h"

#include <oneapi/dpl/algorithm>
#include <functional>
#include <cassert>

ONEDPL_TEST_NUM_MAIN
{
    int i = dpl::max({2, 3, 1}, dpl::greater<int>());
    assert(i == 1);
    i = dpl::max({2, 1, 3}, dpl::greater<int>());
    assert(i == 1);
    i = dpl::max({3, 1, 2}, dpl::greater<int>());
    assert(i == 1);
    i = dpl::max({3, 2, 1}, dpl::greater<int>());
    assert(i == 1);
    i = dpl::max({1, 2, 3}, dpl::greater<int>());
    assert(i == 1);
    i = dpl::max({1, 3, 2}, dpl::greater<int>());
    assert(i == 1);
#if TEST_STD_VER >= 14
    {
    static_assert(dpl::max({1, 3, 2}, dpl::greater<int>()) == 1);
    static_assert(dpl::max({2, 1, 3}, dpl::greater<int>()) == 1);
    static_assert(dpl::max({3, 2, 1}, dpl::greater<int>()) == 1);
    }
#endif

  return 0;
}
