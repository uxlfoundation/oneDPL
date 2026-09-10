// -*- C++ -*-
//===-- direct_iterator.pass.cpp ------------------------------------------===//
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include "support/utils.h"

#include <iostream>
#include <vector>
#include <numeric>

#if __cpp_lib_span >= 202002L
#include <span>
#endif

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    using T = int;

    const int n = 1000;

    std::vector<T> v(n);
    std::iota(v.begin(), v.end(), 0);

    sycl::queue q(sycl::default_selector_v);

    T* p = sycl::malloc_device<T>(n, q);

    q.memcpy(p, v.data(), n * sizeof(T)).wait();

    auto v_ref = std::reduce(v.begin(), v.end(), 0);

    dpl::make_direct_iterator d_first(p);
    dpl::make_direct_iterator d_last(p + n);

    auto v_dev = dpl::reduce(d_first, d_last, 0);

    EXPECT_EQ(v_ref, v_dev);

#if __cpp_lib_span >= 202002L

    std::span<T> x(p, n);

    dpl::make_direct_iterator s_first(x.begin());
    dpl::make_direct_iterator s_last(x.end());

    auto s_dev = dpl::reduce(s_first, s_last, 0);

    EXPECT_EQ(v_ref, s_dev);
#endif
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
