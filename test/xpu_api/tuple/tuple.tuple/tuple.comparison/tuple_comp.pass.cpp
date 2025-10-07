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

#include "support/test_config.h"

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if !_PSTL_TEST_COMPARISON_BROKEN
template <typename X>
bool
test(const X& x)
{
    return (x == x && !(x != x) && x <= x && !(x < x));
}

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItem{1};

    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                int i = 0;
                int j = 0;
                int k = 2;
                dpl::tuple<int, int, int> a(0, 0, 0);
                dpl::tuple<int, int, int> b(0, 0, 1);
                dpl::tuple<int&, int&, int&> c(i, j, k);
                dpl::tuple<const int&, const int&, const int&> d(c);
                ret_acc[0] &= test(a);
                ret_acc[0] &= test(b);
                ret_acc[0] &= test(c);
                ret_acc[0] &= test(d);
                ret_acc[0] &= (!(a > a) && !(b > b));
                ret_acc[0] &= (a >= a && b >= b);
                ret_acc[0] &= (a < b && !(b < a) && a <= b && !(b <= a));
                ret_acc[0] &= (b > a && !(a > b) && b >= a && !(a >= b));
            });
        });
    }
    return ret;
}
#endif // !_PSTL_TEST_COMPARISON_BROKEN

int
main()
{
    bool bProcessed = false;

#if !_PSTL_TEST_COMPARISON_BROKEN
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::tuple comparison check");
    bProcessed = true;
#endif // !_PSTL_TEST_COMPARISON_BROKEN

    return TestUtils::done(bProcessed);
}
