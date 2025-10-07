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

#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if !_PSTL_TEST_COMPARISON_BROKEN
class KernelPairTest;
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            {
                typedef dpl::pair<int, short> P;
                P p1(3, static_cast<short>(4));
                P p2(3, static_cast<short>(4));
                ret_access[0] = ((p1 == p2));
                ret_access[0] &= (!(p1 != p2));
                ret_access[0] &= (!(p1 < p2));
                ret_access[0] &= ((p1 <= p2));
                ret_access[0] &= (!(p1 > p2));
                ret_access[0] &= ((p1 >= p2));
            }

            {
                typedef dpl::pair<int, short> P;
                P p1(2, static_cast<short>(4));
                P p2(3, static_cast<short>(4));
                ret_access[0] &= (!(p1 == p2));
                ret_access[0] &= ((p1 != p2));
                ret_access[0] &= ((p1 < p2));
                ret_access[0] &= ((p1 <= p2));
                ret_access[0] &= (!(p1 > p2));
                ret_access[0] &= (!(p1 >= p2));
            }

            {
                typedef dpl::pair<int, short> P;
                P p1(3, static_cast<short>(2));
                P p2(3, static_cast<short>(4));
                ret_access[0] &= (!(p1 == p2));
                ret_access[0] &= ((p1 != p2));
                ret_access[0] &= ((p1 < p2));
                ret_access[0] &= ((p1 <= p2));
                ret_access[0] &= (!(p1 > p2));
                ret_access[0] &= (!(p1 >= p2));
            }

            {
                typedef dpl::pair<int, short> P;
                P p1(3, static_cast<short>(4));
                P p2(2, static_cast<short>(4));
                ret_access[0] &= (!(p1 == p2));
                ret_access[0] &= ((p1 != p2));
                ret_access[0] &= (!(p1 < p2));
                ret_access[0] &= (!(p1 <= p2));
                ret_access[0] &= ((p1 > p2));
                ret_access[0] &= ((p1 >= p2));
            }

            {
                typedef dpl::pair<int, short> P;
                P p1(3, static_cast<short>(4));
                P p2(3, static_cast<short>(2));
                ret_access[0] &= (!(p1 == p2));
                ret_access[0] &= ((p1 != p2));
                ret_access[0] &= (!(p1 < p2));
                ret_access[0] &= (!(p1 <= p2));
                ret_access[0] &= ((p1 > p2));
                ret_access[0] &= ((p1 >= p2));
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::pair comparison check");
}

#endif // !_PSTL_TEST_COMPARISON_BROKEN

int
main()
{
    bool bProcessed = false;

#if !_PSTL_TEST_COMPARISON_BROKEN
    kernel_test();
    bProcessed = true;
#endif // !_PSTL_TEST_COMPARISON_BROKEN

    return TestUtils::done(bProcessed);
}
