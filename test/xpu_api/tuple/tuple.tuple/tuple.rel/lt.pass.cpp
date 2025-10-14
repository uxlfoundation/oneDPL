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

#include "support/test_macros.h"
#include "support/utils.h"

#if !_PSTL_TEST_COMPARISON_BROKEN
class KernelTupleLTTest1;
class KernelTupleLTTest2;

void
kernel_test1(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTupleLTTest1>([=]() {
            {
                const dpl::tuple<> t1;
                const dpl::tuple<> t2;

                ret_access[0] = (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }

            {
                const dpl::tuple<long> t1(1);
                const dpl::tuple<float> t2(1.f);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long> t1(1);
                const dpl::tuple<float> t2(0.9f);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long> t1(1);
                const dpl::tuple<float> t2(1.1f);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<float, long> t2(1.f, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<float, long> t2(0.9f, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<float, long> t2(1.1f, 2);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<float, long> t2(1.f, 1);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<float, long> t2(1.f, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int, float> t1(1, 2, 3.f);
                const dpl::tuple<float, long, int> t2(1.f, 2, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int, float> t1(1, 2, 3.f);
                const dpl::tuple<float, long, int> t2(0.9f, 2, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int, float> t1(1, 2, 3.f);
                const dpl::tuple<float, long, int> t2(1.1f, 2, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int, float> t1(1, 2, 3.f);
                const dpl::tuple<float, long, int> t2(1.f, 1, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int, float> t1(1, 2, 3.f);
                const dpl::tuple<float, long, int> t2(1.f, 3, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int, float> t1(1, 2, 3.f);
                const dpl::tuple<float, long, int> t2(1.f, 2, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int, float> t1(1, 2, 3.f);
                const dpl::tuple<float, long, int> t2(1.f, 2, 4);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple comparison operators check");
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTupleLTTest2>([=]() {
            {
                const dpl::tuple<long> t1(1);
                const dpl::tuple<double> t2(1);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long> t1(1);
                const dpl::tuple<double> t2(0.9);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long> t1(1);
                const dpl::tuple<double> t2(1.1);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<double, long> t2(1, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<double, long> t2(0.9, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<double, long> t2(1.1, 2);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<double, long> t2(1, 1);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int> t1(1, 2);
                const dpl::tuple<double, long> t2(1, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int, double> t1(1, 2, 3);
                const dpl::tuple<double, long, int> t2(1, 2, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int, double> t1(1, 2, 3);
                const dpl::tuple<double, long, int> t2(0.9, 2, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int, double> t1(1, 2, 3);
                const dpl::tuple<double, long, int> t2(1.1, 2, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int, double> t1(1, 2, 3);
                const dpl::tuple<double, long, int> t2(1, 1, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int, double> t1(1, 2, 3);
                const dpl::tuple<double, long, int> t2(1, 3, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                const dpl::tuple<long, int, double> t1(1, 2, 3);
                const dpl::tuple<double, long, int> t2(1, 2, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                const dpl::tuple<long, int, double> t1(1, 2, 3);
                const dpl::tuple<double, long, int> t2(1, 2, 4);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple comparison check");
}
#endif // !_PSTL_TEST_COMPARISON_BROKEN

int
main()
{
    bool bProcessed = false;

#if !_PSTL_TEST_COMPARISON_BROKEN
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test2(deviceQueue);
    }
    bProcessed = true;
#endif // !_PSTL_TEST_COMPARISON_BROKEN

    return TestUtils::done(bProcessed);
}
