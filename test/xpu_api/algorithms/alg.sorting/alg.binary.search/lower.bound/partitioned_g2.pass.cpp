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

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/functional>

#include <iostream>

#include "support/utils.h"

struct X
{
    int val;

    bool
    odd() const
    {
        return val % 2;
    }

    // Partitioned so that all odd values come before even values:
    bool
    operator<(const X& x) const
    {
        return this->odd() && !x.odd();
    }
    bool
    operator==(const X& x) const
    {
        return this->val == x.val;
    }
};

bool
kernel_test1(sycl::queue& deviceQueue)
{
    // Test with range that is partitioned, but not sorted.
    X seq[] = {1, 3, 5, 7, 1, 6, 4, 2};
    auto tmp = seq;
    const int N = sizeof(seq) / sizeof(seq[0]);
    bool ret = false;
    bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};

    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, item1);
        sycl::buffer<X, 1> buffer3(seq, itemN);
        deviceQueue
            .submit([&](sycl::handler& cgh) {
                auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
                auto check_access = buffer2.get_access<sycl::access::mode::write>(cgh);
                auto access = buffer3.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<class KernelTest1>([=]() {
                    X arr[] = {1, 3, 5, 7, 1, 6, 4, 2};
                    // check if there is change after data transfer
                    check_access[0] = TestUtils::check_data(&access[0], arr, N);
                    if (check_access[0])
                    {
                        auto itBegin = &access[0];
                        auto itEnd = &access[0] + N;

                        auto part1 = dpl::lower_bound(itBegin, itEnd, X{2});
                        ret_access[0] = (part1 != itEnd);
                        ret_access[0] &= (part1->val == 6);
                        auto part2 = dpl::lower_bound(itBegin, itEnd, X{2}, dpl::less<X>{});
                        ret_access[0] &= (part2 != itEnd);
                        ret_access[0] &= (part2->val == 6);

                        auto part3 = dpl::lower_bound(itBegin, itEnd, X{9});
                        ret_access[0] &= (part3 != itEnd);
                        ret_access[0] &= (part3->val == 1);
                        auto part4 = dpl::lower_bound(itBegin, itEnd, X{9}, dpl::less<X>{});
                        ret_access[0] &= (part4 != itEnd);
                        ret_access[0] &= (part4->val == 1);
                    }
                });
            })
            .wait();
    }
    // check if there is change after executing kernel function
    check &= TestUtils::check_data(tmp, seq, N);
    if (!check)
        return false;
    return ret;
}

struct Y
{
    double val;

    // Not irreflexive, so not a strict weak order.
    bool
    operator<(const Y& y) const
    {
        return val < int(y.val);
    }
    bool
    operator==(const Y& y) const
    {
        return val == y.val;
    }
};

bool
kernel_test2(sycl::queue& deviceQueue)
{
    Y seq[] = {-0.1, 1.2, 5.0, 5.2, 5.1, 5.9, 5.5, 6.0};
    auto tmp = seq;
    const int N = sizeof(seq) / sizeof(seq[0]);
    bool ret = false;
    bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{8};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, item1);
        sycl::buffer<Y, 1> buffer3(seq, itemN);
        deviceQueue
            .submit([&](sycl::handler& cgh) {
                auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
                auto check_access = buffer2.get_access<sycl::access::mode::write>(cgh);
                auto access = buffer3.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<class KernelTest2>([=]() {
                    Y arr[] = {-0.1, 1.2, 5.0, 5.2, 5.1, 5.9, 5.5, 6.0};
                    // check if there is change after data transfer
                    check_access[0] = TestUtils::check_data(&access[0], arr, N);
                    if (check_access[0])
                    {
                        auto itBegin = &access[0];
                        auto itEnd = &access[0] + N;

                        auto part1 = std::lower_bound(itBegin, itEnd, Y{5.5});
                        ret_access[0] = (part1 != itEnd);
                        ret_access[0] &= (part1->val == 5.0);
                        auto part2 = std::lower_bound(itBegin, itEnd, Y{5.5}, std::less<Y>{});
                        ret_access[0] &= (part2 != itEnd);
                        ret_access[0] &= (part2->val == 5.0);

                        auto part3 = std::lower_bound(itBegin, itEnd, Y{1.0});
                        ret_access[0] &= (part3 != itEnd);
                        ret_access[0] &= (part3->val == 1.2);
                        auto part4 = std::lower_bound(itBegin, itEnd, Y{1.0}, std::less<Y>{});
                        ret_access[0] &= (part4 != itEnd);
                        ret_access[0] &= (part4->val == 1.2);
                    }
                });
            })
            .wait();
    }
    // check if there is change after executing kernel function
    check &= TestUtils::check_data(tmp, seq, N);
    if (!check)
        return false;
    return ret;
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    auto ret = kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        ret &= kernel_test2(deviceQueue);
    }
    EXPECT_TRUE(ret, "Wrong result of lower_bound in kernel_test1 (or in kernel_test2)");

    return TestUtils::done();
}
