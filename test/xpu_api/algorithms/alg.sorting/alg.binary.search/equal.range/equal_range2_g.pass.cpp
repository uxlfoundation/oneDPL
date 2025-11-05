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

#include <iostream>

#include "support/utils.h"

// A comparison, equivalent to std::greater<int> without the
// dependency on <functional>.
struct gt
{
    bool
    operator()(const int& x, const int& y) const
    {
        return x > y;
    }
};

// Each test performs general-case, bookend, not-found condition,
// and predicate functional checks.

// equal_range, with and without comparison predicate
bool
kernel_test()
{
    using dpl::equal_range;
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    bool check = false;

    typedef std::pair<const int*, const int*> Ipair;
    const int A[] = {1, 2, 3, 3, 3, 5, 8};
    const int C[] = {8, 5, 3, 3, 3, 2, 1};
    auto A1 = A, C1 = C;
    const int N = sizeof(A) / sizeof(int);
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};

    const int first = A[0];
    const int last = A[N - 1];

    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, item1);
        sycl::buffer<int, 1> buffer3(A, itemN);
        sycl::buffer<int, 1> buffer4(C, itemN);
        deviceQueue
            .submit([&](sycl::handler& cgh) {
                auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
                auto check_access = buffer2.get_access<sycl::access::mode::write>(cgh);
                auto access1 = buffer3.get_access<sycl::access::mode::write>(cgh);
                auto access2 = buffer4.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<class KernelTest1>([=]() {
                    const int A1[] = {1, 2, 3, 3, 3, 5, 8};
                    const int C1[] = {8, 5, 3, 3, 3, 2, 1};
                    // check if there is change after data transfer
                    check_access[0] = TestUtils::check_data(&access1[0], A1, N);
                    check_access[0] &= TestUtils::check_data(&access2[0], C1, N);
                    if (check_access[0])
                    {
                        auto itBegin = &access1[0];
                        auto itEnd = &access1[0] + N;

                        Ipair p = equal_range(itBegin, itEnd, 3);
                        ret_access[0] = (p.first == itBegin + 2);
                        ret_access[0] &= (p.second == itBegin + 5);

                        Ipair q = equal_range(itBegin, itEnd, first);
                        ret_access[0] &= (q.first == itBegin + 0);
                        ret_access[0] &= (q.second == itBegin + 1);

                        Ipair r = equal_range(itBegin, itEnd, last);
                        ret_access[0] &= (r.first == itEnd - 1);
                        ret_access[0] &= (r.second == itEnd);

                        Ipair s = equal_range(itBegin, itEnd, 4);
                        ret_access[0] &= (s.first == itBegin + 5);
                        ret_access[0] &= (s.second == itBegin + 5);

                        Ipair t = equal_range(&access2[0], &access2[0] + N, 3, gt());
                        ret_access[0] &= (t.first == &access2[0] + 2);
                        ret_access[0] &= (t.second == &access2[0] + 5);

                        auto itBegin2 = &access2[0];
                        auto itEnd2 = &access2[0] + N;

                        Ipair u = equal_range(itBegin2, itEnd2, first, gt());
                        ret_access[0] &= (u.first == itEnd2 - 1);
                        ret_access[0] &= (u.second == itEnd2);

                        Ipair v = equal_range(itBegin2, itEnd2, last, gt());
                        ret_access[0] &= (v.first == itBegin2 + 0);
                        ret_access[0] &= (v.second == itBegin2 + 1);

                        Ipair w = equal_range(itBegin2, itEnd2, 4, gt());
                        ret_access[0] &= (w.first == itBegin2 + 2);
                        ret_access[0] &= (w.second == itBegin2 + 2);
                    }
                });
            })
            .wait();
    }
    // check if there is change after executing kernel function
    check &= TestUtils::check_data(A, A1, N);
    check &= TestUtils::check_data(C, C1, N);
    if (!check)
        return false;
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of equal_range in kernel_test");

    return TestUtils::done();
}
