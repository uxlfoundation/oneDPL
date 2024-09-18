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
    X(int v) : val(v){};
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
kernel_test()
{
    // Test with range that is partitioned, but not sorted.
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    X seq[] = {1, 3, 5, 7, 1, 6, 4};
    auto tmp = seq;
    bool ret = false;
    bool check = false;
    const int N = sizeof(seq) / sizeof(seq[0]);
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, itemN);
        sycl::buffer<X, 1> buffer3(seq, itemN);
        deviceQueue
            .submit([&](sycl::handler& cgh) {
                auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
                auto check_access = buffer2.get_access<sycl::access::mode::write>(cgh);
                auto access = buffer3.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<class KernelTest>([=]() {
                    X tmp[] = {1, 3, 5, 7, 1, 6, 4};
                    // check if there is change after data transfer
                    check_access[0] = TestUtils::check_data(&access[0], &tmp[0], N);

                    if (check_access[0])
                    {
                        auto itBegin = &access[0];
                        auto itEnd = &access[0] + N;

                        ret_access[0] = dpl::binary_search(itBegin, itEnd, X{2});
                        ret_access[0] &= dpl::binary_search(itBegin, itEnd, X{2}, dpl::less<X>{});

                        ret_access[0] &= dpl::binary_search(itBegin, itEnd, X{9});
                        ret_access[0] &= dpl::binary_search(itBegin, itEnd, X{9}, dpl::less<X>{});

                        ret_access[0] &= dpl::binary_search(itBegin, itEnd, X{2}, dpl::less<X>{});

                        ret_access[0] &= dpl::binary_search(itBegin, itEnd, X{9});
                        ret_access[0] &= dpl::binary_search(itBegin, itEnd, X{9}, dpl::less<X>{});

                        ret_access[0] &= !(dpl::binary_search(itBegin, itBegin + 5, X{2}));
                        ret_access[0] &= !(dpl::binary_search(itBegin, itBegin + 5, X{2}, dpl::less<X>{}));
                    }
                });
            })
            .wait();
    }
    // check if there is change after executing kernel function
    check &= TestUtils::check_data(seq, tmp, N);
    if (!check)
        return false;
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of binary_search in kernel_test");

    return TestUtils::done();
}
