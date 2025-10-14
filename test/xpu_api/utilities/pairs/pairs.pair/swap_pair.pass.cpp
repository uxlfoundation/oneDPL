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

struct S
{
    int i;
    S() : i(0) {}
    S(int j) : i(j) {}
    bool
    operator==(int x) const
    {
        return i == x;
    }
};

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
                typedef dpl::pair<int, short> P1;
                P1 p1(3, static_cast<short>(4));
                P1 p2(5, static_cast<short>(6));
                p1.swap(p2);
                ret_access[0] = (p1.first == 5);
                ret_access[0] &= (p1.second == 6);
                ret_access[0] &= (p2.first == 3);
                ret_access[0] &= (p2.second == 4);
            }
            {
                typedef dpl::pair<int, S> P1;
                P1 p1(3, S(4));
                P1 p2(5, S(6));
                p1.swap(p2);
                ret_access[0] &= (p1.first == 5);
                ret_access[0] &= (p1.second == 6);
                ret_access[0] &= (p2.first == 3);
                ret_access[0] &= (p2.second == 4);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::pair::swap check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
