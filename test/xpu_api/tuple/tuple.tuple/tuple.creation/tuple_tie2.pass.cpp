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

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                int x1 = 0;
                int x2 = 0;
                int y1 = 0;
                int y2 = 0;
                dpl::tuple<int, int> ta(1, 1);
                dpl::tuple<const int&, const int&> tc(x1, x2);
                dpl::tie(y1, y2) = ta;
                ret_access[0] &= (y1 == 1 && y2 == 1);
                dpl::tie(y1, y2) = tc;
                ret_access[0] &= (y1 == 0 && y2 == 0);
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::tie in kernel_test");

    return TestUtils::done();
}
