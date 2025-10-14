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

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                dpl::tuple<> ta;
                dpl::tuple<> tb;
                ta = tb;

                dpl::tuple<int> tc(1);
                dpl::tuple<int> td(0);
                td = tc;
                ret_access[0] &= (dpl::get<0>(td) == 1);

                int i = 0;
                dpl::tuple<int&> t1(i);
                t1 = tc;
                ret_access[0] = (i == 1);

                dpl::tuple<const int&> tf(tc);

                dpl::get<0>(tc) = 2;
                ret_access[0] &= (dpl::get<0>(tf) == 2);
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::tuple assignment check");

    return TestUtils::done();
}
