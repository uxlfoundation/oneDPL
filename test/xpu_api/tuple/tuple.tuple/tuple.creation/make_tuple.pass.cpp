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

class KernelMakeTupleTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelMakeTupleTest>([=]() {
            int i = 4;
            float j = 5.f;
            dpl::tuple<int, int&, float&> t = dpl::make_tuple(1, dpl::ref(i), dpl::ref(j));

            ret_access[0] = (dpl::get<0>(t) == 1);
            ret_access[0] &= (dpl::get<1>(t) == i);
            ret_access[0] &= (dpl::get<2>(t) == j);

            i = 2;
            j = 3.5f;
            ret_access[0] &= (dpl::get<0>(t) == 1);
            ret_access[0] &= (dpl::get<1>(t) == 2);
            ret_access[0] &= (dpl::get<2>(t) == 3.5f);

            dpl::get<1>(t) = 0;
            dpl::get<2>(t) = 0.f;
            ret_access[0] &= (i == 0);
            ret_access[0] &= (j == 0.f);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::make_tuple check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
