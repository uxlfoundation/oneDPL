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

#include <oneapi/dpl/array>

#include "support/utils.h"

int
main()
{
    bool ret = true;

    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                {
                    typedef int T;
                    typedef dpl::array<T, 3> C;
                    C c = {1, 2, 35};
                    c.fill(55);
                    ret_acc[0] &= (c.size() == 3);
                    ret_acc[0] &= (c[0] == 55);
                    ret_acc[0] &= (c[1] == 55);
                    ret_acc[0] &= (c[2] == 55);
                }
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::array::fill");

    return TestUtils::done();
}
