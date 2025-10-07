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

struct NoDefault
{
    NoDefault() {}
    NoDefault(int) {}
};

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
                    typedef dpl::array<float, 3> C;
                    C c;
                    ret_acc[0] &= (c.size() == 3);
                }
                {
                    typedef dpl::array<int, 0> C;
                    C c;
                    ret_acc[0] &= (c.size() == 0);
                }
                {
                    typedef dpl::array<NoDefault, 0> C;
                    C c;
                    ret_acc[0] &= (c.size() == 0);
                    C c1 = {};
                    ret_acc[0] &= (c1.size() == 0);
                    C c2 = {{}};
                    ret_acc[0] &= (c2.size() == 0);
                }
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::array");

    return TestUtils::done();
}
