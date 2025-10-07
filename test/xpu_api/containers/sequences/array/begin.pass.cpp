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
    {
        auto ret = true;
        {
            sycl::queue myQueue = TestUtils::get_test_queue();
            sycl::buffer<bool, 1> buf1(&ret, sycl::range<1>(1));

            myQueue.submit([&](sycl::handler& cgh) {
                auto ret_access = buf1.get_access<sycl::access::mode::read_write>(cgh);

                cgh.single_task<class KernelBeginTest>([=]() {
                    typedef int T;
                    typedef dpl::array<T, 3> C;
                    C c = {1, 2, 35};
                    C::iterator i;
                    i = c.begin();
                    ret_access[0] &= (*i == 1);
                    ret_access[0] &= (&*i == c.data());
                    *i = 55;
                    ret_access[0] &= (c[0] == 55);
                });
            });
        }

        EXPECT_TRUE(ret, "Wrong result of work with dpl::array::begin");
    }

    return TestUtils::done();
}
