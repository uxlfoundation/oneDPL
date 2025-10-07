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
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

int
main()
{
    bool ret = true;
    {
        sycl::queue deviceQueue = TestUtils::get_test_queue();
        sycl::range<1> numOfItems{1};
        sycl::buffer<bool, 1> buf1(&ret, numOfItems);

        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf1.get_access<sycl::access::mode::read_write>(cgh);

            cgh.single_task<class KernelIteratorTest1>([=]() {
                {
                    typedef dpl::array<int, 5> C;
                    C c;
                    C::iterator i;
                    i = c.begin();
                    C::const_iterator j;
                    j = c.cbegin();
                    ret_acc[0] &= (i == j);
                }
                {
                    typedef dpl::array<int, 0> C;
                    C c;
                    C::iterator i;
                    i = c.begin();
                    C::const_iterator j;
                    j = c.cbegin();
                    ret_acc[0] &= (i == j);
                }
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::array::begin / dpl::array::cbegin");

    return TestUtils::done();
}
