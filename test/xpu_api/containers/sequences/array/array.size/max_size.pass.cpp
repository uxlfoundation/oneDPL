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

// <array>
// class array
// bool max_size() const noexcept;

#include "support/test_config.h"

#include <oneapi/dpl/array>

#include "support/utils.h"

int
main()
{
    {
        bool ret = true;
        {
            sycl::queue myQueue = TestUtils::get_test_queue();
            sycl::range<1> numOfItems{1};
            sycl::buffer<bool, 1> buf1(&ret, numOfItems);

            myQueue.submit([&](sycl::handler& cgh) {
                auto ret_acc = buf1.get_access<sycl::access::mode::write>(cgh);

                cgh.single_task<class KernelMaxSizeTest1>([=]() {
                    {
                        typedef dpl::array<int, 2> C;
                        C c;
                        (void) noexcept(c.max_size());
                        ret_acc[0] &= (c.max_size() == 2);
                    }
                    {
                        typedef dpl::array<int, 0> C;
                        C c;
                        (void) noexcept(c.max_size());
                        ret_acc[0] &= (c.max_size() == 0);
                    }
                });
            });
        }

        EXPECT_TRUE(ret, "Wrong result of work with dpl::array::max/dpl::array::size");
    }

    return TestUtils::done();
}
