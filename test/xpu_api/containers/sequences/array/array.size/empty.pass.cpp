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
// bool empty() const noexcept;

#include "support/test_config.h"

#include <oneapi/dpl/array>

#include "support/utils.h"

int
main()
{
    {
        auto ret = true;
        {
            sycl::queue deviceQueue = TestUtils::get_test_queue();
            sycl::buffer<bool, 1> buf(&ret, sycl::range<1>(1));

            deviceQueue.submit([&](sycl::handler& cgh) {
                auto ret_acc = buf.get_access<sycl::access::mode::read_write>(cgh);

                cgh.single_task<class KernelEmptyTest1>([=]() {
                    {
                        typedef dpl::array<int, 2> C;
                        C c;
                        ret_acc[0] &= (!c.empty());
                    }
                    {
                        typedef dpl::array<int, 0> C;
                        C c;
                        ret_acc[0] &= (c.empty());
                    }
                });
            });
        }

        EXPECT_TRUE(ret, "Wrong result of work with dpl::array::empty");
    }

    return TestUtils::done();
}
