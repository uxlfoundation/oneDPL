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
#include <oneapi/dpl/memory>

#include "support/utils.h"

void
test_contiguous()
{
    bool ret = true;
    {
        sycl::queue myQueue = TestUtils::get_test_queue();
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>(1));

        myQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf.get_access<sycl::access::mode::read_write>(cgh);

            cgh.single_task<class KernelContiguousTest>([=]() {
                typedef float T;
                typedef dpl::array<T, 3> C;
                C c = {1, 2, 3};
                for (size_t i = 0; i < c.size(); ++i)
                    ret_acc[0] &= (*(c.begin() + i) == *(dpl::addressof(*c.begin()) + i));
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of contiquous check");
}

int
main()
{
    test_contiguous();

    return TestUtils::done();
}
