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
#include "support/move_only.h"

void
swap(MoveOnly&, MoveOnly&)
{
    // Intentionally skip swapping to provide effect different from std::swap
}

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    dpl::tuple<> t1, t2;
                    dpl::swap(t1, t2);

                    ret_acc[0] &= (t1 == t2);
                }
                {
                    dpl::tuple<int> t1(1), t2(2);
                    dpl::swap(t1, t2);

                    ret_acc[0] &= (dpl::get<0>(t1) == 2 && dpl::get<0>(t2) == 1);
                }
                {
                    dpl::tuple<int, float> t1(1, 1.0f), t2(2, 2.0f);
                    dpl::swap(t1, t2);

                    ret_acc[0] &= (dpl::get<0>(t1) == 2 && dpl::get<0>(t2) == 1);
                    ret_acc[0] &= (dpl::get<1>(t1) == 2.0f && dpl::get<1>(t2) == 1.0f);
                }
                {
                    dpl::tuple<int, float, MoveOnly> t1(1, 1.0f, MoveOnly(1)), t2(2, 2.0f, MoveOnly(2));

                    dpl::swap(t1, t2);

                    ret_acc[0] &= (dpl::get<0>(t1) == 2 && dpl::get<0>(t2) == 1);
                    ret_acc[0] &= (dpl::get<1>(t1) == 2.0f && dpl::get<1>(t2) == 1.0f);
                    ret_acc[0] &= (dpl::get<2>(t1) == MoveOnly(1) && dpl::get<2>(t2) == MoveOnly(2));
                }
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::swap check");

    return TestUtils::done();
}
