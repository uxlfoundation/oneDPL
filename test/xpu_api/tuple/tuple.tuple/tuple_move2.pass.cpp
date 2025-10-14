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
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef dpl::tuple<MoveOnly> move_only_tuple;

                move_only_tuple t1(MoveOnly{});
                move_only_tuple t2(dpl::move(t1));
                move_only_tuple t3 = dpl::move(t2);
                t1 = dpl::move(t3);

                typedef dpl::tuple<MoveOnly, MoveOnly> move_only_tuple2;

                move_only_tuple2 t4(MoveOnly{}, MoveOnly{});
                move_only_tuple2 t5(dpl::move(t4));
                move_only_tuple2 t6 = dpl::move(t5);
                t4 = dpl::move(t6);
            });
        });
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
