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
#include "support/utils_invoke.h"

bool
kernel_test(sycl::queue deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                dpl::tuple<int, float> a(1, 2.f), b;
                b = dpl::move(a);
                ret_access[0] &= (dpl::get<0>(b) == 1 && dpl::get<1>(b) == 2.f);
                ret_access[0] &= (dpl::get<0>(a) == 1 && dpl::get<1>(a) == 2.f);

                dpl::tuple<int, double> c(dpl::move(b));
                ret_access[0] &= (dpl::get<0>(c) == 1 && dpl::get<1>(c) == 2.f);
                ret_access[0] &= (dpl::get<0>(b) == 1 && dpl::get<1>(b) == 2.f);
            });
        });
    }
    return ret;
}

int
main()
{
    bool processed = false;

    sycl::queue deviceQueue = TestUtils::get_test_queue();
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        auto ret = kernel_test(deviceQueue);
        EXPECT_TRUE(ret, "Wrong result of dpl::tuple move check");
        processed = true;
    }

    return TestUtils::done(processed);
}
