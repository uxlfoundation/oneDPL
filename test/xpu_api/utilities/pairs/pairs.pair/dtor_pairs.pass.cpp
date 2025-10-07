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

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

class KernelPairTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            static_assert(dpl::is_trivially_destructible<dpl::pair<int, float>>::value);
            ret_access[0] = (dpl::is_trivially_destructible<dpl::pair<int, float>>::value);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::is_trivially_destructible check for dpl::pair");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
