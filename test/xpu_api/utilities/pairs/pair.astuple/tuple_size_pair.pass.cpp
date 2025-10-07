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
            {
                typedef dpl::pair<int, short> P1;
                static_assert(dpl::tuple_size<P1>::value == 2);
                static_assert(std::tuple_size_v<P1> == 2);
                ret_access[0] = (dpl::tuple_size<P1>::value == 2);
                ret_access[0] &= (std::tuple_size_v<P1> == 2);
            }

            {
                typedef dpl::pair<int, short> const P1;
                static_assert(dpl::tuple_size<P1>::value == 2);
                static_assert(std::tuple_size_v<P1> == 2);
                ret_access[0] &= (dpl::tuple_size<P1>::value == 2);
                ret_access[0] &= (std::tuple_size_v<P1> == 2);
            }

            {
                typedef dpl::pair<int, short> volatile P1;
                static_assert(dpl::tuple_size<P1>::value == 2);
                static_assert(std::tuple_size_v<P1> == 2);
                ret_access[0] &= (dpl::tuple_size<P1>::value == 2);
                ret_access[0] &= (std::tuple_size_v<P1> == 2);
            }

            {
                typedef dpl::pair<int, short> const volatile P1;
                static_assert(std::tuple_size_v<P1> == 2);
                ret_access[0] &= (dpl::tuple_size<P1>::value == 2);
                ret_access[0] &= (std::tuple_size_v<P1> == 2);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple_size check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
