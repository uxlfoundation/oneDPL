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

void
test_non_default_constructible()
{
    struct X
    {
        X() = delete;
    };

    typedef dpl::pair<int, X> P;
    static_assert(!dpl::is_constructible<P>::value);
    static_assert(!dpl::is_default_constructible<P>::value);
}

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
                dpl::pair<float, short*> p;
                ret_access[0] = (p.first == 0.0f);
                ret_access[0] &= (p.second == nullptr);
            }

            {
                typedef dpl::pair<float, short*> P;
                constexpr P p;
                static_assert(p.first == 0.0f);
                static_assert(p.second == nullptr);
            }

            test_non_default_constructible();
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::pair default constructor check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
