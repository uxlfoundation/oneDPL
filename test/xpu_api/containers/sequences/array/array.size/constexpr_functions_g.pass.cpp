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

#include "support/utils.h"

struct constexpr_member_functions
{
    template <typename _Ttesttype>
    void
    operator()()
    {
        struct _Concept
        {
            void
            __constraint()
            {
                constexpr _Ttesttype a = {};
                constexpr auto v1 __attribute__((unused)) = a.size();
                constexpr auto v2 __attribute__((unused)) = a.max_size();
                constexpr auto v3 __attribute__((unused)) = a.empty();
            }
        };

        _Concept c;
        c.__constraint();
    }
};

bool
kernel_test()
{
    constexpr_member_functions test;
    test.operator()<dpl::array<long, 60>>();
    return true;
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }
    EXPECT_TRUE(ret, "Wrong result of work with dpl::array constexpr methods");

    return TestUtils::done();
}
