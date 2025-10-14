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

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

struct A
{
};

bool
kernel_test()
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                //  Test the explicit deduction guides
                {
                    //  optional(T)
                    dpl::optional opt(5);
                    static_assert(dpl::is_same_v<decltype(opt), dpl::optional<int>>);
                    ret_access[0] &= (static_cast<bool>(opt));
                    ret_access[0] &= (*opt == 5);
                }

                {
                    //  optional(T)
                    dpl::optional opt(A{});
                    static_assert(dpl::is_same_v<decltype(opt), dpl::optional<A>>);
                    ret_access[0] &= (static_cast<bool>(opt));
                }

                //  Test the implicit deduction guides
                {
                    //  optional(optional);
                    dpl::optional<char> source('A');
                    dpl::optional opt(source);
                    static_assert(dpl::is_same_v<decltype(opt), dpl::optional<char>>);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(source));
                    ret_access[0] &= (*opt == *source);
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
    EXPECT_TRUE(ret, "Wrong result of dpl::optional deduction check");

    return TestUtils::done();
}
