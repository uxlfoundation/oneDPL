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
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

using dpl::optional;

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
                {
                    constexpr optional<int> opt(int(5));
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == 5);

                    struct test_constexpr_ctor : public optional<int>
                    {
                        constexpr test_constexpr_ctor(int&& arg) : optional<int>(std::move(arg)) {}
                    };
                }
                {
                    constexpr optional<float> opt(float(3));
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == 3);

                    struct test_constexpr_ctor : public optional<float>
                    {
                        constexpr test_constexpr_ctor(float&& arg) : optional<float>(std::move(arg)) {}
                    };
                }
                {
                    const int x = 42;
                    optional<const int> o(dpl::move(x));
                    ret_access[0] &= (*o == 42);
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
    EXPECT_TRUE(ret, "Wrong result of dpl::optional passed as r-value check");

    return TestUtils::done();
}
