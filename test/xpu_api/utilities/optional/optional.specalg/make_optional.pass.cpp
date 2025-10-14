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

struct TestT
{
    int x;
    int size;
    constexpr TestT(std::initializer_list<int> il) : x(*il.begin()), size(static_cast<int>(il.size())) {}
    constexpr TestT(std::initializer_list<int> il, const int*) : x(*il.begin()), size(static_cast<int>(il.size())) {}
};

bool
kernel_test()
{
    bool ret = true;
    {
        sycl::queue q = TestUtils::get_test_queue();
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                using dpl::optional;
                using dpl::make_optional;
                {
                    optional<int> opt = make_optional(2);
                    ret_access[0] &= (*opt == 2);
                }

                {
                    constexpr auto opt = make_optional<int>('a');
                    static_assert(*opt == int('a'));
                }

                {
                    constexpr auto opt = make_optional<TestT>({42, 2, 3});
                    static_assert(opt->x == 42);
                    static_assert(opt->size == 3);
                }

                {
                    constexpr auto opt = make_optional<TestT>({42, 2, 3}, nullptr);
                    static_assert(opt->x == 42);
                    static_assert(opt->size == 3);
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
    EXPECT_TRUE(ret, "Wrong result of dpl::make_optional in kernel_test");

    return TestUtils::done();
}
