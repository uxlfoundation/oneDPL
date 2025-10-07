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

using dpl::optional;

struct X
{
    int
    test()
    {
        return 0;
    }

    constexpr int
    test() const
    {
        return 3;
    }
};

struct Y
{
    int
    test() noexcept
    {
        return 0;
    }

    int
    test() const noexcept
    {
        return 2;
    }
};

struct Z
{
    const Z*
    operator&() const;

    int
    test()
    {
        return 0;
    }

    constexpr int
    test() const
    {
        return 1;
    }
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
                {
                    optional<X> opt(X{});
                    ret_access[0] &= (opt->test() == 0);
                }
                {
                    constexpr optional<X> opt(X{});
                    static_assert(opt->test() == 3);
                }
                {
                    optional<Y> opt(Y{});
                    ret_access[0] &= (opt->test() == 0);
                }
                {
                    constexpr optional<Y> opt(Y{});
                    ret_access[0] &= (opt->test() == 2);
                }
                {
                    optional<Z> opt(Z{});
                    ret_access[0] &= (opt->test() == 0);
                }
                {
                    constexpr optional<Z> opt(Z{});
                    static_assert(opt->test() == 1);
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
    EXPECT_TRUE(ret, "Wrong result of const dpl::reference::operator-> in kernel_test");

    return TestUtils::done();
}
