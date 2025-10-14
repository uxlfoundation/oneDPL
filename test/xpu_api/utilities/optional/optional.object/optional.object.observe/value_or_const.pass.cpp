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

struct Y
{
    int i_;

    constexpr Y(int i) : i_(i) {}
};

struct X
{
    int i_;

    constexpr X(int i) : i_(i) {}
    constexpr X(const Y& y) : i_(y.i_) {}
    constexpr X(Y&& y) : i_(y.i_ + 1) {}
    friend constexpr bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
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
                    constexpr optional<X> opt(2);
                    constexpr Y y(3);
                    static_assert(opt.value_or(y) == 2);
                }
                {
                    constexpr optional<X> opt(2);
                    static_assert(opt.value_or(Y(3)) == 2);
                }
                {
                    constexpr optional<X> opt;
                    constexpr Y y(3);
                    static_assert(opt.value_or(y) == 3);
                }
                {
                    constexpr optional<X> opt;
                    static_assert(opt.value_or(Y(3)) == 4);
                }
                {
                    const optional<X> opt(2);
                    const Y y(3);
                    ret_access[0] &= (opt.value_or(y) == 2);
                }
                {
                    const optional<X> opt(2);
                    ret_access[0] &= (opt.value_or(Y(3)) == 2);
                }
                {
                    const optional<X> opt;
                    const Y y(3);
                    ret_access[0] &= (opt.value_or(y) == 3);
                }
                {
                    const optional<X> opt;
                    ret_access[0] &= (opt.value_or(Y(3)) == 4);
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
    EXPECT_TRUE(ret, "Wrong result of const dpl::optional::value_or in kernel_test");

    return TestUtils::done();
}
