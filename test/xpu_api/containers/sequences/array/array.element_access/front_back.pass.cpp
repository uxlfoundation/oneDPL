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

// <array>

// reference front();       // constexpr in C++17
// reference back();        // constexpr in C++17
// const_reference front(); // constexpr in C++14
// const_reference back();  // constexpr in C++14

#include "support/test_config.h"

#include <oneapi/dpl/array>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

bool
kernel_test()
{
    sycl::queue myQueue = TestUtils::get_test_queue();
    auto ret = true;
    {
        sycl::buffer<bool, 1> buf1(&ret, sycl::range<1>(1));
        myQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buf1.get_access<sycl::access::mode::read_write>(cgh);

            cgh.single_task<class KernelFrontBackTest>([=]() {
                typedef int T;
                typedef dpl::array<T, 3> C;
                {
                    C c = {1, 2, 35};

                    C::reference r1 = c.front();
                    ret_access[0] &= (r1 == 1);
                    r1 = 55;
                    ret_access[0] &= (c[0] == 55);

                    C::reference r2 = c.back();
                    ret_access[0] &= (r2 == 35);
                    r2 = 75;
                    ret_access[0] &= (c[2] == 75);
                }
                {
                    const C c = {1, 2, 35};
                    C::const_reference r1 = c.front();
                    ret_access[0] &= (r1 == 1);

                    C::const_reference r2 = c.back();
                    ret_access[0] &= (r2 == 35);
                }
                {

                    C c = {};
                    C const& cc = c;
                    ret_access[0] &= (dpl::is_same<decltype(c.back()), typename C::reference>::value == true);
                    ret_access[0] &= (dpl::is_same<decltype(cc.back()), typename C::const_reference>::value == true);
                    (void) noexcept(c.back());
                    (void) noexcept(cc.back());
                    ret_access[0] &= (dpl::is_same<decltype(c.front()), typename C::reference>::value == true);
                    ret_access[0] &= (dpl::is_same<decltype(cc.front()), typename C::const_reference>::value == true);
                    (void) noexcept(c.back());
                    (void) noexcept(cc.back());
                }
                {
                    constexpr C c = {1, 2, 35};
                    constexpr T t1 = c.front();
                    ret_access[0] &= (t1 == 1);
                    constexpr T t2 = c.back();
                    ret_access[0] &= (t2 == 35);
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
    EXPECT_TRUE(ret, "Wrong result of work with dpl::array::front/dpl::array::back in kernel_test()");

    return TestUtils::done();
}
