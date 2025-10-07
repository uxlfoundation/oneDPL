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

class KernelTest1;
class KernelTest2;

template <class KernelTest, class T, class... InitArgs>
bool
kernel_test1(InitArgs&&... args)
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    const optional<T> orig(dpl::forward<InitArgs>(args)...);
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<const optional<T>, 1> buffer2(&orig, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto orig_access = buffer2.template get_access<sycl::access::mode::read>(cgh);
            cgh.single_task<KernelTest>([=]() {
                optional<T> rhs(orig_access[0]);
                bool rhs_engaged = static_cast<bool>(rhs);
                optional<T> lhs = dpl::move(rhs);
                if (rhs_engaged)
                    ret_access[0] &= (*lhs == *orig_access[0]);
            });
        });
    }
    return ret;
}

bool
kernel_test2()
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
                    optional<const int> o(42);
                    optional<const int> o2(dpl::move(o));
                    ret_access[0] &= (*o2 == 42);
                }
                {
                    constexpr dpl::optional<int> o1{4};
                    constexpr dpl::optional<int> o2 = dpl::move(o1);
                    static_assert(*o2 == 4);
                }
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test1<KernelTest1, int>();
    ret &= kernel_test1<KernelTest2, int>(3);
    ret &= kernel_test2();
    EXPECT_TRUE(ret, "Wrong result of dpl::optional and dpl::move check");

    return TestUtils::done();
}
