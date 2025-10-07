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
test1(InitArgs&&... args)
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    const optional<T> rhs(dpl::forward<InitArgs>(args)...);
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<optional<T>, 1> buffer2(&rhs, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto rhs_access = buffer2.template get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest>([=]() {
                bool rhs_engaged = static_cast<bool>(rhs_access[0]);
                optional<T> lhs = rhs_access[0];
                ret_access[0] &= (static_cast<bool>(lhs) == rhs_engaged);
                if (rhs_engaged)
                    ret_access[0] &= (*lhs == *rhs_access[0]);
            });
        });
    }
    return ret;
}

bool
test2()
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    const optional<const int> o(42);
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<optional<const int>, 1> buffer2(&o, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto o_access = buffer2.template get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                optional<const int> o2(o_access[0]);
                ret_access[0] &= (*o2 == 42);
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = test1<KernelTest1, int>();
    ret &= test1<KernelTest2, int>(3);
    ret &= test2();
    EXPECT_TRUE(ret, "Wrong result of dpl::optional constructor anc copy check");

    return TestUtils::done();
}
