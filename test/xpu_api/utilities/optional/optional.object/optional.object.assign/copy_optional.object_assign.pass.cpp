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

template <class Tp>
bool
assign_empty()
{
    optional<Tp> opt{42};

    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<optional<Tp>, 1> buffer2(&opt, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto lhs_access = buffer2.template get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                const optional<Tp> rhs;
                lhs_access[0] = rhs;
                ret_access[0] &= (!lhs_access[0].has_value() && !rhs.has_value());
            });
        });
    }
    return ret;
}

template <class Tp>
bool
assign_value()
{
    optional<Tp> opt{42};

    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<optional<Tp>, 1> buffer2(&opt, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto lhs_access = buffer2.template get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                const optional<Tp> rhs(100);
                lhs_access[0] = rhs;
                ret_access[0] &= (lhs_access[0].has_value() && rhs.has_value() && *lhs_access[0] == *rhs);
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = assign_empty<int>();
    ret &= assign_value<int>();
    EXPECT_TRUE(ret, "Wrong result of dpl::optional copy check");

    return TestUtils::done();
}
