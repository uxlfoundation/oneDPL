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
                optional<Tp> rhs;
                lhs_access[0] = dpl::move(rhs);
                ret_access[0] &= !lhs_access[0].has_value() && !rhs.has_value();
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
                optional<Tp> rhs(101);
                lhs_access[0] = dpl::move(rhs);
                ret_access[0] &= lhs_access[0].has_value() && rhs.has_value() && *lhs_access[0] == Tp{101};
            });
        });
    }
    return ret;
}

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
            cgh.single_task<class KernelTest3>([=]() {
                {
                    static_assert(dpl::is_nothrow_move_assignable<optional<int>>::value);
                    optional<int> opt;
                    constexpr optional<int> opt2;
                    opt = dpl::move(opt2);
                    static_assert(static_cast<bool>(opt2) == false);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt;
                    constexpr optional<int> opt2(2);
                    opt = dpl::move(opt2);
                    static_assert(static_cast<bool>(opt2) == true);
                    static_assert(*opt2 == 2);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
                }
                {
                    optional<int> opt(3);
                    constexpr optional<int> opt2;
                    opt = dpl::move(opt2);
                    static_assert(static_cast<bool>(opt2) == false);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt(3);
                    constexpr optional<int> opt2(2);
                    opt = dpl::move(opt2);
                    static_assert(static_cast<bool>(opt2) == true);
                    static_assert(*opt2 == 2);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
                }
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
    ret &= kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::move check");

    return TestUtils::done();
}
