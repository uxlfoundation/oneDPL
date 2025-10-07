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

#include "assignable_from.h"

using dpl::optional;

struct Y1
{
    Y1() = default;
    Y1(const int&) {}
    Y1&
    operator=(const Y1&) = delete;
};

struct Y2
{
    Y2() = default;
    Y2(const int&) = delete;
    Y2&
    operator=(const int&)
    {
        return *this;
    }
};

class B
{
};
class D : public B
{
};

bool
test_ambigious_assign()
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                using OptInt = dpl::optional<int>;
                {
                    using T = AssignableFrom<OptInt&&>;
                    {
                        OptInt a(42);
                        dpl::optional<T> t;
                        t = dpl::move(a);
                        ret_access[0] &= (t->type_constructed == 1);
                        ret_access[0] &= (t->type_assigned == 0);
                        ret_access[0] &= (t->int_constructed == 0);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        using Opt = dpl::optional<T>;
                        static_assert(!dpl::is_assignable<Opt&, const OptInt&&>::value);
                        static_assert(!dpl::is_assignable<Opt&, const OptInt&>::value);
                        static_assert(!dpl::is_assignable<Opt&, OptInt&>::value);
                    }
                }
                {
                    using T = AssignableFrom<OptInt const&&>;
                    {
                        const OptInt a(42);
                        dpl::optional<T> t;
                        t = dpl::move(a);
                        ret_access[0] &= (t->type_constructed == 1);
                        ret_access[0] &= (t->type_assigned == 0);
                        ret_access[0] &= (t->int_constructed == 0);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        OptInt a(42);
                        dpl::optional<T> t;
                        t = dpl::move(a);
                        ret_access[0] &= (t->type_constructed == 1);
                        ret_access[0] &= (t->type_assigned == 0);
                        ret_access[0] &= (t->int_constructed == 0);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        using Opt = dpl::optional<T>;
                        static_assert(dpl::is_assignable<Opt&, OptInt&&>::value);
                        static_assert(!dpl::is_assignable<Opt&, const OptInt&>::value);
                        static_assert(!dpl::is_assignable<Opt&, OptInt&>::value);
                    }
                }
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
            cgh.single_task<class KernelTest2>([=]() {
                {
                    optional<int> opt;
                    optional<short> opt2;
                    opt = dpl::move(opt2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt;
                    optional<short> opt2(short{2});
                    opt = dpl::move(opt2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
                }
                {
                    optional<int> opt(3);
                    optional<short> opt2;
                    opt = dpl::move(opt2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt(3);
                    optional<short> opt2(short{2});
                    opt = dpl::move(opt2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
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
    auto ret = test_ambigious_assign();
    ret &= kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::iptional assign check");

    return TestUtils::done();
}
