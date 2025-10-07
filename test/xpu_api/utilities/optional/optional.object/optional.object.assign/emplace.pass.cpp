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

class X
{
    int i_;
    int j_ = 0;

  public:
    X() : i_(0) {}
    X(int i) : i_(i) {}
    X(int i, int j) : i_(i), j_(j) {}

    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
    }
};

template <class T>
bool
test_one_arg()
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<T>([=]() {
                using Opt = dpl::optional<T>;
                {
                    Opt opt;
                    auto& v = opt.emplace();
                    static_assert(dpl::is_same_v<T&, decltype(v)>);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(0));
                    ret_access[0] &= (&v == &*opt);
                }
                {
                    Opt opt;
                    auto& v = opt.emplace(1);
                    static_assert(dpl::is_same_v<T&, decltype(v)>);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(1));
                    ret_access[0] &= (&v == &*opt);
                }
                {
                    Opt opt(2);
                    auto& v = opt.emplace();
                    static_assert(dpl::is_same_v<T&, decltype(v)>);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(0));
                    ret_access[0] &= (&v == &*opt);
                }
                {
                    Opt opt(2);
                    auto& v = opt.emplace(1);
                    static_assert(dpl::is_same_v<T&, decltype(v)>);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(1));
                    ret_access[0] &= (&v == &*opt);
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
                optional<const int> opt;
                auto& v = opt.emplace(42);
                static_assert(dpl::is_same_v<const int&, decltype(v)>);
                ret_access[0] &= (*opt == 42);
                ret_access[0] &= (v == 42);
                opt.emplace();
                ret_access[0] &= (*opt == 0);
            });
        });
    }
    return ret;
}

int
main()
{
    using T = int;
    auto ret = test_one_arg<T>();
    ret &= test_one_arg<const T>();
    ret &= kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::optional::emplace check");

    return TestUtils::done();
}
