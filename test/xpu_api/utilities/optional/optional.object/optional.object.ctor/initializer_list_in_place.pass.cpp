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
#include <oneapi/dpl/array>

#include "support/test_macros.h"
#include "support/utils.h"

using dpl::in_place;
using dpl::in_place_t;
using dpl::optional;

// TODO required to unify
class X
{
    int i_;
    int j_ = 0;

  public:
    X() : i_(0) {}
    X(int i) : i_(i) {}
    X(int i, int j) : i_(i), j_(j) {}

    ~X() {}

    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
    }
};

// TODO required to unify
class Y
{
    int i_;
    int j_ = 0;

  public:
    constexpr Y() : i_(0) {}
    constexpr Y(int i) : i_(i) {}
    constexpr Y(std::initializer_list<int> il) : i_(il.begin()[0]), j_(il.begin()[1]) {}

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
    }
};

void
kernel_test()
{
    sycl::queue q = TestUtils::get_test_queue();
    {
        q.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                {
                    static_assert(!dpl::is_constructible<X, std::initializer_list<int>&>::value);
                    static_assert(!dpl::is_constructible<optional<X>, std::initializer_list<int>&>::value);
                }
                {
                    static_assert(dpl::is_constructible<optional<Y>, std::initializer_list<int>&>::value);
                    constexpr optional<Y> opt(in_place, {3, 1});
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == Y{3, 1});

                    struct test_constexpr_ctor : public optional<Y>
                    {
                        constexpr test_constexpr_ctor(in_place_t, std::initializer_list<int> i)
                            : optional<Y>(in_place, i)
                        {
                        }
                    };
                }
            });
        });
    }
}

int
main()
{
    kernel_test();
    return TestUtils::done();
}
