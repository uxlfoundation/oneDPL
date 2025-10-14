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
    constexpr Y(int i, int j) : i_(i), j_(j) {}

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
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
                    constexpr optional<int> opt(in_place, 5);
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == 5);

                    struct test_constexpr_ctor : public optional<int>
                    {
                        constexpr test_constexpr_ctor(in_place_t, int i) : optional<int>(in_place, i) {}
                    };
                }
                {
                    optional<const int> opt(in_place, 5);
                    ret_access[0] &= (*opt == 5);
                }
                {
                    const optional<X> opt(in_place);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == X());
                }
                {
                    const optional<X> opt(in_place, 5);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == X(5));
                }
                {
                    const optional<X> opt(in_place, 5, 4);
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == X(5, 4));
                }
                {
                    constexpr optional<Y> opt(in_place);
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == Y());

                    struct test_constexpr_ctor : public optional<Y>
                    {
                        constexpr test_constexpr_ctor(in_place_t) : optional<Y>(in_place) {}
                    };
                }
                {
                    constexpr optional<Y> opt(in_place, 5);
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == Y(5));

                    struct test_constexpr_ctor : public optional<Y>
                    {
                        constexpr test_constexpr_ctor(in_place_t, int i) : optional<Y>(in_place, i) {}
                    };
                }
                {
                    constexpr optional<Y> opt(in_place, 5, 4);
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == Y(5, 4));

                    struct test_constexpr_ctor : public optional<Y>
                    {
                        constexpr test_constexpr_ctor(in_place_t, int i, int j) : optional<Y>(in_place, i, j) {}
                    };
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
    EXPECT_TRUE(ret, "Wrong result of dpl::in_place check");

    return TestUtils::done();
}
