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

class X
{
    int i_;

  public:
    X(int i) : i_(i) {}
    X(X&& x) = default;
    X&
    operator=(X&&) = default;
    ~X() {}

    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
    }
};

class Y
{
    int i_;

  public:
    int nsf = 0;

    Y(int i) : i_(i) {}
    Y(Y&&) = default;
    ~Y() {}

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_;
    }
    friend void
    swap(Y& x, Y& y)
    {
        dpl::swap(x.i_, y.i_);
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
                    optional<int> opt1;
                    optional<int> opt2;
                    ASSERT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<int> opt1(1);
                    optional<int> opt2;
                    ASSERT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<int> opt1;
                    optional<int> opt2(2);
                    ASSERT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<int> opt1(1);
                    optional<int> opt2(2);
                    ASSERT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<X> opt1;
                    optional<X> opt2;
                    ASSERT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<X> opt1(1);
                    optional<X> opt2;
                    ASSERT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<X> opt1;
                    optional<X> opt2(2);
                    ASSERT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<X> opt1(1);
                    optional<X> opt2(2);
                    ASSERT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<Y> opt1;
                    optional<Y> opt2;
                    ASSERT_NOT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<Y> opt1(1);
                    optional<Y> opt2;
                    ASSERT_NOT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<Y> opt1;
                    optional<Y> opt2(2);
                    ASSERT_NOT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<Y> opt1(1);
                    optional<Y> opt2(2);

                    (*opt1).nsf = 100;
                    (*opt2).nsf = 200;

                    ASSERT_NOT_NOEXCEPT(opt1.swap(opt2));
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);

                    ret_access[0] &= ((*opt1).nsf == 100);
                    ret_access[0] &= ((*opt2).nsf == 200);
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
    EXPECT_TRUE(ret, "Wrong result of dpl::optional::swap in kernel_test");

    return TestUtils::done();
}
