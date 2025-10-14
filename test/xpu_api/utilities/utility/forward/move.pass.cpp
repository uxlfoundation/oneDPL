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

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/move_only.h"

MoveOnly
source()
{
    return MoveOnly();
}
const MoveOnly
csource()
{
    return MoveOnly();
}

void test(MoveOnly) {}

struct A
{
    A() = default;
    A(const A&) { ++copy_ctor; }
    A(A&&) { ++move_ctor; }
    A&
    operator=(const A&) = delete;
    int copy_ctor = 0;
    int move_ctor = 0;
};

constexpr bool
test_constexpr_move()
{
    int y = 42;
    const int cy = y;
    return dpl::move(y) == 42 && dpl::move(cy) == 42 && dpl::move(static_cast<int&&>(y)) == 42 &&
           dpl::move(static_cast<int const&&>(y)) == 42;
}

bool
kernel_test()
{

    int x = 42;
    const int& cx = x;
    bool ret = false;
    { // Test return type and noexcept.
        static_assert(dpl::is_same_v<decltype(dpl::move(x)), int&&>);
        ASSERT_NOEXCEPT(dpl::move(x));
        static_assert(dpl::is_same_v<decltype(dpl::move(cx)), const int&&>);
        ASSERT_NOEXCEPT(dpl::move(cx));
        static_assert(dpl::is_same_v<decltype(dpl::move(42)), int&&>);
        ASSERT_NOEXCEPT(dpl::move(42));
    }
    { // test copy and move semantics
        A a;
        const A ca = A();

        ret = (a.copy_ctor == 0);
        ret &= (a.move_ctor == 0);

        A a2 = a;
        ret &= (a2.copy_ctor == 1);
        ret &= (a2.move_ctor == 0);

        A a3 = dpl::move(a);
        ret &= (a3.copy_ctor == 0);
        ret &= (a3.move_ctor == 1);

        A a4 = ca;
        ret &= (a4.copy_ctor == 1);
        ret &= (a4.move_ctor == 0);

        A a5 = dpl::move(ca);
        ret &= (a5.copy_ctor == 1);
        ret &= (a5.move_ctor == 0);
    }
    { // test on a move only type
        MoveOnly mo;
        test(dpl::move(mo));
        test(source());
    }

    {
        constexpr int y = 42;
        static_assert(dpl::move(y) == 42);
        static_assert(test_constexpr_move());
    }

    return ret;
}

class KernelTest;

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of dpl::move check");

    return TestUtils::done();
}
