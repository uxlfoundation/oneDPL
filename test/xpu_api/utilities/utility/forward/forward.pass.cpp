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

struct A
{
};

A
source() noexcept
{
    return A();
}
const A
csource() noexcept
{
    return A();
}

constexpr bool
test_constexpr_forward()
{
    int x = 42;
    const int cx = 101;
    return dpl::forward<int&>(x) == 42 && dpl::forward<int>(x) == 42 && dpl::forward<const int&>(x) == 42 &&
           dpl::forward<const int>(x) == 42 && dpl::forward<int&&>(x) == 42 && dpl::forward<const int&&>(x) == 42 &&
           dpl::forward<const int&>(cx) == 101 && dpl::forward<const int>(cx) == 101;
}

void
kernel_test()
{
    [[maybe_unused]] A a;
    [[maybe_unused]] const A ca = A();

    static_assert(dpl::is_same_v<decltype(dpl::forward<A&>(a)), A&>);
    static_assert(dpl::is_same_v<decltype(dpl::forward<A>(a)), A&&>);
    static_assert(dpl::is_same_v<decltype(dpl::forward<A>(source())), A&&>);
    ASSERT_NOEXCEPT(dpl::forward<A&>(a));
    ASSERT_NOEXCEPT(dpl::forward<A>(a));
    ASSERT_NOEXCEPT(dpl::forward<A>(source()));

    static_assert(dpl::is_same_v<decltype(dpl::forward<const A&>(a)), const A&>);
    static_assert(dpl::is_same_v<decltype(dpl::forward<const A>(a)), const A&&>);
    static_assert(dpl::is_same_v<decltype(dpl::forward<const A>(source())), const A&&>);
    ASSERT_NOEXCEPT(dpl::forward<const A&>(a));
    ASSERT_NOEXCEPT(dpl::forward<const A>(a));
    ASSERT_NOEXCEPT(dpl::forward<const A>(source()));

    static_assert(dpl::is_same_v<decltype(dpl::forward<const A&>(ca)), const A&>);
    static_assert(dpl::is_same_v<decltype(dpl::forward<const A>(ca)), const A&&>);
    static_assert(dpl::is_same_v<decltype(dpl::forward<const A>(csource())), const A&&>);
    ASSERT_NOEXCEPT(dpl::forward<const A&>(ca));
    ASSERT_NOEXCEPT(dpl::forward<const A>(ca));
    ASSERT_NOEXCEPT(dpl::forward<const A>(csource()));

    {
        constexpr int i2 = dpl::forward<int>(42);
        static_assert(dpl::forward<int>(42) == 42);
        static_assert(dpl::forward<const int&>(i2) == 42);
        static_assert(test_constexpr_forward());
    }
}

class KernelTest;

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> numOfItems{1};
    {
        deviceQueue.submit([&](sycl::handler& cgh) { cgh.single_task<class KernelTest>([=]() { kernel_test(); }); });
    }

    return TestUtils::done();
}
