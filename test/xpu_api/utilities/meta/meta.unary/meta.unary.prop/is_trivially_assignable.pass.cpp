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

#include "support/test_macros.h"
#include "support/utils.h"

template <class T, class U>
void
test_is_trivially_assignable()
{
    static_assert(dpl::is_trivially_assignable<T, U>::value);
    static_assert(dpl::is_trivially_assignable_v<T, U>);
}

template <class T, class U>
void
test_is_not_trivially_assignable()
{
    static_assert(!dpl::is_trivially_assignable<T, U>::value);
    static_assert(!dpl::is_trivially_assignable_v<T, U>);
}

struct A
{
};

struct B
{
    void operator=(A);
};

struct C
{
    void
    operator=(C&); // not const
};

bool
kernel_test()
{
    test_is_trivially_assignable<int&, int&>();
    test_is_trivially_assignable<int&, int>();
    test_is_trivially_assignable<int&, float>();

    test_is_not_trivially_assignable<int, int&>();
    test_is_not_trivially_assignable<int, int>();
    test_is_not_trivially_assignable<B, A>();
    test_is_not_trivially_assignable<A, B>();
    test_is_not_trivially_assignable<C&, C&>();
    return true;
}

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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_trivially_assignable check");

    return TestUtils::done();
}
