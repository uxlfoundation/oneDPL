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

template <class T>
void
test_is_trivially_constructible()
{
    static_assert(dpl::is_trivially_constructible<T>::value);
    static_assert(dpl::is_trivially_constructible_v<T>);
}

template <class T, class A0>
void
test_is_trivially_constructible()
{
    static_assert(dpl::is_trivially_constructible<T, A0>::value);
    static_assert(dpl::is_trivially_constructible_v<T, A0>);
}

template <class T>
void
test_is_not_trivially_constructible()
{
    static_assert(!dpl::is_trivially_constructible<T>::value);
    static_assert(!dpl::is_trivially_constructible_v<T>);
}

template <class T, class A0>
void
test_is_not_trivially_constructible()
{
    static_assert(!dpl::is_trivially_constructible<T, A0>::value);
    static_assert(!dpl::is_trivially_constructible_v<T, A0>);
}

template <class T, class A0, class A1>
void
test_is_not_trivially_constructible()
{
    static_assert(!dpl::is_trivially_constructible<T, A0, A1>::value);
    static_assert(!dpl::is_trivially_constructible_v<T, A0, A1>);
}

struct A
{
    explicit A(int);
    A(int, float);
};

bool
kernel_test()
{
    test_is_trivially_constructible<int>();
    test_is_trivially_constructible<int, const int&>();

    test_is_not_trivially_constructible<A, int>();
    test_is_not_trivially_constructible<A, int, float>();
    test_is_not_trivially_constructible<A>();
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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_trivially_constructible check");

    return TestUtils::done();
}
