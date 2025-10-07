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
test_is_trivial()
{
    static_assert(dpl::is_trivial<T>::value);
    static_assert(dpl::is_trivial<const T>::value);
    static_assert(dpl::is_trivial<volatile T>::value);
    static_assert(dpl::is_trivial<const volatile T>::value);
    static_assert(dpl::is_trivial_v<T>);
    static_assert(dpl::is_trivial_v<const T>);
    static_assert(dpl::is_trivial_v<volatile T>);
    static_assert(dpl::is_trivial_v<const volatile T>);
}

template <class T>
void
test_is_not_trivial()
{
    static_assert(!dpl::is_trivial<T>::value);
    static_assert(!dpl::is_trivial<const T>::value);
    static_assert(!dpl::is_trivial<volatile T>::value);
    static_assert(!dpl::is_trivial<const volatile T>::value);
    static_assert(!dpl::is_trivial_v<T>);
    static_assert(!dpl::is_trivial_v<const T>);
    static_assert(!dpl::is_trivial_v<volatile T>);
    static_assert(!dpl::is_trivial_v<const volatile T>);
}

struct A
{
};

struct B
{
    B();
};

bool
kernel_test()
{
    test_is_trivial<int>();
    test_is_trivial<A>();

    test_is_not_trivial<int&>();
    test_is_not_trivial<volatile int&>();
    test_is_not_trivial<B>();
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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_trivial check");

    return TestUtils::done();
}
