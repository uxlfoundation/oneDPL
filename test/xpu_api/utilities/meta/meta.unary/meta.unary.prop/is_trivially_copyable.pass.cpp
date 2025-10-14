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
test_is_trivially_copyable()
{
    static_assert(dpl::is_trivially_copyable<T>::value);
    static_assert(dpl::is_trivially_copyable<const T>::value);
    static_assert(dpl::is_trivially_copyable<volatile T>::value);
    static_assert(dpl::is_trivially_copyable<const volatile T>::value);
    static_assert(dpl::is_trivially_copyable_v<T>);
    static_assert(dpl::is_trivially_copyable_v<const T>);
    static_assert(dpl::is_trivially_copyable_v<volatile T>);
    static_assert(dpl::is_trivially_copyable_v<const volatile T>);
}

template <class T>
void
test_is_not_trivially_copyable()
{
    static_assert(!dpl::is_trivially_copyable<T>::value);
    static_assert(!dpl::is_trivially_copyable<const T>::value);
    static_assert(!dpl::is_trivially_copyable<volatile T>::value);
    static_assert(!dpl::is_trivially_copyable<const volatile T>::value);
    static_assert(!dpl::is_trivially_copyable_v<T>);
    static_assert(!dpl::is_trivially_copyable_v<const T>);
    static_assert(!dpl::is_trivially_copyable_v<volatile T>);
    static_assert(!dpl::is_trivially_copyable_v<const volatile T>);
}

struct A
{
    int i_;
};

struct B
{
    int i_;
    ~B() { assert(i_ == 0); }
};

struct C
{
    C();
};

bool
kernel_test()
{
    test_is_trivially_copyable<int>();
    test_is_trivially_copyable<const int>();
    test_is_trivially_copyable<A>();
    test_is_trivially_copyable<const A>();
    test_is_trivially_copyable<C>();

    test_is_not_trivially_copyable<int&>();
    test_is_not_trivially_copyable<const A&>();
    test_is_not_trivially_copyable<B>();
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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_trivially_copyable check");

    return TestUtils::done();
}
