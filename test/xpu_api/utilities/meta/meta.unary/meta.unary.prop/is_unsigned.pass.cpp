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
test_is_unsigned()
{
    static_assert(dpl::is_unsigned<T>::value);
    static_assert(dpl::is_unsigned<const T>::value);
    static_assert(dpl::is_unsigned<volatile T>::value);
    static_assert(dpl::is_unsigned<const volatile T>::value);
    static_assert(dpl::is_unsigned_v<T>);
    static_assert(dpl::is_unsigned_v<const T>);
    static_assert(dpl::is_unsigned_v<volatile T>);
    static_assert(dpl::is_unsigned_v<const volatile T>);
}

template <class T>
void
test_is_not_unsigned()
{
    static_assert(!dpl::is_unsigned<T>::value);
    static_assert(!dpl::is_unsigned<const T>::value);
    static_assert(!dpl::is_unsigned<volatile T>::value);
    static_assert(!dpl::is_unsigned<const volatile T>::value);
    static_assert(!dpl::is_unsigned_v<T>);
    static_assert(!dpl::is_unsigned_v<const T>);
    static_assert(!dpl::is_unsigned_v<volatile T>);
    static_assert(!dpl::is_unsigned_v<const volatile T>);
}

struct Class
{
    ~Class();
};

struct A; // incomplete

bool
kernel_test()
{
    test_is_not_unsigned<void>();
    test_is_not_unsigned<int&>();
    test_is_not_unsigned<Class>();
    test_is_not_unsigned<int*>();
    test_is_not_unsigned<const int*>();
    test_is_not_unsigned<char[3]>();
    test_is_not_unsigned<char[]>();
    test_is_not_unsigned<int>();
    test_is_not_unsigned<float>();
    test_is_not_unsigned<A>();

    test_is_unsigned<bool>();
    test_is_unsigned<unsigned>();

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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_unsigned check");

    return TestUtils::done();
}
