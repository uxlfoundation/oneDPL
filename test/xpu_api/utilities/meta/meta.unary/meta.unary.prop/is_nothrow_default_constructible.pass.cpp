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
test_is_nothrow_default_constructible()
{
    static_assert(dpl::is_nothrow_default_constructible<T>::value);
    static_assert(dpl::is_nothrow_default_constructible<const T>::value);
    static_assert(dpl::is_nothrow_default_constructible<volatile T>::value);
    static_assert(dpl::is_nothrow_default_constructible<const volatile T>::value);
    static_assert(dpl::is_nothrow_default_constructible_v<T>);
    static_assert(dpl::is_nothrow_default_constructible_v<const T>);
    static_assert(dpl::is_nothrow_default_constructible_v<volatile T>);
    static_assert(dpl::is_nothrow_default_constructible_v<const volatile T>);
}

template <class T>
void
test_has_not_nothrow_default_constructor()
{
    static_assert(!dpl::is_nothrow_default_constructible<T>::value);
    static_assert(!dpl::is_nothrow_default_constructible<const T>::value);
    static_assert(!dpl::is_nothrow_default_constructible<volatile T>::value);
    static_assert(!dpl::is_nothrow_default_constructible<const volatile T>::value);
    static_assert(!dpl::is_nothrow_default_constructible_v<T>);
    static_assert(!dpl::is_nothrow_default_constructible_v<const T>);
    static_assert(!dpl::is_nothrow_default_constructible_v<volatile T>);
    static_assert(!dpl::is_nothrow_default_constructible_v<const volatile T>);
}

class Empty
{
};

union Union
{
};

struct bit_zero
{
    int : 0;
};

struct A
{
    A();
};

struct ANT
{
    ANT() noexcept;
};

bool
kernel_test()
{
    test_has_not_nothrow_default_constructor<void>();
    test_has_not_nothrow_default_constructor<int&>();
    test_has_not_nothrow_default_constructor<A>();

    test_is_nothrow_default_constructible<ANT>();
    test_is_nothrow_default_constructible<Union>();
    test_is_nothrow_default_constructible<Empty>();
    test_is_nothrow_default_constructible<int>();
    test_is_nothrow_default_constructible<float>();
    test_is_nothrow_default_constructible<int*>();
    test_is_nothrow_default_constructible<const int*>();
    test_is_nothrow_default_constructible<char[3]>();
    test_is_nothrow_default_constructible<bit_zero>();
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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_nothrow_default_constructible check");

    return TestUtils::done();
}
