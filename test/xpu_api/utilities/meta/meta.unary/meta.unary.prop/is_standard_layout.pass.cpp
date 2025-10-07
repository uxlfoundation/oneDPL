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
test_is_standard_layout()
{
    static_assert(dpl::is_standard_layout<T>::value);
    static_assert(dpl::is_standard_layout<const T>::value);
    static_assert(dpl::is_standard_layout<volatile T>::value);
    static_assert(dpl::is_standard_layout<const volatile T>::value);
    static_assert(dpl::is_standard_layout_v<T>);
    static_assert(dpl::is_standard_layout_v<const T>);
    static_assert(dpl::is_standard_layout_v<volatile T>);
    static_assert(dpl::is_standard_layout_v<const volatile T>);
}

template <class T>
void
test_is_not_standard_layout()
{
    static_assert(!dpl::is_standard_layout<T>::value);
    static_assert(!dpl::is_standard_layout<const T>::value);
    static_assert(!dpl::is_standard_layout<volatile T>::value);
    static_assert(!dpl::is_standard_layout<const volatile T>::value);
    static_assert(!dpl::is_standard_layout_v<T>);
    static_assert(!dpl::is_standard_layout_v<const T>);
    static_assert(!dpl::is_standard_layout_v<volatile T>);
    static_assert(!dpl::is_standard_layout_v<const volatile T>);
}

template <class T1, class T2>
struct test_pair
{
    T1 first;
    T2 second;
};

bool
kernel_test()
{
    test_is_standard_layout<int>();
    test_is_standard_layout<int[3]>();
    test_is_standard_layout<test_pair<int, float>>();

    test_is_not_standard_layout<int&>();
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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_standard_layout check");

    return TestUtils::done();
}
