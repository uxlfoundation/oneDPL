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
test_is_same()
{
    static_assert(dpl::is_same<T, U>::value);
    static_assert(!dpl::is_same<const T, U>::value);
    static_assert(!dpl::is_same<T, const U>::value);
    static_assert(dpl::is_same<const T, const U>::value);
    static_assert(dpl::is_same_v<T, U>);
    static_assert(!dpl::is_same_v<const T, U>);
    static_assert(!dpl::is_same_v<T, const U>);
    static_assert(dpl::is_same_v<const T, const U>);
}

template <class T, class U>
void
test_is_same_ref()
{
    static_assert(dpl::is_same<T, U>::value);
    static_assert(dpl::is_same<const T, U>::value);
    static_assert(dpl::is_same<T, const U>::value);
    static_assert(dpl::is_same<const T, const U>::value);
    static_assert(dpl::is_same_v<T, U>);
    static_assert(dpl::is_same_v<const T, U>);
    static_assert(dpl::is_same_v<T, const U>);
    static_assert(dpl::is_same_v<const T, const U>);
}

template <class T, class U>
void
test_is_not_same()
{
    static_assert(!dpl::is_same<T, U>::value);
}

struct Class
{
    ~Class();
};

bool
kernel_test()
{
    test_is_same<int, int>();
    test_is_same<void, void>();
    test_is_same<Class, Class>();
    test_is_same<int*, int*>();
    test_is_same_ref<int&, int&>();

    test_is_not_same<int, void>();
    test_is_not_same<void, Class>();
    test_is_not_same<Class, int*>();
    test_is_not_same<int*, int&>();
    test_is_not_same<int&, int>();

    return true;
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

    EXPECT_TRUE(ret, "Wrong result of work with dpl::is_same");

    return TestUtils::done();
}
