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
#include "support/utils_invoke.h"

#if TEST_STD_VER < 20

template <class T>
void
test_is_pod(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(dpl::is_pod<T>::value);
            static_assert(dpl::is_pod<const T>::value);
            static_assert(dpl::is_pod<volatile T>::value);
            static_assert(dpl::is_pod<const volatile T>::value);
            static_assert(dpl::is_pod_v<T>);
            static_assert(dpl::is_pod_v<const T>);
            static_assert(dpl::is_pod_v<volatile T>);
            static_assert(dpl::is_pod_v<const volatile T>);
        });
    });
}

template <class T>
void
test_is_not_pod(sycl::queue deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_pod<T>::value);
            static_assert(!dpl::is_pod<const T>::value);
            static_assert(!dpl::is_pod<volatile T>::value);
            static_assert(!dpl::is_pod<const volatile T>::value);
            static_assert(!dpl::is_pod_v<T>);
            static_assert(!dpl::is_pod_v<const T>);
            static_assert(!dpl::is_pod_v<volatile T>);
            static_assert(!dpl::is_pod_v<const volatile T>);
        });
    });
}

struct Class
{
    ~Class();
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_pod<void>(deviceQueue);
    test_is_not_pod<int&>(deviceQueue);
    test_is_not_pod<Class>(deviceQueue);

    test_is_pod<int>(deviceQueue);
    test_is_pod<int*>(deviceQueue);
    test_is_pod<const int*>(deviceQueue);
    test_is_pod<char[3]>(deviceQueue);
    test_is_pod<char[]>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_pod<double>(deviceQueue);
    }
}
#endif

int
main()
{
    bool __processed = false;
#if TEST_STD_VER < 20
    kernel_test();
    __processed = true;
#endif

    return TestUtils::done(__processed);
}
