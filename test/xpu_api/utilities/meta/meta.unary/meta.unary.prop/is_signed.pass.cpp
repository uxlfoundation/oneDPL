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

template <class KernelTest, class T>
void
test_is_signed(sycl::queue deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_signed<T>::value);
            static_assert(dpl::is_signed<const T>::value);
            static_assert(dpl::is_signed<volatile T>::value);
            static_assert(dpl::is_signed<const volatile T>::value);
            static_assert(dpl::is_signed_v<T>);
            static_assert(dpl::is_signed_v<const T>);
            static_assert(dpl::is_signed_v<volatile T>);
            static_assert(dpl::is_signed_v<const volatile T>);
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_signed(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_signed<T>::value);
            static_assert(!dpl::is_signed<const T>::value);
            static_assert(!dpl::is_signed<volatile T>::value);
            static_assert(!dpl::is_signed<const volatile T>::value);
            static_assert(!dpl::is_signed_v<T>);
            static_assert(!dpl::is_signed_v<const T>);
            static_assert(!dpl::is_signed_v<volatile T>);
            static_assert(!dpl::is_signed_v<const volatile T>);
        });
    });
}

struct Class
{
    ~Class();
};

struct A; // incomplete

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;
class KernelTest11;
class KernelTest12;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_signed<KernelTest1, void>(deviceQueue);
    test_is_not_signed<KernelTest2, int&>(deviceQueue);
    test_is_not_signed<KernelTest3, Class>(deviceQueue);
    test_is_not_signed<KernelTest4, int*>(deviceQueue);
    test_is_not_signed<KernelTest5, const int*>(deviceQueue);
    test_is_not_signed<KernelTest6, char[3]>(deviceQueue);
    test_is_not_signed<KernelTest7, char[]>(deviceQueue);
    test_is_not_signed<KernelTest8, bool>(deviceQueue);
    test_is_not_signed<KernelTest9, unsigned>(deviceQueue);
    test_is_not_signed<KernelTest10, A>(deviceQueue);

    test_is_signed<KernelTest11, int>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_signed<KernelTest12, double>(deviceQueue);
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
