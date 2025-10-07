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

template <class KernelTest, class T, unsigned A>
void
test_extent(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::extent<T>::value == A);
            static_assert(dpl::extent<const T>::value == A);
            static_assert(dpl::extent<volatile T>::value == A);
            static_assert(dpl::extent<const volatile T>::value == A);

            static_assert(dpl::extent_v<T> == A);
            static_assert(dpl::extent_v<const T> == A);
            static_assert(dpl::extent_v<volatile T> == A);
            static_assert(dpl::extent_v<const volatile T> == A);
        });
    });
}

template <class KernelTest, class T, unsigned A>
void
test_extent1(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::extent<T, 1>::value == A);
            static_assert(dpl::extent<const T, 1>::value == A);
            static_assert(dpl::extent<volatile T, 1>::value == A);
            static_assert(dpl::extent<const volatile T, 1>::value == A);

            static_assert(dpl::extent_v<T, 1> == A);
            static_assert(dpl::extent_v<const T, 1> == A);
            static_assert(dpl::extent_v<volatile T, 1> == A);
            static_assert(dpl::extent_v<const volatile T, 1> == A);
        });
    });
}

struct Class
{
    ~Class();
};

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
class KernelTest13;
class KernelTest14;
class KernelTest15;
class KernelTest16;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_extent<KernelTest1, void, 0>(deviceQueue);
    test_extent<KernelTest2, int&, 0>(deviceQueue);
    test_extent<KernelTest3, Class, 0>(deviceQueue);
    test_extent<KernelTest4, int*, 0>(deviceQueue);
    test_extent<KernelTest5, const int*, 0>(deviceQueue);
    test_extent<KernelTest6, int, 0>(deviceQueue);
    test_extent<KernelTest7, bool, 0>(deviceQueue);
    test_extent<KernelTest8, unsigned, 0>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_extent<KernelTest9, double, 0>(deviceQueue);
    }

    test_extent<KernelTest10, int[2], 2>(deviceQueue);
    test_extent<KernelTest11, int[2][4], 2>(deviceQueue);
    test_extent<KernelTest12, int[][4], 0>(deviceQueue);

    test_extent1<KernelTest13, int, 0>(deviceQueue);
    test_extent1<KernelTest14, int[2], 0>(deviceQueue);
    test_extent1<KernelTest15, int[2][4], 4>(deviceQueue);
    test_extent1<KernelTest16, int[][4], 4>(deviceQueue);
}

int
main()
{
    kernel_test();

    return 0;
}
