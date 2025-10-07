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

struct A
{
};

struct B
{
    void operator=(A);
};

template <class KernelTest, class T, class U>
void
test_is_assignable(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_assignable<T, U>::value);
            static_assert(dpl::is_assignable_v<T, U>);
        });
    });
}

template <class KernelTest, class T, class U>
void
test_is_not_assignable(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_assignable<T, U>::value);
            static_assert(!dpl::is_assignable_v<T, U>);
        });
    });
}

struct D;

struct C
{
};

struct E
{
    C
    operator=(int);
};

template <typename T>
struct X
{
    T t;
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

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_assignable<KernelTest1, int&, int&>(deviceQueue);
    test_is_assignable<KernelTest2, int&, int>(deviceQueue);
    test_is_assignable<KernelTest3, B, A>(deviceQueue);
    test_is_assignable<KernelTest4, void*&, void*>(deviceQueue);

    test_is_assignable<KernelTest5, E, int>(deviceQueue);

    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_assignable<KernelTest6, int&, double>(deviceQueue);
    }

    test_is_not_assignable<KernelTest7, int, int&>(deviceQueue);
    test_is_not_assignable<KernelTest8, int, int>(deviceQueue);
    test_is_not_assignable<KernelTest9, A, B>(deviceQueue);
    test_is_not_assignable<KernelTest10, void, const void>(deviceQueue);
    test_is_not_assignable<KernelTest11, const void, const void>(deviceQueue);
    test_is_not_assignable<KernelTest12, int(), int>(deviceQueue);

    //  pointer to incomplete template type
    test_is_assignable<KernelTest13, X<D>*&, X<D>*>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
