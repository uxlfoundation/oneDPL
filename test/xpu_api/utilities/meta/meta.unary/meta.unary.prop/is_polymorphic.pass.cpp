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

template <class KernelName, class T>
void
test_is_not_polymorphic(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelName>([=]() {
            static_assert(!dpl::is_polymorphic<T>::value);
            static_assert(!dpl::is_polymorphic<const T>::value);
            static_assert(!dpl::is_polymorphic<volatile T>::value);
            static_assert(!dpl::is_polymorphic<const volatile T>::value);
            static_assert(!dpl::is_polymorphic_v<T>);
            static_assert(!dpl::is_polymorphic_v<const T>);
            static_assert(!dpl::is_polymorphic_v<volatile T>);
            static_assert(!dpl::is_polymorphic_v<const volatile T>);
        });
    });
}

template <class KernelName, class T>
void
test_is_polymorphic(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelName>([=]() {
            static_assert(dpl::is_polymorphic<T>::value);
            static_assert(dpl::is_polymorphic<const T>::value);
            static_assert(dpl::is_polymorphic<volatile T>::value);
            static_assert(dpl::is_polymorphic<const volatile T>::value);
            static_assert(dpl::is_polymorphic_v<T>);
            static_assert(dpl::is_polymorphic_v<const T>);
            static_assert(dpl::is_polymorphic_v<volatile T>);
            static_assert(dpl::is_polymorphic_v<const volatile T>);
        });
    });
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

#if !TEST_CLASS_FINAL_BROKEN
class Final final
{
};
#endif // !TEST_CLASS_FINAL_BROKEN

struct Base
{
    virtual ~Base() = default;
};

struct Derived : Base
{
};

class KernelName1;
class KernelName2;
class KernelName3;
class KernelName4;
class KernelName5;
class KernelName6;
class KernelName7;
class KernelName8;
class KernelName9;
class KernelName10;
class KernelName11;
class KernelName12;
class KernelName13;
class KernelName14;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_polymorphic<KernelName1, void>(deviceQueue);
    test_is_not_polymorphic<KernelName2, int&>(deviceQueue);
    test_is_not_polymorphic<KernelName3, int>(deviceQueue);
    test_is_not_polymorphic<KernelName4, int*>(deviceQueue);
    test_is_not_polymorphic<KernelName5, const int*>(deviceQueue);
    test_is_not_polymorphic<KernelName6, char[3]>(deviceQueue);
    test_is_not_polymorphic<KernelName7, char[]>(deviceQueue);
    test_is_not_polymorphic<KernelName8, Union>(deviceQueue);
    test_is_not_polymorphic<KernelName9, Empty>(deviceQueue);
    test_is_not_polymorphic<KernelName10, bit_zero>(deviceQueue);
    test_is_not_polymorphic<KernelName11, Final>(deviceQueue);

    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_not_polymorphic<KernelName12, double>(deviceQueue);
    }

    test_is_polymorphic<KernelName13, Base>(deviceQueue);
    test_is_polymorphic<KernelName14, Derived>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
