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

template <class T>
void
test_function_imp(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_reference<T>::value);
            static_assert(!dpl::is_arithmetic<T>::value);
            static_assert(!dpl::is_fundamental<T>::value);
            static_assert(!dpl::is_object<T>::value);
            static_assert(!dpl::is_scalar<T>::value);
            static_assert(dpl::is_compound<T>::value);
            static_assert(!dpl::is_member_pointer<T>::value);
        });
    });
}

template <class T>
void
test_function(sycl::queue& deviceQueue)
{
    test_function_imp<T>(deviceQueue);
    test_function_imp<const T>(deviceQueue);
    test_function_imp<volatile T>(deviceQueue);
    test_function_imp<const volatile T>(deviceQueue);
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_function<int(double)>(deviceQueue);
        test_function<int(double, char)>(deviceQueue);
    }
    test_function<void()>(deviceQueue);
    test_function<void(int)>(deviceQueue);
}

int
main()
{
    kernel_test();

    return 0;
}
