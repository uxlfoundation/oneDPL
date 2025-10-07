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

#include <cstdint>

template <class T, unsigned A>
void
test_alignment_of(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            const unsigned AlignofResult = alignof(T);
            static_assert(AlignofResult == A, "Golden value does not match result of alignof keyword");
            static_assert(dpl::alignment_of<T>::value == AlignofResult);
            static_assert(dpl::alignment_of<T>::value == A);
            static_assert(dpl::alignment_of<const T>::value == A);
            static_assert(dpl::alignment_of<volatile T>::value == A);
            static_assert(dpl::alignment_of<const volatile T>::value == A);

            static_assert(dpl::alignment_of_v<T> == A);
            static_assert(dpl::alignment_of_v<const T> == A);
            static_assert(dpl::alignment_of_v<volatile T> == A);
            static_assert(dpl::alignment_of_v<const volatile T> == A);
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
    test_alignment_of<int&, 4>(deviceQueue);
    test_alignment_of<Class, 1>(deviceQueue);
    test_alignment_of<int*, sizeof(intptr_t)>(deviceQueue);
    test_alignment_of<const int*, sizeof(intptr_t)>(deviceQueue);
    test_alignment_of<char[3], 1>(deviceQueue);
    test_alignment_of<int, 4>(deviceQueue);
    test_alignment_of<unsigned, 4>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_alignment_of<double, alignof(double)>(deviceQueue);
    }
}

int
main()
{
    kernel_test();

    return 0;
}
