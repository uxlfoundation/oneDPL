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

void
kernel_test1(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            {
                typedef dpl::aligned_union<10, char>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<10, char>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 1);
                static_assert(sizeof(T1) == 10);
            }
            {
                typedef dpl::aligned_union<10, short>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<10, short>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 2);
                static_assert(sizeof(T1) == 10);
            }
            {
                typedef dpl::aligned_union<10, int>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<10, int>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 4);
                static_assert(sizeof(T1) == 12);
            }
            {
                typedef dpl::aligned_union<10, short, char>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<10, short, char>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 2);
                static_assert(sizeof(T1) == 10);
            }
            {
                typedef dpl::aligned_union<10, char, short>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<10, char, short>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 2);
                static_assert(sizeof(T1) == 10);
            }
            {
                typedef dpl::aligned_union<2, int, char, short>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<2, int, char, short>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 4);
                static_assert(sizeof(T1) == 4);
            }
            {
                typedef dpl::aligned_union<2, char, int, short>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<2, char, int, short>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 4);
                static_assert(sizeof(T1) == 4);
            }
            {
                typedef dpl::aligned_union<2, char, short, int>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<2, char, short, int>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 4);
                static_assert(sizeof(T1) == 4);
            }
        });
    });
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
            {
                typedef dpl::aligned_union<10, double>::type T1;
                ASSERT_SAME_TYPE(T1, dpl::aligned_union_t<10, double>);
                static_assert(dpl::is_trivial<T1>::value);
                static_assert(dpl::is_standard_layout<T1>::value);
                static_assert(dpl::alignment_of<T1>::value == 8);
                static_assert(sizeof(T1) == 16);
            }
        });
    });
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);

    const auto device = deviceQueue.get_device();
    if (TestUtils::has_type_support<double>(device))
    {
        kernel_test2(deviceQueue);
    }

    return TestUtils::done();
}
