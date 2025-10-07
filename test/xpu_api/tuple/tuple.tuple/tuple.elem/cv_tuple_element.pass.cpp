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

#include <oneapi/dpl/tuple>
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
                using tuple_type = dpl::tuple<float, short, int>;
                static_assert(dpl::is_same_v<std::tuple_element<0, const tuple_type>::type, const float>);
                static_assert(dpl::is_same_v<std::tuple_element<1, volatile tuple_type>::type, volatile short>);
                static_assert(
                    dpl::is_same_v<std::tuple_element<2, const volatile tuple_type>::type, const volatile int>);

                static_assert(dpl::is_same_v<std::tuple_element_t<0, const tuple_type>, const float>);
                static_assert(dpl::is_same_v<std::tuple_element_t<1, volatile tuple_type>, volatile short>);
                static_assert(dpl::is_same_v<std::tuple_element_t<2, const volatile tuple_type>, const volatile int>);
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
                using tuple_type = dpl::tuple<double, void, int>;
                static_assert(dpl::is_same_v<std::tuple_element<0, const tuple_type>::type, const double>);
                static_assert(dpl::is_same_v<std::tuple_element_t<0, const tuple_type>, const double>);
            }
        });
    });
}

int
main()
{

    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test2(deviceQueue);
    }

    return TestUtils::done();
}
