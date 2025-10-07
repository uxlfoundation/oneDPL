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
#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"

class KernelTupleSizeTest;

template <class KernelName, class T, dpl::size_t N>
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> numOfItems{1};
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelName>([=]() {
            static_assert(dpl::tuple_size<T>::value == N);
            static_assert(dpl::tuple_size<const T>::value == N);
            static_assert(dpl::tuple_size<volatile T>::value == N);
            static_assert(dpl::tuple_size<const volatile T>::value == N);

            static_assert(std::tuple_size_v<T> == N);
            static_assert(std::tuple_size_v<const T> == N);
            static_assert(std::tuple_size_v<volatile T> == N);
            static_assert(std::tuple_size_v<const volatile T> == N);
        });
    });
}

class KernelName1;
class KernelName2;
class KernelName3;

int
main()
{
    kernel_test<KernelName1, std::tuple<>, 0>();
    kernel_test<KernelName2, std::tuple<int>, 1>();
    kernel_test<KernelName3, std::tuple<int, int, int>, 3>();

    return TestUtils::done();
}
