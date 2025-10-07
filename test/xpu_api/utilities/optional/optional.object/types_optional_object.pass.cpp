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

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

using dpl::optional;

template <class KernelTest, class Opt, class T>
void
test()
{
    sycl::queue q = TestUtils::get_test_queue();
    {

        q.submit([&](sycl::handler& cgh) {
            cgh.single_task<KernelTest>([=]() { static_assert(dpl::is_same<typename Opt::value_type, T>::value); });
        });
    }
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test<KernelTest1, optional<int>, int>();
    test<KernelTest2, optional<const int>, const int>();
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test<KernelTest3, optional<double>, double>();
        test<KernelTest4, optional<const double>, const double>();
    }

    return TestUtils::done();
}
