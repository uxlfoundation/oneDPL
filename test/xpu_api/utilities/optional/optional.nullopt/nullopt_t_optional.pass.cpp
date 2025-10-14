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

using dpl::nullopt;
using dpl::nullopt_t;

constexpr bool
test()
{
    nullopt_t foo{nullopt};
    (void)foo;
    return true;
}

void
kernel_test()
{
    sycl::queue q = TestUtils::get_test_queue();
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest>([=]() {
            static_assert(dpl::is_empty_v<nullopt_t>);
            static_assert(!dpl::is_default_constructible_v<nullopt_t>);

            static_assert(dpl::is_same_v<const nullopt_t, decltype(nullopt)>);
            static_assert(test());
        });
    });
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
