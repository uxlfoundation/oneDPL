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

void
kernel_test()
{
    sycl::queue q = TestUtils::get_test_queue();
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest>([=]() {
            using dpl::optional;
            {
                constexpr optional<int> opt;
                static_assert(!opt.has_value());
                ASSERT_NOEXCEPT(opt.has_value());
                static_assert(dpl::is_same<decltype(opt.has_value()), bool>::value);
            }
            {
                constexpr optional<int> opt(0);
                static_assert(opt.has_value());
            }
        });
    });
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
