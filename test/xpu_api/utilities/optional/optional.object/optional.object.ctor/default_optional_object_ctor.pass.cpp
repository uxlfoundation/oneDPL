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

using dpl::optional;

template <class Opt>
void
test_constexpr()
{
    sycl::queue q = TestUtils::get_test_queue();
    {
        q.submit([&](sycl::handler& cgh) {
            cgh.single_task<Opt>([=]() {
                static_assert(dpl::is_nothrow_default_constructible<Opt>::value);
                static_assert(dpl::is_trivially_destructible<Opt>::value);
                static_assert(dpl::is_trivially_destructible<typename Opt::value_type>::value);

                constexpr Opt opt;
                static_assert(static_cast<bool>(opt) == false);
            });
        });
    }
}

int
main()
{
    test_constexpr<optional<int>>();
    test_constexpr<optional<int*>>();

    return TestUtils::done();
}
