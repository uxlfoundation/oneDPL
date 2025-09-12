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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/memory>
#include <oneapi/dpl/ranges>

#include "support/test_config.h"
#include "support/test_macros.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING
#include "std_ranges_memory_test.h"
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto uninitialized_value_construct_checker =
        [](const auto& res, const auto& r) {
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == 0;})
                && std::ranges::all_of(r, [](const auto& v) { return v.val2 == -1;}); // -1 means no initialization

            return std::pair<bool, bool>{bres1, bres2};
        };

    test_memory_algo<Elem_0, -1>{}.run(dpl_ranges::uninitialized_value_construct, uninitialized_value_construct_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
