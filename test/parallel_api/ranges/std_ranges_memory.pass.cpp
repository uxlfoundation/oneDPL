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

#include "std_ranges_memory_test.h"

//defualt initialization, initialization by custom value (fill)
struct Elem
{
    int val1;
    int val2;

    Elem() { val1 = 1; }
    Elem(int v) { val2 = v;}
};

//value initialization
struct Elem_0
{
    int val1; //value initialization
    int val2;

    Elem_0(): val1() {}
};

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto uninitialized_default_construct_checker = 
        [](const auto& res, const auto& r) { 
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == 1;})
                && std::ranges::all_of(r, [](const auto& v) { return v.val2 == -1;});  //-1 means no initialization.

            return std::pair<bool, bool>{bres1, bres2};
        };
 
    test_memory_algo<Elem, -1>{}.run_host(dpl_ranges::uninitialized_default_construct, uninitialized_default_construct_checker);

    auto uninitialized_value_construct_checker = 
        [](const auto& res, const auto& r) {
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == 0;})
                && std::ranges::all_of(r, [](const auto& v) { return v.val2 == -1;}); //-1 means no initialization.

            return std::pair<bool, bool>{bres1, bres2};
        };
 
    test_memory_algo<Elem_0, -1>{}.run_host(dpl_ranges::uninitialized_value_construct, uninitialized_value_construct_checker);

    auto uninitialized_fill_checker = 
        [](const auto& res, const auto& r, const auto& value) {
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == -1;})//-1 means no initialization.
                && std::ranges::all_of(r, [value](const auto& v) { return v.val2 == value;});

            return std::pair<bool, bool>{bres1, bres2};
        };
 
    test_memory_algo<Elem, -1>{}.run_host(dpl_ranges::uninitialized_fill, uninitialized_fill_checker, 2);

#if TEST_DPCPP_BACKEND_PRESENT
    test_memory_algo<Elem, -1>{}.run_device(dpl_ranges::uninitialized_default_construct, uninitialized_default_construct_checker);
    test_memory_algo<Elem_0, -1>{}.run_device(dpl_ranges::uninitialized_value_construct, uninitialized_value_construct_checker);
    test_memory_algo<Elem, -1>{}.run_device(dpl_ranges::uninitialized_fill, uninitialized_fill_checker, 2);
#endif //TEST_DPCPP_BACKEND_PRESENT

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
