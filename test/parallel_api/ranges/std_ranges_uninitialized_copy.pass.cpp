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

#include "support/test_config.h"
#include "support/test_macros.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING
#include <ranges>

#include "std_ranges_memory_test.h"

namespace test_std_ranges
{
template<>
constexpr int test_mode_id<std::remove_cvref_t<decltype(oneapi::dpl::ranges::uninitialized_copy)>> = 1;
}
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto uninitialized_copy_move_checker =
        [](const auto& res, auto&& r_in, auto&& r_out) {
            using InRange = std::remove_cvref_t<decltype(r_in)>;
            using OutRange = std::remove_cvref_t<decltype(r_out)>;

            using Size = std::common_type_t<std::ranges::range_size_t<InRange>, std::ranges::range_size_t<OutRange>>;
            const Size sz = std::ranges::min((Size)std::ranges::size(r_in), (Size)std::ranges::size(r_out));

            const bool bres1 = (res.in == std::ranges::borrowed_iterator_t<InRange>(std::ranges::begin(r_in) + sz)
                && res.out == std::ranges::borrowed_iterator_t<OutRange>(std::ranges::begin(r_out) + sz));

            const bool bres2 = std::ranges::all_of(r_out, [](const auto& v) { return v.val1 == -1;})
                && std::ranges::equal(std::ranges::take_view(r_in, sz), std::ranges::take_view(r_out, sz),
                       [](const auto& v1, const auto& v2) { return v1.val2 == v2.val2;})
                && std::ranges::all_of(std::ranges::drop_view(r_out, sz), [](const auto& v) { return v.val2 == -1;});

            return std::pair<bool, bool>{bres1, bres2};
        };

    test_memory_algo<Elem, -1>{}.run(dpl_ranges::uninitialized_copy, uninitialized_copy_move_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
