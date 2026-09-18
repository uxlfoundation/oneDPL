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
#include "std_ranges_memory_archetypes_test.h"

namespace test_std_ranges
{
template<>
constexpr int test_mode_id<std::remove_cvref_t<decltype(oneapi::dpl::ranges::uninitialized_copy)>> = 1;
template<>
constexpr int test_mode_id<std::remove_cvref_t<decltype(oneapi::dpl::ranges::uninitialized_move)>> = 1;

} //namespace test_std_ranges
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // The single required operation is std::default_initializable. The default constructor is
    // user-provided, so only val1 is written and val2 must keep the no-initialization pattern.
    auto default_construct_checker =
        [](const auto& res, const auto& r) {
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == 1 && v.val2 == -1;});

            return std::pair<bool, bool>{bres1, bres2};
        };

    test_memory_algo<default_construct_archetype, -1, 0>{}.run(dpl_ranges::uninitialized_default_construct, default_construct_checker);

    // The default constructor is defaulted on its first declaration, so value-initialization
    // zero-initializes the whole object, including val2.
    auto value_construct_checker =
        [](const auto& res, const auto& r) {
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == 0 && v.val2 == 0;});

            return std::pair<bool, bool>{bres1, bres2};
        };

    test_memory_algo<value_construct_archetype, -1, 1>{}.run(dpl_ranges::uninitialized_value_construct, value_construct_checker);

    // The filler type differs from the range value type, so the only required operation is
    // std::constructible_from<fill_archetype, const fill_source&>.
    auto fill_checker =
        [](const auto& res, const auto& r, const auto& value) {
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == -1;})
                && std::ranges::all_of(r, [value](const auto& v) { return v.val2 == value.val;});

            return std::pair<bool, bool>{bres1, bres2};
        };

    test_memory_algo<fill_archetype, -1, 2>{}.run(dpl_ranges::uninitialized_fill, fill_checker, fill_source{2});

    // Input and output element types are different, which the requires-clause of uninitialized_copy
    // and uninitialized_move explicitly allows. copy_archetype is constructible only from
    // transfer_source&, move_archetype only from transfer_source&&.
    auto transfer_checker =
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

    test_memory_algo<transfer_source, -1, 3, copy_archetype>{}.run(dpl_ranges::uninitialized_copy, transfer_checker);
    test_memory_algo<transfer_source, -1, 4, move_archetype>{}.run(dpl_ranges::uninitialized_move, transfer_checker);

    // The single required operation is std::destructible.
    auto destroy_checker =
        [](const auto& res, const auto& r) {
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == -1 && v.val2 == 3;});

            return std::pair<bool, bool>{bres1, bres2};
        };

    test_memory_algo<destroy_archetype, -1, 5>{}.run(dpl_ranges::destroy, destroy_checker);

    // The same algorithms over a range which is random access and sized, but neither contiguous nor
    // common.
    run_archetype_view_all_policies<default_construct_archetype, 6>(
        dpl_ranges::uninitialized_default_construct,
        [](const auto& v) { return v.val1 == 1 && v.val2 == -1; }, "uninitialized_default_construct");

    run_archetype_view_all_policies<value_construct_archetype, 7>(
        dpl_ranges::uninitialized_value_construct,
        [](const auto& v) { return v.val1 == 0 && v.val2 == 0; }, "uninitialized_value_construct");

    run_archetype_view_all_policies<destroy_archetype, 8>(
        dpl_ranges::destroy, [](const auto& v) { return v.val1 == -1 && v.val2 == 3; }, "destroy");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
