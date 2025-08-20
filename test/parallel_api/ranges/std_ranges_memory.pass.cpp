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

// A type for testing: default initialization, initialization by custom value, initialization via copy constructor
// It is sufficient to initialize only one field, the other can be used to verify that the raw memory is correctly set
struct Elem
{
    int val1;
    int val2;

    Elem() { val1 = 1; }
    Elem(int v) { val2 = v; }
    Elem(const Elem& elem) { val2 = elem.val2; }
};

// A type for testing: value initialization, initialization via move constructor, destroy
// It is sufficient to initialize only one field, the other can be used to verify that the raw memory is correctly set
struct Elem_0
{
    int val1;
    int val2;

    Elem_0(): val1() {} //val1 has a zero-initialization here
    Elem_0(Elem_0&& elem) { val2 = elem.val2; }
    Elem_0(int v) { val2 = v; }
    ~Elem_0() { val2 = 3;}
};

namespace test_std_ranges
{
template<>
constexpr int test_mode_id<std::remove_cvref_t<decltype(oneapi::dpl::ranges::uninitialized_copy)>> = 1;

template<>
constexpr int test_mode_id<std::remove_cvref_t<decltype(oneapi::dpl::ranges::uninitialized_move)>> = 1;

}
#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
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
    //test_memory_algo<Elem_0, -1>{}.run(dpl_ranges::uninitialized_move, uninitialized_copy_move_checker);

    auto destroy_checker =
        [](const auto& res, const auto& r) {
            using R = std::remove_cvref_t<decltype(r)>;
            bool bres1 = (res == std::ranges::borrowed_iterator_t<R>(std::ranges::begin(r) + std::ranges::size(r)));
            bool bres2 = std::ranges::all_of(r, [](const auto& v) { return v.val1 == -1;}) // -1 means no initialization
                && std::ranges::all_of(r, [](const auto& v) { return v.val2 == 3;});

            return std::pair<bool, bool>{bres1, bres2};
        };

    test_memory_algo<Elem_0, -1>{}.run(dpl_ranges::destroy, destroy_checker);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
