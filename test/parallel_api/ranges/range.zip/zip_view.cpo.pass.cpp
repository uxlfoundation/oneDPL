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

// std::views::zip

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <ranges>

#include <array>
#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>

#include "types.h"

#include <oneapi/dpl/ranges>

namespace dpl = oneapi::dpl;

static_assert(std::is_invocable_v<decltype((dpl::views::zip))>);
static_assert(!std::is_invocable_v<decltype((dpl::views::zip)), int>);
static_assert(std::is_invocable_v<decltype((dpl::views::zip)), SizedRandomAccessView>);
static_assert(
    std::is_invocable_v<decltype((dpl::views::zip)), SizedRandomAccessView, std::ranges::iota_view<int, int>>);
static_assert(!std::is_invocable_v<decltype((dpl::views::zip)), SizedRandomAccessView, int>);

void test() {
  {
    // zip zero arguments
    auto v = dpl::views::zip();
    assert(std::ranges::empty(v));
    static_assert(std::is_same_v<decltype(v), std::ranges::empty_view<tuple_type<>>>);
  }

  {
    // zip a view
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::same_as<dpl::ranges::zip_view<SizedRandomAccessView>> decltype(auto) v =
        dpl::views::zip(SizedRandomAccessView{buffer});
    assert(std::ranges::size(v) == 8);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v)>, tuple_type<int&>>);
  }

  {
    // zip a viewable range
    std::array a{1, 2, 3};
    std::same_as<dpl::ranges::zip_view<std::ranges::ref_view<std::array<int, 3>>>> decltype(auto) v =
        dpl::views::zip(a);
    assert(&(std::get<0>(*v.begin())) == &(a[0]));
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v)>, tuple_type<int&>>);
  }

  {
    // zip the zip_view
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::same_as<dpl::ranges::zip_view<SizedRandomAccessView, SizedRandomAccessView>> decltype(auto) v =
        dpl::views::zip(SizedRandomAccessView{buffer}, SizedRandomAccessView{buffer});

    std::same_as<
        dpl::ranges::zip_view<dpl::ranges::zip_view<SizedRandomAccessView, SizedRandomAccessView>>> decltype(auto) v2 =
        dpl::views::zip(v);

#ifdef _LIBCPP_VERSION // libc++ doesn't implement P2165R4 yet
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v2)>, tuple_type<std::pair<int&, int&>>>);
#else
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v2)>, tuple_type<tuple_type<int&, int&>>>);
#endif
  }
}

#endif //_ENABLE_STD_RANGES_TESTING
int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
