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

// constexpr auto operator[](difference_type n) const requires
//        all_random_access<Const, Views...>

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <ranges>
#include <cassert>

#include "../types.h"

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // random_access_range
    dpl_ranges::zip_view v(SizedRandomAccessView{buffer}, std::views::iota(0));
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

#ifdef _LIBCPP_VERSION // libc++ doesn't implement P2165R4 yet
    static_assert(std::is_same_v<decltype(it[2]), std::pair<int&, int>>);
#else
    static_assert(std::is_same_v<decltype(it[2]), tuple_type<int&, int>>);
#endif
  }

  {
    // contiguous_range
    dpl_ranges::zip_view v(ContiguousCommonView{buffer}, ContiguousCommonView{buffer});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

#ifdef _LIBCPP_VERSION // libc++ doesn't implement P2165R4 yet
    static_assert(std::is_same_v<decltype(it[2]), std::pair<int&, int&>>);
#else
    static_assert(std::is_same_v<decltype(it[2]), tuple_type<int&, int&>>);
#endif
  }

  {
    // non random_access_range
    dpl_ranges::zip_view v(BidiCommonView{buffer});
    auto iter = v.begin();
    const auto canSubscript = [](auto&& it) { return requires { it[0]; }; };
    static_assert(!canSubscript(iter));
  }
}
#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
