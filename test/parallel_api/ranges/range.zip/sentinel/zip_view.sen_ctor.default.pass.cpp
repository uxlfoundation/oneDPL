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

// sentinel() = default;

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <cassert>
#include <ranges>
#include <tuple>

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

struct PODSentinel {
  bool b; // deliberately uninitialised

  friend constexpr bool operator==(int*, const PODSentinel& s) { return s.b; }
};

struct Range : std::ranges::view_base {
  int* begin() const;
  PODSentinel end();
};

template<typename>
struct print_type;

void test() {
  {
    using R = dpl_ranges::zip_view<Range>;
    using Sentinel = std::ranges::sentinel_t<R>;
    static_assert(!std::is_same_v<Sentinel, std::ranges::iterator_t<R>>);

    std::ranges::iterator_t<R> it;
  
    Sentinel s1;
    assert(it != s1); // PODSentinel.b is initialised to false

    Sentinel s2 = {};
    assert(it != s2); // PODSentinel.b is initialised to false
  }
}
#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif //_ENABLE_STD_RANGES_TESTING
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
