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

// friend constexpr auto iter_move(const iterator& i) noexcept(see below);

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>
#include <tuple>

#include "../types.h"

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

template <typename... Types>
using tuple_type = oneapi::dpl::__internal::tuple<Types...>;

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
};

void test() {
  {
    // underlying iter_move noexcept
    std::array a1{1, 2, 3, 4};
    const std::array a2{3.0, 4.0};

    dpl_ranges::zip_view v(a1, a2, std::views::iota(3L));
    assert(std::ranges::iter_move(v.begin()) == std::make_tuple(1, 3.0, 3L));
    static_assert(std::is_same_v<decltype(std::ranges::iter_move(v.begin())), tuple_type<int&&, const double&&, long>>);

    auto it = v.begin();
    static_assert(noexcept(std::ranges::iter_move(it)));
  }

  {
    // underlying iter_move may throw
    auto throwingMoveRange =
        std::views::iota(0, 2) | std::views::transform([](auto) noexcept { return ThrowingMove{}; });
    dpl_ranges::zip_view v(throwingMoveRange);
    auto it = v.begin();
    static_assert(!noexcept(std::ranges::iter_move(it)));
  }

  {
    // underlying iterators' iter_move are called through ranges::iter_move
    adltest::IterMoveSwapRange r1{}, r2{};
    assert(r1.iter_move_called_times == 0);
    assert(r2.iter_move_called_times == 0);
    dpl_ranges::zip_view v(r1, r2);
    auto it = v.begin();
    {
      [[maybe_unused]] auto&& i = std::ranges::iter_move(it);
      assert(r1.iter_move_called_times == 1);
      assert(r2.iter_move_called_times == 1);
    }
    {
      [[maybe_unused]] auto&& i = std::ranges::iter_move(it);
      assert(r1.iter_move_called_times == 2);
      assert(r2.iter_move_called_times == 2);
    }
  }
}
#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
