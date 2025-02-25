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

// constexpr iterator& operator--() requires all-bidirectional<Const, Views...>;
// constexpr iterator operator--(int) requires all-bidirectional<Const, Views...>;

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>

#include "../types.h"

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

template <class Iter>
concept canDecrement = requires(Iter it) { --it; } || requires(Iter it) { it--; };

struct NonBidi : IntBufferView {
  using IntBufferView::IntBufferView;
  using iterator = forward_iterator<int*>;
  constexpr iterator begin() const { return iterator(buffer_); }
  constexpr sentinel_wrapper<iterator> end() const { return sentinel_wrapper<iterator>(iterator(buffer_ + size_)); }
};

void test() {
  std::array a{1, 2, 3, 4};
  std::array b{4.1, 3.2, 4.3};
  {
    // all random access
    dpl_ranges::zip_view v(a, b, std::views::iota(0, 5));
    auto it = v.end();
    using Iter = decltype(it);

    static_assert(std::is_same_v<decltype(--it), Iter&>);
    auto& it_ref = --it;
    assert(&it_ref == &it);

    assert(&(std::get<0>(*it)) == &(a[2]));
    assert(&(std::get<1>(*it)) == &(b[2]));
    assert(std::get<2>(*it) == 2);

    static_assert(std::is_same_v<decltype(it--), Iter>);
    it--;
    assert(&(std::get<0>(*it)) == &(a[1]));
    assert(&(std::get<1>(*it)) == &(b[1]));
    assert(std::get<2>(*it) == 1);
  }

  {
    // all bidi+
    int buffer[2] = {1, 2};

    dpl_ranges::zip_view v(BidiCommonView{buffer}, std::views::iota(0, 5));
    auto it = v.begin();
    using Iter = decltype(it);

    ++it;
    ++it;

    static_assert(std::is_same_v<decltype(--it), Iter&>);
    auto& it_ref = --it;
    assert(&it_ref == &it);

    assert(it == ++v.begin());

    static_assert(std::is_same_v<decltype(it--), Iter>);
    auto tmp = it--;
    assert(it == v.begin());
    assert(tmp == ++v.begin());
  }

  {
    // non bidi
    int buffer[3] = {4, 5, 6};
    dpl_ranges::zip_view v(a, NonBidi{buffer});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(!canDecrement<Iter>);
  }
}

#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
