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

// zip_view() = default;

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

constexpr int buff[] = {1, 2, 3};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + 3) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

private:
  int const* begin_;
  int const* end_;
};

struct NoDefaultCtrView : std::ranges::view_base {
  NoDefaultCtrView() = delete;
  int* begin() const;
  int* end() const;
};

// The default constructor requires all underlying views to be default constructible.
// It is implicitly required by the tuple's constructor. If any of the iterators are
// not default constructible, zip iterator's =default would be implicitly deleted.
static_assert(std::is_default_constructible_v<dpl_ranges::zip_view<DefaultConstructibleView>>);
static_assert(
    std::is_default_constructible_v<dpl_ranges::zip_view<DefaultConstructibleView, DefaultConstructibleView>>);
static_assert(!std::is_default_constructible_v<dpl_ranges::zip_view<DefaultConstructibleView, NoDefaultCtrView>>);
static_assert(!std::is_default_constructible_v<dpl_ranges::zip_view<NoDefaultCtrView, NoDefaultCtrView>>);
static_assert(!std::is_default_constructible_v<dpl_ranges::zip_view<NoDefaultCtrView>>);

int test() {
  {
    using View = dpl_ranges::zip_view<DefaultConstructibleView, DefaultConstructibleView>;
    View v = View(); // the default constructor is not explicit
    assert(v.size() == 3);
    auto it = v.begin();
#ifdef _LIBCPP_VERSION // libc++ doesn't implement P2165R4 yet
    using Value = std::pair<const int&, const int&>;
#else
    using Value = tuple_type<const int&, const int&>;
#endif
    assert(*it++ == Value(buff[0], buff[0]));
    assert(*it++ == Value(buff[1], buff[1]));
    assert(*it == Value(buff[2], buff[2]));
  }

  return 0;
}
#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
