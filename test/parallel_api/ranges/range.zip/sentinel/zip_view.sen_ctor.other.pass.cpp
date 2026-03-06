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

// constexpr sentinel(sentinel<!Const> s);

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <cassert>
#include <ranges>

#include "../types.h"

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::experimental::ranges;

template <class T>
struct convertible_sentinel_wrapper {
  explicit convertible_sentinel_wrapper() = default;
  constexpr convertible_sentinel_wrapper(const T& it) : it_(it) {}

  template <class U>
    requires std::convertible_to<const U&, T>
  constexpr convertible_sentinel_wrapper(const convertible_sentinel_wrapper<U>& other) : it_(other.it_) {}

  constexpr friend bool operator==(convertible_sentinel_wrapper const& self, const T& other) {
    return self.it_ == other;
  }
  T it_;
};

struct NonSimpleNonCommonConvertibleView : IntBufferView {
  using IntBufferView::IntBufferView;

  constexpr int* begin() { return buffer_; }
  constexpr const int* begin() const { return buffer_; }
  constexpr convertible_sentinel_wrapper<int*> end() { return convertible_sentinel_wrapper<int*>(buffer_ + size_); }
  constexpr convertible_sentinel_wrapper<const int*> end() const {
    return convertible_sentinel_wrapper<const int*>(buffer_ + size_);
  }
};

static_assert(!std::ranges::common_range<NonSimpleNonCommonConvertibleView>);
static_assert(std::ranges::random_access_range<NonSimpleNonCommonConvertibleView>);
static_assert(!std::ranges::sized_range<NonSimpleNonCommonConvertibleView>);
static_assert(std::convertible_to<std::ranges::sentinel_t<NonSimpleNonCommonConvertibleView>,
                                  std::ranges::sentinel_t<NonSimpleNonCommonConvertibleView const>>);
static_assert(!simple_view<NonSimpleNonCommonConvertibleView>);

void test() {
  int buffer1[4] = {1, 2, 3, 4};
  int buffer2[5] = {1, 2, 3, 4, 5};
  dpl_ranges::zip_view v{NonSimpleNonCommonConvertibleView(buffer1), NonSimpleNonCommonConvertibleView(buffer2)};
  static_assert(!std::ranges::common_range<decltype(v)>);
  auto sent1 = v.end();
  std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
  static_assert(!std::is_same_v<decltype(sent1), decltype(sent2)>);

  assert(v.begin() != sent2);
  assert(std::as_const(v).begin() != sent2);
  assert(v.begin() + 4 == sent2);
  assert(std::as_const(v).begin() + 4 == sent2);

  // Cannot create a non-const iterator from a const iterator.
  static_assert(!std::constructible_from<decltype(sent1), decltype(sent2)>);
}
#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
