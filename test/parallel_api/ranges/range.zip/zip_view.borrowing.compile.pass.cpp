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

// template<class... Views>
// inline constexpr bool enable_borrowed_range<zip_view<Views...>> =
//      (enable_borrowed_range<Views> && ...);

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING
#include <ranges>

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::experimental::ranges;

struct Borrowed : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<Borrowed> = true;

static_assert(std::ranges::borrowed_range<Borrowed>);

struct NonBorrowed : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};
static_assert(!std::ranges::borrowed_range<NonBorrowed>);

// test borrowed_range
static_assert(std::ranges::borrowed_range<dpl_ranges::zip_view<Borrowed>>);
static_assert(std::ranges::borrowed_range<dpl_ranges::zip_view<Borrowed, Borrowed>>);
static_assert(!std::ranges::borrowed_range<dpl_ranges::zip_view<Borrowed, NonBorrowed>>);
static_assert(!std::ranges::borrowed_range<dpl_ranges::zip_view<NonBorrowed>>);
static_assert(!std::ranges::borrowed_range<dpl_ranges::zip_view<NonBorrowed, NonBorrowed>>);
#endif 

int main() {
  return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
