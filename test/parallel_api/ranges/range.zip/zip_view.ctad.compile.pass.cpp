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

// template <class... Rs>
// zip_view(Rs&&...) -> zip_view<views::all_t<Rs>...>;

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <cassert>
#include <utility>

#include <oneapi/dpl/ranges>
#include <ranges>

namespace dpl_ranges = oneapi::dpl::experimental::ranges;

struct Container {
  int* begin() const;
  int* end() const;
};

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

#if __GNUC__ && _ONEDPL_GCC_VERSION >= 120100
void testCTAD() {  
  static_assert(std::is_same_v<decltype(dpl_ranges::zip_view(Container{})),
                               dpl_ranges::zip_view<std::ranges::owning_view<Container>>>);

  static_assert(std::is_same_v<decltype(dpl_ranges::zip_view(Container{}, View{})),
                               dpl_ranges::zip_view<std::ranges::owning_view<Container>, View>>);

  Container c{};
  static_assert(std::is_same_v<
                decltype(dpl_ranges::zip_view(Container{}, View{}, c)),
                dpl_ranges::zip_view<std::ranges::owning_view<Container>, View, std::ranges::ref_view<Container>>>);
}
#endif

#endif //_ENABLE_STD_RANGES_TESTING

int main()
{
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}