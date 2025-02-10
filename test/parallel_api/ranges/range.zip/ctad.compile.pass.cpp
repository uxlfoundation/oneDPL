//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <class... Rs>
// zip_view(Rs&&...) -> zip_view<views::all_t<Rs>...>;

#include <cassert>
#include <utility>

#include <oneapi/dpl/ranges>
#include <ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

struct Container {
  int* begin() const;
  int* end() const;
};

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

void testCTAD() {
  using t1 = std::ranges::owning_view<Container>;
  auto var = dpl_ranges::zip_view(Container{});
  static_assert(std::is_same_v<decltype(dpl_ranges::zip_view(Container{})),
                               dpl_ranges::zip_view<std::ranges::owning_view<Container>>>);

  static_assert(std::is_same_v<decltype(dpl_ranges::zip_view(Container{}, View{})),
                               dpl_ranges::zip_view<std::ranges::owning_view<Container>, View>>);

  Container c{};
  static_assert(std::is_same_v<
                decltype(dpl_ranges::zip_view(Container{}, View{}, c)),
                dpl_ranges::zip_view<std::ranges::owning_view<Container>, View, std::ranges::ref_view<Container>>>);
}
