//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator(iterator<!Const> i)
//       requires Const && (convertible_to<iterator_t<Views>,
//                                         iterator_t<maybe-const<Const, Views>>> && ...);

#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <ranges>

#include <cassert>
#include <tuple>

#include "../types.h"

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

using ConstIterIncompatibleView = BasicView<forward_iterator<int*>, forward_iterator<int*>,
                                            random_access_iterator<const int*>, random_access_iterator<const int*>>;
static_assert(!std::convertible_to<std::ranges::iterator_t<ConstIterIncompatibleView>,
                                   std::ranges::iterator_t<const ConstIterIncompatibleView>>);

void test() {
  int buffer[3] = {1, 2, 3};

  {
    dpl_ranges::zip_view v(NonSimpleCommon{buffer});
    auto iter1 = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    assert(iter1 == iter2);

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);

    // We cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
  }

  {
    // underlying non-const to const not convertible
    dpl_ranges::zip_view v(ConstIterIncompatibleView{buffer});
    auto iter1 = v.begin();
    auto iter2 = std::as_const(v).begin();

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);

    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter2), decltype(iter1)>);
  }
}
#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
