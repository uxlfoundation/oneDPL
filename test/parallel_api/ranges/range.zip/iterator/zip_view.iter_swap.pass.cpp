//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// friend constexpr void iter_swap(const iterator& l, const iterator& r) noexcept(see below)
//   requires (indirectly_swappable<iterator_t<maybe-const<Const, Views>>> && ...);

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <array>
#include <cassert>
#include <ranges>

#include "../types.h"

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::ranges;

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
  ThrowingMove& operator=(ThrowingMove&&){return *this;}
};

void test() {
  {
    std::array a1{1, 2, 3, 4};
    std::array a2{0.1, 0.2, 0.3};
    dpl_ranges::zip_view v(a1, a2);
    auto iter1 = v.begin();
    auto iter2 = ++v.begin();

    std::ranges::iter_swap(iter1, iter2);

    assert(a1[0] == 2);
    assert(a1[1] == 1);
    assert(a2[0] == 0.2);
    assert(a2[1] == 0.1);

    auto [x1, y1] = *iter1;
    assert(&x1 == &a1[0]);
    assert(&y1 == &a2[0]);

    auto [x2, y2] = *iter2;
    assert(&x2 == &a1[1]);
    assert(&y2 == &a2[1]);

    static_assert(noexcept(std::ranges::iter_swap(iter1, iter2)));
  }

  {
    // underlying iter_swap may throw
    std::array<ThrowingMove, 2> iterSwapMayThrow{};
    dpl_ranges::zip_view v(iterSwapMayThrow);
    auto iter1 = v.begin();
    auto iter2 = ++v.begin();
    static_assert(!noexcept(std::ranges::iter_swap(iter1, iter2)));
  }

  {
    // underlying iterators' iter_move are called through ranges::iter_swap
    adltest::IterMoveSwapRange r1, r2;
    assert(r1.iter_swap_called_times == 0);
    assert(r2.iter_swap_called_times == 0);

    dpl_ranges::zip_view v{r1, r2};
    auto it1 = v.begin();
    auto it2 = std::ranges::next(it1, 3);

    std::ranges::iter_swap(it1, it2);
    assert(r1.iter_swap_called_times == 2);
    assert(r2.iter_swap_called_times == 2);

    std::ranges::iter_swap(it1, it2);
    assert(r1.iter_swap_called_times == 4);
    assert(r2.iter_swap_called_times == 4);
  }
}
#endif //_ENABLE_STD_RANGES_TESTING

int main() {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
