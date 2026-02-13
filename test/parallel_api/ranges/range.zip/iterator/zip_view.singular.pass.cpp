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

// UNSUPPORTED: no-exceptions

// If the invocation of any non-const member function of `iterator` exits via an
// exception, the iterator acquires a singular value.

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#if defined(_MSC_VER)
  #pragma warning(push)
  #pragma warning(disable : 4702) // unreachable code (MSVC)
#endif

#include <ranges>

#include <tuple>

#include "../types.h"

#include <oneapi/dpl/ranges>

namespace dpl_ranges = oneapi::dpl::experimental::ranges;

struct ThrowOnIncrementIterator {
  int* it_;

  using value_type = int;
  using difference_type = std::intptr_t;
  using iterator_concept = std::input_iterator_tag;

  ThrowOnIncrementIterator() = default;
  explicit ThrowOnIncrementIterator(int* it) : it_(it) {}

  ThrowOnIncrementIterator& operator++() {
    ++it_;
    throw 5;
    return *this;
  }

  void operator++(int) { ++it_; }

  int& operator*() const { return *it_; }

  friend bool operator==(ThrowOnIncrementIterator const&, ThrowOnIncrementIterator const&) = default;
};

struct ThrowOnIncrementView : IntBufferView {
  ThrowOnIncrementIterator begin() const { return ThrowOnIncrementIterator{buffer_}; }
  ThrowOnIncrementIterator end() const { return ThrowOnIncrementIterator{buffer_ + size_}; }
};

// Cannot run the test at compile time because it is not allowed to throw exceptions
void test() {
  int buffer[] = {1, 2, 3};
  {
    // zip iterator should be able to be destroyed after member function throws
    dpl_ranges::zip_view v{ThrowOnIncrementView{buffer}};
    auto it = v.begin();
    try {
      ++it;
      assert(false); // should not be reached as the above expression should throw.
    } catch (int e) {
      assert(e == 5);
    }
  }

  {
    // zip iterator should be able to be assigned after member function throws
    dpl_ranges::zip_view v{ThrowOnIncrementView{buffer}};
    auto it = v.begin();
    try {
      ++it;
      assert(false); // should not be reached as the above expression should throw.
    } catch (int e) {
      assert(e == 5);
    }
    it = v.begin();
    auto [x] = *it;
    assert(x == 1);
  }
}

#if defined(_MSC_VER)
  #pragma warning(pop) //unreachable code
#endif

#endif //_ENABLE_STD_RANGES_TESTING

int main(int, char**) {
#if _ENABLE_STD_RANGES_TESTING
    test();
#endif
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
