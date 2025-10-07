//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   operator/(const T& lhs, const complex<T>& rhs);

#include "support/test_complex.h"
#include "./../cases.h"

template <class T>
void
test(const T& lhs, const dpl::complex<T>& rhs, dpl::complex<T> x)
{
    assert(is_about(lhs / rhs, x));
}

template <class T>
void
test()
{
    T lhs(-8.5f);
    dpl::complex<T> rhs(1.5f, 2.5f);
    dpl::complex<T>   x(-1.5f, 2.5f);
    test(lhs, rhs, x);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
