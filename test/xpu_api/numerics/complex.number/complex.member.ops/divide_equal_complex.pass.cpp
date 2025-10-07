//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator/=(const complex& rhs);

#include "support/test_complex.h"
#include "./../cases.h"

template <class T>
void
test()
{
    dpl::complex<T> c(-4, 7.5f);
    const dpl::complex<T> c2(1.5f, 2.5f);
    assert(c.real() == -4);
    assert(c.imag() == 7.5f);
    c /= c2;
    assert(is_about(c.real(), 1.5f));
    assert(is_about(c.imag(), 2.5f));
    c /= c2;
    assert(is_about(c.real(), 1));
    assert(is_about(c.imag(), 0));

    dpl::complex<T> c3;

#if _PSTL_TEST_COMPLEX_NON_FLOAT_AVAILABLE && !_PSTL_GLIBCXX_TEST_COMPLEX_DIV_EQ_BROKEN
    c3 = c;
    dpl::complex<int> ic (1,1);
    c3 /= ic;
    assert(is_about(c3.real(),  0.5f));
    assert(is_about(c3.imag(), -0.5f));
#endif // _PSTL_TEST_COMPLEX_NON_FLOAT_AVAILABLE && !_PSTL_GLIBCXX_TEST_COMPLEX_DIV_EQ_BROKEN

    c3 = c;
    dpl::complex<float> fc (1,1);
    c3 /= fc;
    assert(is_about(c3.real(),  0.5f));
    assert(is_about(c3.imag(), -0.5f));
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
