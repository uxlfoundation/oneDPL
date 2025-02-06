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
//   exp(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::exp(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(0, 0), dpl::complex<T>(1, 0));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::exp(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(r.real() == 1.0);
            assert(r.imag() == 0);
#if !_PSTL_ICC_TEST_COMPLEX_EXP_BROKEN_TEST_EDGES
            assert(std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
#endif
        }
        else if (std::isfinite(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
#if !_PSTL_ICC_TEST_COMPLEX_EXP_BROKEN_TEST_EDGES
            assert(std::isnan(r.real()));   // test case: 45, 46, 47, 48, 49, 50, 51, 52, 144, 145, 146, 147, 148, 149, 150 and etc.
            assert(std::isnan(r.imag()));   // test case: 45, 46, 47, 48, 49, 50, 51, 52, 144, 145, 146, 147, 148, 149, 150 and etc.
#endif
        }
        else if (std::isfinite(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
#if !_PSTL_ICC_TEST_COMPLEX_EXP_BROKEN_TEST_EDGES
            assert(std::isnan(r.real()));   // test case: 34, 35, 36, 37, 38, 39, 40, 41
            assert(std::isnan(r.imag()));   // test case: 34, 35, 36, 37, 38, 39, 40, 41
#endif
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && testcases[i].imag() == 0)
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            assert(r.imag() == 0);
            assert(std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isinf(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(r.imag() == 0);
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isinf(testcases[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isnan(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(r.imag() == 0);
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isnan(testcases[i].imag()))
        {
            assert(std::isinf(r.real()));
#if !_PSTL_ICC_TEST_COMPLEX_EXP_BROKEN_TEST_EDGES && !_PSTL_ICC_TEST_COMPLEX_EXP_BROKEN_TEST_EDGES_LATEST
            assert(std::isnan(r.imag()));
#endif
        }
        else if (std::isnan(testcases[i].real()) && testcases[i].imag() == 0)
        {
            assert(std::isnan(r.real()));
            assert(r.imag() == 0);
            assert(std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && testcases[i].imag() != 0)
        {
#if !_PSTL_ICC_TEST_COMPLEX_EXP_BROKEN_TEST_EDGES
            assert(std::isnan(r.real()));       // test case: 32, 34, 43, 54, 65, 76, 95, 98, 109, 120, 131, 142
            assert(std::isnan(r.imag()));       // test case: 32, 34, 43, 54, 65, 76, 95, 98, 109, 120, 131, 142
#endif
        }
        else if (std::isnan(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::isfinite(testcases[i].imag()) && std::abs(testcases[i].imag()) <= 1)
        {
            assert(!std::signbit(r.real()));
#if !_PSTL_TEST_COMPLEX_EXP_BROKEN
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
#endif
        }
        else if (std::isinf(r.real()) && testcases[i].imag() == 0) {
            assert(r.imag() == 0);
            assert(std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
