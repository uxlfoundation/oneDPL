//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <complex>

// template<class T>
//   complex<T>
//   acosh(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::acosh(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(TestUtils::infinity_val<T>, 1), dpl::complex<T>(TestUtils::infinity_val<T>, 0));
}

void test_edges()
{
    const double pi = std::atan2(+0., -0.);
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::acosh(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(!std::signbit(r.real()));
            if (std::signbit(testcases[i].imag()))
                assert(is_about(r.imag(), -pi/2));
            else
                assert(is_about(r.imag(),  pi/2));
        }
        else if (testcases[i].real() == 1 && testcases[i].imag() == 0)
        {
            assert(r.real() == 0);
            assert(!std::signbit(r.real()));
            assert(r.imag() == 0);
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (testcases[i].real() == -1 && testcases[i].imag() == 0)
        {
            assert(r.real() == 0);
            assert(!std::signbit(r.real()));
            if (std::signbit(testcases[i].imag()))
                assert(is_about(r.imag(), -pi));
            else
                assert(is_about(r.imag(),  pi));
        }
        else if (std::isfinite(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            if (std::signbit(testcases[i].imag()))
                assert(is_about(r.imag(), -pi/2));
            else
                assert(is_about(r.imag(),  pi/2));
        }
        else if (std::isfinite(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
#if !_PSTL_CLANG_TEST_COMPLEX_ACOS_IS_NAN_CASE_BROKEN
            assert(std::isnan(r.imag()));
#endif // _PSTL_CLANG_TEST_COMPLEX_ACOS_IS_NAN_CASE_BROKEN
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isfinite(testcases[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            if (std::signbit(testcases[i].imag()))
                assert(is_about(r.imag(), -pi));
            else
                assert(is_about(r.imag(),  pi));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isfinite(testcases[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            assert(r.imag() == 0);
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isinf(testcases[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            if (std::signbit(testcases[i].imag()))
                assert(is_about(r.imag(), -0.75 * pi));
            else
                assert(is_about(r.imag(),  0.75 * pi));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isinf(testcases[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            if (std::signbit(testcases[i].imag()))
                assert(is_about(r.imag(), -0.25 * pi));
            else
                assert(is_about(r.imag(),  0.25 * pi));
        }
        else if (std::isinf(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
#if !_PSTL_TEST_COMPLEX_ACOSH_BROKEN
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            assert(std::isnan(r.imag()));
#endif // _PSTL_TEST_COMPLEX_ACOSH_BROKEN
        }
        else if (std::isnan(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
#if !_PSTL_TEST_COMPLEX_ACOSH_BROKEN
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            assert(std::isnan(r.imag()));
#endif // _PSTL_TEST_COMPLEX_ACOSH_BROKEN
        }
        else if (std::isnan(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else
        {
            assert(!std::signbit(r.real()));
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
#if !_PSTL_TEST_COMPLEX_OP_ACOSH_USING_DOUBLE
    test<float>();
#else
    IF_DOUBLE_SUPPORT(test<float>())
#endif

    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
