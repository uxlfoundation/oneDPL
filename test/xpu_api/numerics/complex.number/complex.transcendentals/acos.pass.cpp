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
//   acos(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::acos(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(TestUtils::infinity_val<T>, 1), dpl::complex<T>(0, -TestUtils::infinity_val<T>));
}

void test_edges()
{
    const double pi = std::atan2(+0., -0.);
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::acos(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(is_about(r.real(), pi/2));
            assert(r.imag() == 0);
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (testcases[i].real() == 0 && std::isnan(testcases[i].imag()))
        {
#if !_PSTL_TEST_COMPLEX_ACOS_BROKEN                         // testcases[37]
            assert(is_about(r.real(), pi/2));
            assert(std::isnan(r.imag()));
#endif // _PSTL_TEST_COMPLEX_ACOS_BROKEN
        }
        else if (std::isfinite(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
#if !_PSTL_TEST_COMPLEX_ACOS_BROKEN_IN_KERNEL
            assert(is_about(r.real(), pi/2));
            assert(std::isinf(r.imag()));
#endif
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isfinite(testcases[i].real()) && testcases[i].real() != 0 && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isfinite(testcases[i].imag()))
        {
            assert(is_about(r.real(), pi));
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isfinite(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(!std::signbit(r.real()));
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isinf(testcases[i].imag()))
        {
            assert(is_about(r.real(), 0.75 * pi));
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isinf(testcases[i].imag()))
        {
            assert(is_about(r.real(), 0.25 * pi));
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
#if !_PSTL_TEST_COMPLEX_ACOS_BROKEN
            assert(std::isnan(r.real()));
            assert(std::isinf(r.imag()));
#endif // _PSTL_TEST_COMPLEX_ACOS_BROKEN
        }
        else if (std::isnan(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
#if !_PSTL_TEST_COMPLEX_ACOS_BROKEN
            assert(std::isnan(r.real()));
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
#endif // _PSTL_TEST_COMPLEX_ACOS_BROKEN
        }
        else if (std::isnan(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (!std::signbit(testcases[i].real()) && !std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert( std::signbit(r.imag()));
        }
        else if (std::signbit(testcases[i].real()) && !std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert( std::signbit(r.imag()));
        }
        else if (std::signbit(testcases[i].real()) && std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert(!std::signbit(r.imag()));
        }
        else if (!std::signbit(testcases[i].real()) && std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert(!std::signbit(r.imag()));
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
#if !_PSTL_TEST_COMPLEX_OP_ACOS_USING_DOUBLE
    test<float>();
#else
    IF_DOUBLE_SUPPORT(test<float>())
#endif

    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
