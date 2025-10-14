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
//   T
//   arg(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test()
{
    dpl::complex<T> z(1, 0);
    assert(dpl::arg(z) == 0);
}

void test_edges()
{
    const double pi = std::atan2(+0., -0.);
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = dpl::arg(testcases[i]);
        if (std::isnan(testcases[i].real()) || std::isnan(testcases[i].imag()))
            assert(std::isnan(r));
        else
        {
            switch (classify(testcases[i]))
            {
            case zero:
                if (std::signbit(testcases[i].real()))
                {
                    if (std::signbit(testcases[i].imag()))
                        assert(is_about(r, -pi));
                    else
                        assert(is_about(r, pi));
                }
                else
                {
                    assert(std::signbit(testcases[i].imag()) == std::signbit(r));
                }
                break;
            case non_zero:
                if (testcases[i].real() == 0)
                {
                    if (testcases[i].imag() < 0)
                        assert(is_about(r, -pi/2));
                    else
                        assert(is_about(r, pi/2));
                }
                else if (testcases[i].imag() == 0)
                {
                    if (testcases[i].real() < 0)
                    {
                        if (std::signbit(testcases[i].imag()))
                            assert(is_about(r, -pi));
                        else
                            assert(is_about(r, pi));
                    }
                    else
                    {
                        assert(r == 0);
                        assert(std::signbit(testcases[i].imag()) == std::signbit(r));
                    }
                }
                else if (testcases[i].imag() > 0)
                    assert(r > 0);
                else
                    assert(r < 0);
                break;
            case inf:
                if (std::isinf(testcases[i].real()) && std::isinf(testcases[i].imag()))
                {
                    if (testcases[i].real() < 0)
                    {
                        if (testcases[i].imag() > 0)
                            assert(is_about(r, 0.75 * pi));
                        else
                            assert(is_about(r, -0.75 * pi));
                    }
                    else
                    {
                        if (testcases[i].imag() > 0)
                            assert(is_about(r, 0.25 * pi));
                        else
                            assert(is_about(r, -0.25 * pi));
                    }
                }
                else if (std::isinf(testcases[i].real()))
                {
                    if (testcases[i].real() < 0)
                    {
                        if (std::signbit(testcases[i].imag()))
                            assert(is_about(r, -pi));
                        else
                            assert(is_about(r, pi));
                    }
                    else
                    {
                        assert(r == 0);
                        assert(std::signbit(r) == std::signbit(testcases[i].imag()));
                    }
                }
                else
                {
                    if (testcases[i].imag() < 0)
                        assert(is_about(r, -pi/2));
                    else
                        assert(is_about(r, pi/2));
                }
                break;
            }
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
