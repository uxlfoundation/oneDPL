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
//   polar(const T& rho, const T& theta = T());  // changed from '0' by LWG#2870

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const T& rho, dpl::complex<T> x)
{
    assert(dpl::polar(rho) == x);
}

template <class T>
void
test(const T& rho, const T& theta, dpl::complex<T> x)
{
    assert(dpl::polar(rho, theta) == x);
}

template <class T>
void
test()
{
    test(T(0), dpl::complex<T>(0, 0));
    test(T(1), dpl::complex<T>(1, 0));
    test(T(100), dpl::complex<T>(100, 0));
    test(T(0), T(0), dpl::complex<T>(0, 0));
    test(T(1), T(0), dpl::complex<T>(1, 0));
    test(T(100), T(0), dpl::complex<T>(100, 0));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = real(testcases[i]);
        double theta = imag(testcases[i]);
        dpl::complex<double> z = dpl::polar(r, theta);
        switch (classify(r))
        {
        case zero:
            if (std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
            {
#if !_PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES         // testcases[59]
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
#endif // !_PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES
            }
            else
            {
                assert(z == dpl::complex<double>());
            }
            break;
        case non_zero:
            if (std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
            {
#if !_PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES         // testcases[1]
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
#endif // !_PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES
            }
            else
            {
                assert(is_about(dpl::abs(z), r));
            }
            break;
        case inf:
            if (r < 0)
            {
#if !_PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES         // testcases[55]
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
#endif // _PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES
            }
            else
            {
#if !_PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES         // testcases[42]
                assert(classify(z) == inf);
#endif // _PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES
                if (classify(theta) != NaN && classify(theta) != inf)
                {
                    assert(classify(dpl::real(z)) != NaN);
#if !_PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES         // testcases[97]
                    assert(classify(dpl::imag(z)) != NaN);
#endif // _PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES
                }
            }
            break;
        case NaN:
        case non_zero_nan:
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
            break;
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
