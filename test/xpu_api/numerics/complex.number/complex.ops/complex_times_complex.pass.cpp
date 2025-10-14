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
//   operator*(const complex<T>& lhs, const complex<T>& rhs);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& lhs, const dpl::complex<T>& rhs, dpl::complex<T> x)
{
    assert(lhs * rhs == x);
}

template <class T>
void
test()
{
    dpl::complex<T> lhs(1.5, 2.5);
    dpl::complex<T> rhs(1.5, 2.5);
    dpl::complex<T>   x(-4.0, 7.5);
    test(lhs, rhs, x);
}

// test edges

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            dpl::complex<double> r = testcases[i] * testcases[j];
            switch (classify(testcases[i]))
            {
            case zero:
                switch (classify(testcases[j]))
                {
                case zero:
                    assert(classify(r) == zero);
                    break;
                case non_zero:
                    assert(classify(r) == zero);
                    break;
                case inf:
                    assert(classify(r) == NaN);
                    break;
                case NaN:
                    assert(classify(r) == NaN);
                    break;
                case non_zero_nan:
                    assert(classify(r) == NaN);
                    break;
                }
                break;
            case non_zero:
                switch (classify(testcases[j]))
                {
                case zero:
                    assert(classify(r) == zero);
                    break;
                case non_zero:
                    assert(classify(r) == non_zero);
                    break;
                case inf:
#if !_PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN            // testcases[0], testcases[33]
                    assert(classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN
                    break;
                case NaN:
                    assert(classify(r) == NaN);
                    break;
                case non_zero_nan:
                    assert(classify(r) == NaN);
                    break;
                }
                break;
            case inf:
                switch (classify(testcases[j]))
                {
                case zero:
                    assert(classify(r) == NaN);
                    break;
                case non_zero:
#if !_PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN            // testcases[33], testcases[0]
                    assert(classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN
                    break;
                case inf:
#if !_PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN            // testcases[33], testcases[33]
                    assert(classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN
                    break;
                case NaN:
                    assert(classify(r) == NaN);
                    break;
                case non_zero_nan:
#if !_PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN            // testcases[33], testcases[34]
                    assert(classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN
                    break;
                }
                break;
            case NaN:
                switch (classify(testcases[j]))
                {
                case zero:
                    assert(classify(r) == NaN);
                    break;
                case non_zero:
                    assert(classify(r) == NaN);
                    break;
                case inf:
                    assert(classify(r) == NaN);
                    break;
                case NaN:
                    assert(classify(r) == NaN);
                    break;
                case non_zero_nan:
                    assert(classify(r) == NaN);
                    break;
                }
                break;
            case non_zero_nan:
                switch (classify(testcases[j]))
                {
                case zero:
                    assert(classify(r) == NaN);
                    break;
                case non_zero:
                    assert(classify(r) == NaN);
                    break;
                case inf:
#if !_PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN                // testcases[33], testcases[34]
                    assert(classify(r) == inf);
#endif // _PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN
                    break;
                case NaN:
                    assert(classify(r) == NaN);
                    break;
                case non_zero_nan:
                    assert(classify(r) == NaN);
                    break;
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
