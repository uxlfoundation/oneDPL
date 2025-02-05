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
//   pow(const T& x, const complex<T>& y);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const T& a, const dpl::complex<T>& b, dpl::complex<T> x)
{
    dpl::complex<T> c = dpl::pow(a, b);
    assert(is_about(dpl::real(c), dpl::real(x)));
    assert(std::abs(dpl::imag(c)) < T(1.e-6));
}

template <class T>
void
test()
{
    test(T(2), dpl::complex<T>(2), dpl::complex<T>(4));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            dpl::complex<double> r = dpl::pow(dpl::real(testcases[i]), testcases[j]);
            dpl::complex<double> z = dpl::exp(testcases[j] * dpl::log(dpl::complex<double>(dpl::real(testcases[i]))));
            if (std::isnan(dpl::real(r)))
            {
#if !_PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES       // testcases[1], testcases[13]
                assert(std::isnan(dpl::real(z)));
#endif // _PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES
            }
            else
            {
#if !_PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES       // testcases[0], testcases[55]
                assert(dpl::real(r) == dpl::real(z));
#endif // _PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES
            }
            if (std::isnan(dpl::imag(r)))
            {
#if !_PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES       // testcases[0], testcases[27]
                assert(std::isnan(dpl::imag(z)));
#endif // _PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES
            }
            else
            {
#if !_PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES       // testcases[0], testcases[5]
                assert(is_about(dpl::imag(r), dpl::imag(z)));
#endif // _PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES
            }
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
#if !_PSTL_TEST_COMPLEX_OP_POW_SCALAR_COMPLEX_USING_DOUBLE
    test<float>();
#else
    IF_DOUBLE_SUPPORT(test<float>())
#endif

    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
