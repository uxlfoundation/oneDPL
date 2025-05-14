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
//   pow(const complex<T>& x, const T& y);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& a, const T& b, dpl::complex<T> x)
{
    dpl::complex<T> c = dpl::pow(a, b);
    assert(is_about(dpl::real(c), dpl::real(x)));
    assert(is_about(dpl::imag(c), dpl::imag(x)));
}

template <class T>
void
test()
{
    test(dpl::complex<T>(2, 3), T(2), dpl::complex<T>(-5, 12));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            dpl::complex<double> r = dpl::pow(testcases[i], dpl::real(testcases[j]));
            [[maybe_unused]] dpl::complex<double> z = dpl::exp(dpl::complex<double>(dpl::real(testcases[j])) * dpl::log(testcases[i]));
            if (std::isnan(dpl::real(r)))
            {
#if !_PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES       // testcases[0], testcases[33]
                assert(std::isnan(dpl::real(z)));
#endif // _PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES
            }
            else
            {
#if !_PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES       // testcases[0], testcases[35]
                assert(dpl::real(r) == dpl::real(z));
#endif // _PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES
            }
            if (std::isnan(dpl::imag(r)))
            {
#if !_PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES       // testcases[0], testcases[42]
                assert(std::isnan(dpl::imag(z)));
#endif // _PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES
            }
            else
            {
#if !_PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES       // testcases[0], testcases[36]
                assert(dpl::imag(r) == dpl::imag(z));
#endif // _PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES
            }
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
#if _PSTL_ICC_TEST_COMPLEX_MSVC_MATH_DOUBLE_REQ
    IF_DOUBLE_SUPPORT(test<float>())
#else
    test<float>();
#endif
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
