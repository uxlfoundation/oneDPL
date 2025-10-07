//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <complex>

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const T& x, const complex<U>& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const U& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const complex<U>& y);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
double
promote(T, ::std::enable_if_t<std::is_integral_v<T>>* = 0);

float promote(float);
double promote(double);
long double promote(long double);

template <class T, class U>
void
test(T x, const dpl::complex<U>& y)
{
    typedef decltype(promote(x) + promote(dpl::real(y))) V;
    static_assert((std::is_same_v<decltype(dpl::pow(x, y)), dpl::complex<V> >));
    assert(is_about(dpl::pow(x, y), dpl::pow(dpl::complex<V>(x, 0), dpl::complex<V>(y))));
}

template <class T, class U>
void
test(const dpl::complex<T>& x, U y)
{
    typedef decltype(promote(dpl::real(x)) + promote(y)) V;
    static_assert((std::is_same_v<decltype(dpl::pow(x, y)), dpl::complex<V> >));
    assert(is_about(dpl::pow(x, y), dpl::pow(dpl::complex<V>(x), dpl::complex<V>(y, 0))));
}

template <class T, class U>
void
test(const dpl::complex<T>& x, const dpl::complex<U>& y)
{
    typedef decltype(promote(dpl::real(x)) + promote(dpl::real(y))) V;
    static_assert((std::is_same_v<decltype(dpl::pow(x, y)), dpl::complex<V> >));
    assert(dpl::pow(x, y) == dpl::pow(dpl::complex<V>(x), dpl::complex<V>(y)));
}

template <class T, class U>
void
test(::std::enable_if_t<std::is_integral_v<T>>* = 0, ::std::enable_if_t<!std::is_integral_v<U>>* = 0)
{
    test(T(3), dpl::complex<U>(4, 5));
    test(dpl::complex<U>(3, 4), T(5));
}

template <class T, class U>
void
test(::std::enable_if_t<!std::is_integral_v<T>>* = 0, ::std::enable_if_t<!std::is_integral_v<U>>* = 0)
{
    test(T(3), dpl::complex<U>(4, 5));
    test(dpl::complex<T>(3, 4), U(5));
    test(dpl::complex<T>(3, 4), dpl::complex<U>(5, 6));
}

ONEDPL_TEST_NUM_MAIN
{
    IF_DOUBLE_SUPPORT(
#if !_PSTL_GLIBCXX_TEST_COMPLEX_POW_BROKEN
                      test<int, float>();
#endif // !_PSTL_GLIBCXX_TEST_COMPLEX_POW_BROKEN

                      test<unsigned, float>();
                      test<long long, float>();

                      test<int, double>();
                      test<unsigned, double>();
                      test<long long, double>();
                      test<float, double>();
                      test<double, float>())

    IF_LONG_DOUBLE_SUPPORT(test<unsigned, long double>();
                           test<long long, long double>();
                           test<float, long double>();
                           test<double, long double>();
                           test<long double, float>();
                           test<long double, double>())

  return 0;
}
