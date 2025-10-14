//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<Arithmetic T>
//   T
//   real(const T& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T, int x>
void
test(::std::enable_if_t<std::is_integral_v<T>>* = 0)
{
    static_assert((std::is_same_v<decltype(dpl::real(T(x))), double>));
    assert(dpl::real(T(x)) == x);

    constexpr T val {x};
    STD_COMPLEX_TESTS_STATIC_ASSERT(dpl::real(val) == val);
    constexpr dpl::complex<T> t{val, val};
    STD_COMPLEX_TESTS_STATIC_ASSERT(t.real() == x );
}

template <class T, int x>
void
test(::std::enable_if_t<!std::is_integral_v<T>>* = 0)
{
    static_assert((std::is_same_v<decltype(dpl::real(T(x))), T>));
    assert(dpl::real(T(x)) == x);

    constexpr T val {x};
    STD_COMPLEX_TESTS_STATIC_ASSERT(dpl::real(val) == val);
    constexpr dpl::complex<T> t{val, val};
    STD_COMPLEX_TESTS_STATIC_ASSERT(t.real() == x);
}

template <class T>
void
test()
{
    test<T, 0>();
    test<T, 1>();
    test<T, 10>();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
#if _PSTL_TEST_COMPLEX_NON_FLOAT_AVAILABLE
    IF_DOUBLE_SUPPORT(test<int>())
    IF_DOUBLE_SUPPORT(test<unsigned>())
    IF_DOUBLE_SUPPORT(test<long long>())
#endif // _PSTL_TEST_COMPLEX_NON_FLOAT_AVAILABLE

  return 0;
}
