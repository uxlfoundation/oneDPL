//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// test cases

#ifndef _CASES_H
#define _CASES_H

#include <oneapi/dpl/complex>
#include <cassert>
#include <type_traits>

// ATTENTION: the indexes of this array are used in the test_edges() function in comments for broken cases
// So please add new testcases to the end of the array.
const dpl::complex<double> testcases[] =
{
    dpl::complex<double>( 1.e-6,  1.e-6),
    dpl::complex<double>(-1.e-6,  1.e-6),
    dpl::complex<double>(-1.e-6, -1.e-6),
    dpl::complex<double>( 1.e-6, -1.e-6),

    dpl::complex<double>( 1.e+6,  1.e-6),
    dpl::complex<double>(-1.e+6,  1.e-6),
    dpl::complex<double>(-1.e+6, -1.e-6),
    dpl::complex<double>( 1.e+6, -1.e-6),

    dpl::complex<double>( 1.e-6,  1.e+6),
    dpl::complex<double>(-1.e-6,  1.e+6),
    dpl::complex<double>(-1.e-6, -1.e+6),
    dpl::complex<double>( 1.e-6, -1.e+6),

    dpl::complex<double>( 1.e+6,  1.e+6),
    dpl::complex<double>(-1.e+6,  1.e+6),
    dpl::complex<double>(-1.e+6, -1.e+6),
    dpl::complex<double>( 1.e+6, -1.e+6),

    dpl::complex<double>(-0, -1.e-6),
    dpl::complex<double>(-0,  1.e-6),
    dpl::complex<double>(-0,  1.e+6),
    dpl::complex<double>(-0, -1.e+6),
    dpl::complex<double>( 0, -1.e-6),
    dpl::complex<double>( 0,  1.e-6),
    dpl::complex<double>( 0,  1.e+6),
    dpl::complex<double>( 0, -1.e+6),

    dpl::complex<double>(-1.e-6, -0),
    dpl::complex<double>( 1.e-6, -0),
    dpl::complex<double>( 1.e+6, -0),
    dpl::complex<double>(-1.e+6, -0),
    dpl::complex<double>(-1.e-6,  0),
    dpl::complex<double>( 1.e-6,  0),
    dpl::complex<double>( 1.e+6,  0),
    dpl::complex<double>(-1.e+6,  0),

    dpl::complex<double>(NAN, NAN),
    dpl::complex<double>(-INFINITY, NAN),
    dpl::complex<double>(-2, NAN),
    dpl::complex<double>(-1, NAN),
    dpl::complex<double>(-0.5, NAN),
    dpl::complex<double>(-0., NAN),
    dpl::complex<double>(+0., NAN),
    dpl::complex<double>(0.5, NAN),
    dpl::complex<double>(1, NAN),
    dpl::complex<double>(2, NAN),
    dpl::complex<double>(INFINITY, NAN),

    dpl::complex<double>(NAN, -INFINITY),
    dpl::complex<double>(-INFINITY, -INFINITY),
    dpl::complex<double>(-2, -INFINITY),
    dpl::complex<double>(-1, -INFINITY),
    dpl::complex<double>(-0.5, -INFINITY),
    dpl::complex<double>(-0., -INFINITY),
    dpl::complex<double>(+0., -INFINITY),
    dpl::complex<double>(0.5, -INFINITY),
    dpl::complex<double>(1, -INFINITY),
    dpl::complex<double>(2, -INFINITY),
    dpl::complex<double>(INFINITY, -INFINITY),

    dpl::complex<double>(NAN, -2),
    dpl::complex<double>(-INFINITY, -2),
    dpl::complex<double>(-2, -2),
    dpl::complex<double>(-1, -2),
    dpl::complex<double>(-0.5, -2),
    dpl::complex<double>(-0., -2),
    dpl::complex<double>(+0., -2),
    dpl::complex<double>(0.5, -2),
    dpl::complex<double>(1, -2),
    dpl::complex<double>(2, -2),
    dpl::complex<double>(INFINITY, -2),

    dpl::complex<double>(NAN, -1),
    dpl::complex<double>(-INFINITY, -1),
    dpl::complex<double>(-2, -1),
    dpl::complex<double>(-1, -1),
    dpl::complex<double>(-0.5, -1),
    dpl::complex<double>(-0., -1),
    dpl::complex<double>(+0., -1),
    dpl::complex<double>(0.5, -1),
    dpl::complex<double>(1, -1),
    dpl::complex<double>(2, -1),
    dpl::complex<double>(INFINITY, -1),

    dpl::complex<double>(NAN, -0.5),
    dpl::complex<double>(-INFINITY, -0.5),
    dpl::complex<double>(-2, -0.5),
    dpl::complex<double>(-1, -0.5),
    dpl::complex<double>(-0.5, -0.5),
    dpl::complex<double>(-0., -0.5),
    dpl::complex<double>(+0., -0.5),
    dpl::complex<double>(0.5, -0.5),
    dpl::complex<double>(1, -0.5),
    dpl::complex<double>(2, -0.5),
    dpl::complex<double>(INFINITY, -0.5),

    dpl::complex<double>(NAN, -0.),
    dpl::complex<double>(-INFINITY, -0.),
    dpl::complex<double>(-2, -0.),
    dpl::complex<double>(-1, -0.),
    dpl::complex<double>(-0.5, -0.),
    dpl::complex<double>(-0., -0.),
    dpl::complex<double>(+0., -0.),
    dpl::complex<double>(0.5, -0.),
    dpl::complex<double>(1, -0.),
    dpl::complex<double>(2, -0.),
    dpl::complex<double>(INFINITY, -0.),

    dpl::complex<double>(NAN, +0.),
    dpl::complex<double>(-INFINITY, +0.),
    dpl::complex<double>(-2, +0.),
    dpl::complex<double>(-1, +0.),
    dpl::complex<double>(-0.5, +0.),
    dpl::complex<double>(-0., +0.),
    dpl::complex<double>(+0., +0.),
    dpl::complex<double>(0.5, +0.),
    dpl::complex<double>(1, +0.),
    dpl::complex<double>(2, +0.),
    dpl::complex<double>(INFINITY, +0.),

    dpl::complex<double>(NAN, 0.5),
    dpl::complex<double>(-INFINITY, 0.5),
    dpl::complex<double>(-2, 0.5),
    dpl::complex<double>(-1, 0.5),
    dpl::complex<double>(-0.5, 0.5),
    dpl::complex<double>(-0., 0.5),
    dpl::complex<double>(+0., 0.5),
    dpl::complex<double>(0.5, 0.5),
    dpl::complex<double>(1, 0.5),
    dpl::complex<double>(2, 0.5),
    dpl::complex<double>(INFINITY, 0.5),

    dpl::complex<double>(NAN, 1),
    dpl::complex<double>(-INFINITY, 1),
    dpl::complex<double>(-2, 1),
    dpl::complex<double>(-1, 1),
    dpl::complex<double>(-0.5, 1),
    dpl::complex<double>(-0., 1),
    dpl::complex<double>(+0., 1),
    dpl::complex<double>(0.5, 1),
    dpl::complex<double>(1, 1),
    dpl::complex<double>(2, 1),
    dpl::complex<double>(INFINITY, 1),

    dpl::complex<double>(NAN, 2),
    dpl::complex<double>(-INFINITY, 2),
    dpl::complex<double>(-2, 2),
    dpl::complex<double>(-1, 2),
    dpl::complex<double>(-0.5, 2),
    dpl::complex<double>(-0., 2),
    dpl::complex<double>(+0., 2),
    dpl::complex<double>(0.5, 2),
    dpl::complex<double>(1, 2),
    dpl::complex<double>(2, 2),
    dpl::complex<double>(INFINITY, 2),

    dpl::complex<double>(NAN, INFINITY),
    dpl::complex<double>(-INFINITY, INFINITY),
    dpl::complex<double>(-2, INFINITY),
    dpl::complex<double>(-1, INFINITY),
    dpl::complex<double>(-0.5, INFINITY),
    dpl::complex<double>(-0., INFINITY),
    dpl::complex<double>(+0., INFINITY),
    dpl::complex<double>(0.5, INFINITY),
    dpl::complex<double>(1, INFINITY),
    dpl::complex<double>(2, INFINITY),
    dpl::complex<double>(INFINITY, INFINITY)
};

enum {zero, non_zero, inf, NaN, non_zero_nan};

template <class T>
int
classify(const dpl::complex<T>& x)
{
    if (x == dpl::complex<T>())
        return zero;
    if (std::isinf(x.real()) || std::isinf(x.imag()))
        return inf;
    if (std::isnan(x.real()) && std::isnan(x.imag()))
        return NaN;
    if (std::isnan(x.real()))
    {
        if (x.imag() == T(0))
            return NaN;
        return non_zero_nan;
    }
    if (std::isnan(x.imag()))
    {
        if (x.real() == T(0))
            return NaN;
        return non_zero_nan;
    }
    return non_zero;
}

inline
int
classify(double x)
{
    if (x == 0)
        return zero;
    if (std::isinf(x))
        return inf;
    if (std::isnan(x))
        return NaN;
    return non_zero;
}

template <typename T>
constexpr auto __tol = std::numeric_limits<T>::epsilon() * 1e5;

template <typename X, typename Y>
::std::enable_if_t<!std::numeric_limits<X>::is_integer || !std::numeric_limits<Y>::is_integer, bool>
is_about(X x, Y y, const X eps = __tol<X>)
{
    return std::fabs(x - y) <= eps;
}

template <typename T>
::std::enable_if_t<!std::numeric_limits<T>::is_integer, bool>
is_about(const dpl::complex<T>& x, const dpl::complex<T>& y, const T eps = __tol<T>)
{
    return is_about(::std::abs(y - x), T(0.), eps);
}

#endif // _CASES_H
