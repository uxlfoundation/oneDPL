//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <cmath>

#include "support/test_complex.h"
#include "support/test_macros.h"

#include <oneapi/dpl/cmath>
#include <oneapi/dpl/limits>
#include <type_traits>
#include <cassert>

#define ONEDPL_TEST_DECLARE                                                                           \
template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime>

#define ONEDPL_TEST_CALL(fnc)                                                                         \
fnc<HasDoubleSupportInRuntime, HasLongDoubleSupportInCompiletime>();

// convertible to int/float/double/etc
template <class T, int N=0>
struct Value {
    operator T () { return T(N); }
};

// See PR21083
// Ambiguous is a user-defined type that defines its own overloads of cmath
// functions. When the std overloads are candidates too (by using or adl),
// they should not interfere.
struct Ambiguous : std::true_type { // ADL
    operator float () { return 0.f; }
    operator double () { return 0.; }
};

namespace oneapi
{
namespace dpl
{
Ambiguous abs(Ambiguous) { return Ambiguous(); }
Ambiguous ceil(Ambiguous) { return Ambiguous(); }
Ambiguous exp(Ambiguous) { return Ambiguous(); }
Ambiguous fabs(Ambiguous) { return Ambiguous(); }
Ambiguous floor(Ambiguous) { return Ambiguous(); }
Ambiguous isgreater(Ambiguous, Ambiguous) { return Ambiguous(); }
Ambiguous isgreaterequal(Ambiguous, Ambiguous) { return Ambiguous(); }
Ambiguous isless(Ambiguous, Ambiguous) { return Ambiguous(); }
Ambiguous islessequal(Ambiguous, Ambiguous) { return Ambiguous(); }
Ambiguous copysign(Ambiguous, Ambiguous) { return Ambiguous(); }
Ambiguous fmax(Ambiguous, Ambiguous) { return Ambiguous(); }
Ambiguous fmin(Ambiguous, Ambiguous) { return Ambiguous(); }
Ambiguous isunordered(Ambiguous, Ambiguous) { return Ambiguous(); }
Ambiguous round(Ambiguous) { return Ambiguous(); }
Ambiguous trunc(Ambiguous) { return Ambiguous(); }
}; // namespace dpl
}; // namespace oneapi

template <class T, class = decltype(std::abs(std::declval<T>()))>
std::true_type has_abs_imp(int);
template <class T>
std::false_type has_abs_imp(...);

template <class T>
struct has_abs : decltype(has_abs_imp<T>(0)) {};

ONEDPL_TEST_DECLARE
void test_abs()
{
    // See also "abs.pass.cpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wabsolute-value"
#endif
    static_assert((std::is_same_v<decltype(dpl::abs((float)0)), float>));
    static_assert((std::is_same_v<decltype(dpl::abs((int)0)), int>));
    static_assert((std::is_same_v<decltype(dpl::abs((long)0)), long>));
    static_assert((std::is_same_v<decltype(dpl::abs((long long)0)), long long>));
    static_assert((std::is_same_v<decltype(dpl::abs((unsigned char)0)), int>));
    static_assert((std::is_same_v<decltype(dpl::abs((unsigned short)0)), int>));
    static_assert((std::is_same_v<decltype(dpl::abs((signed char)0)), int>));
    static_assert((std::is_same_v<decltype(dpl::abs((short)0)), int>));
    static_assert((std::is_same_v<decltype(dpl::abs((unsigned char)0)), int>));
    static_assert((std::is_same_v<decltype(dpl::abs((char)0)), int>));
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::abs(Ambiguous())), Ambiguous>)))

    static_assert(!has_abs<unsigned>::value);
    static_assert(!has_abs<unsigned long>::value);
    static_assert(!has_abs<unsigned long long>::value);
    static_assert(!has_abs<size_t>::value);

    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::abs((double)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::abs((long double)0)), long double>)))
#ifdef __clang__
#pragma clang diagnostic pop
#endif

    assert(dpl::abs(-1.f) == 1.f);
    IF_DOUBLE_SUPPORT(assert(dpl::abs(-1.) == 1.));
}

ONEDPL_TEST_DECLARE
void test_ceil()
{
    static_assert((std::is_same_v<decltype(dpl::ceil((float)0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::ceil((bool)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::ceil((unsigned short)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::ceil((int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::ceil((unsigned int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::ceil((long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::ceil((unsigned long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::ceil((long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::ceil((unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::ceil((double)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::ceil((long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::ceil(Ambiguous())), Ambiguous>));
                      assert(dpl::ceil(0) == 0))
}

ONEDPL_TEST_DECLARE
void test_exp()
{
    static_assert((std::is_same_v<decltype(dpl::exp((float)0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::exp((bool)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::exp((unsigned short)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::exp((int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::exp((unsigned int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::exp((long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::exp((unsigned long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::exp((long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::exp((unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::exp((double)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::exp((long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::exp(Ambiguous())), Ambiguous>));
                      assert(dpl::exp(0) == 1))
}

ONEDPL_TEST_DECLARE
void test_fabs()
{
    static_assert((std::is_same_v<decltype(dpl::fabs((float)0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::fabs((bool)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fabs((unsigned short)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fabs((int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fabs((unsigned int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fabs((long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fabs((unsigned long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fabs((long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fabs((unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fabs((double)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::fabs((long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::fabs(Ambiguous())), Ambiguous>));
                      assert(dpl::fabs(-1) == 1));
}

ONEDPL_TEST_DECLARE
void test_floor()
{
    static_assert((std::is_same_v<decltype(dpl::floor((float)0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::floor((bool)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::floor((unsigned short)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::floor((int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::floor((unsigned int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::floor((long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::floor((unsigned long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::floor((long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::floor((unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::floor((double)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::floor((long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::floor(Ambiguous())), Ambiguous>));
                      assert(dpl::floor(1) == 1))
}

ONEDPL_TEST_DECLARE
void test_isgreater()
{
#ifdef isgreater
#error isgreater defined
#endif
    static_assert((std::is_same_v<decltype(dpl::isgreater((float)0, (float)0)), bool>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::isgreater((float)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreater((double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreater((double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreater(0, (double)0)), bool>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::isgreater((float)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreater((double)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreater((long double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreater((long double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreater((long double)0, (long double)0)), bool>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::isgreater(Ambiguous(), Ambiguous())), Ambiguous>));
                      assert(dpl::isgreater(-1.0, 0.F) == false))
}

ONEDPL_TEST_DECLARE
void test_isgreaterequal()
{
#ifdef isgreaterequal
#error isgreaterequal defined
#endif
    static_assert((std::is_same_v<decltype(dpl::isgreaterequal((float)0, (float)0)), bool>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal((float)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal((double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal((double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal(0, (double)0)), bool>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal((float)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal((double)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal((long double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal((long double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isgreaterequal((long double)0, (long double)0)), bool>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::isgreaterequal(Ambiguous(), Ambiguous())), Ambiguous>));
                      assert(dpl::isgreaterequal(-1.0, 0.F) == false))
}

ONEDPL_TEST_DECLARE
void test_isinf()
{
#ifdef isinf
#error isinf defined
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-compare"
#endif

    static_assert((std::is_same_v<decltype(dpl::isinf((float)0)), bool>));
    static_assert((std::is_same_v<decltype(dpl::isinf(0)), bool>));

    if constexpr (HasDoubleSupportInRuntime{})
    {
        typedef decltype(dpl::isinf((double)0)) DoubleRetType;
#if !defined(__linux__) || defined(__clang__)
        static_assert((std::is_same_v<DoubleRetType, bool>));
#else
        // GLIBC < 2.23 defines 'isinf(double)' with a return type of 'int' in
        // all C++ dialects. The test should tolerate this when libc++ can't work
        // around it.
        // See: https://sourceware.org/bugzilla/show_bug.cgi?id=19439
        static_assert((std::is_same_v<DoubleRetType, bool> || std::is_same_v<DoubleRetType, int>));
#endif
    }

    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::isinf((long double)0)), bool>)))
    IF_DOUBLE_SUPPORT(assert(dpl::isinf(-1.0) == false))
#if !_PSTL_ICC_TEST_COMPLEX_ISINF_BROKEN
    assert(dpl::isinf(0) == false);
    assert(dpl::isinf(1) == false);
    assert(dpl::isinf(-1) == false);
    assert(dpl::isinf(std::numeric_limits<int>::max()) == false);
    assert(dpl::isinf(std::numeric_limits<int>::min()) == false);
#endif // !_PSTL_ICC_TEST_COMPLEX_ISINF_BROKEN

#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

ONEDPL_TEST_DECLARE
void test_isless()
{
#ifdef isless
#error isless defined
#endif
    static_assert((std::is_same_v<decltype(dpl::isless((float)0, (float)0)), bool>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::isless((float)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isless((double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isless((double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isless(0, (double)0)), bool>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::isless((float)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isless((double)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isless((long double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isless((long double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isless((long double)0, (long double)0)), bool>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::isless(Ambiguous(), Ambiguous())), Ambiguous>));
                      assert(dpl::isless(-1.0, 0.F) == true))
}

ONEDPL_TEST_DECLARE
void test_islessequal()
{
#ifdef islessequal
#error islessequal defined
#endif
    static_assert((std::is_same_v<decltype(dpl::islessequal((float)0, (float)0)), bool>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::islessequal((float)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::islessequal((double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::islessequal((double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::islessequal(0, (double)0)), bool>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::islessequal((float)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::islessequal((double)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::islessequal((long double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::islessequal((long double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::islessequal((long double)0, (long double)0)), bool>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::islessequal(Ambiguous(), Ambiguous())), Ambiguous>));
                      assert(dpl::islessequal(-1.0, 0.F) == true));
}

ONEDPL_TEST_DECLARE
void test_isnan()
{
#ifdef isnan
#error isnan defined
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-compare"
#endif
    static_assert((std::is_same_v<decltype(dpl::isnan((float)0)), bool>));

    if constexpr (HasDoubleSupportInRuntime{})
    {
        typedef decltype(dpl::isnan((double)0)) DoubleRetType;
#if !defined(__linux__) || defined(__clang__)
        static_assert((std::is_same_v<DoubleRetType, bool>));
#else
        // GLIBC < 2.23 defines 'isinf(double)' with a return type of 'int' in
        // all C++ dialects. The test should tolerate this when libc++ can't work
        // around it.
        // See: https://sourceware.org/bugzilla/show_bug.cgi?id=19439
        static_assert((std::is_same_v<DoubleRetType, bool> || std::is_same_v<DoubleRetType, int>));
#endif
    }

    static_assert((std::is_same_v<decltype(dpl::isnan(0)), bool>));
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::isnan((long double)0)), bool>)))
    IF_DOUBLE_SUPPORT(assert(dpl::isnan(-1.0) == false))
#if !_PSTL_ICC_TEST_COMPLEX_ISNAN_BROKEN
    assert(dpl::isnan(0) == false);
    assert(dpl::isnan(1) == false);
    assert(dpl::isnan(-1) == false);
    assert(dpl::isnan(std::numeric_limits<int>::max()) == false);
    assert(dpl::isnan(std::numeric_limits<int>::min()) == false);
#endif // !_PSTL_ICC_TEST_COMPLEX_ISNAN_BROKEN

#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

ONEDPL_TEST_DECLARE
void test_isunordered()
{
#ifdef isunordered
#error isunordered defined
#endif
    static_assert((std::is_same_v<decltype(dpl::isunordered((float)0, (float)0)), bool>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::isunordered((float)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isunordered((double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isunordered((double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isunordered(0, (double)0)), bool>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::isunordered((float)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isunordered((double)0, (long double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isunordered((long double)0, (float)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isunordered((long double)0, (double)0)), bool>));
        static_assert((std::is_same_v<decltype(dpl::isunordered((long double)0, (long double)0)), bool>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::isunordered(Ambiguous(), Ambiguous())), Ambiguous>));
                      assert(dpl::isunordered(-1.0, 0.F) == false));
}

ONEDPL_TEST_DECLARE
void test_copysign()
{
    static_assert((std::is_same_v<decltype(dpl::copysign((float)0, (float)0)), float>));
    static_assert((std::is_same_v<decltype(dpl::copysign((float)0, (unsigned int)0)), double>));
    static_assert((std::is_same_v<decltype(dpl::copysignf(0,0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::copysign((bool)0, (float)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((unsigned short)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((double)0, (long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((int)0, (unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((double)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((float)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((int)0, (int)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::copysign((int)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((long double)0, (unsigned long)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((int)0, (long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((long double)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((float)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::copysign((double)0, (long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::copysign(Ambiguous(), Ambiguous())), Ambiguous>));
                      assert(dpl::copysign(1,1) == 1))
}

ONEDPL_TEST_DECLARE
void test_fmax()
{
    static_assert((std::is_same_v<decltype(dpl::fmax((float)0, (float)0)), float>));
    static_assert((std::is_same_v<decltype(dpl::fmax((float)0, (unsigned int)0)), double>));
    static_assert((std::is_same_v<decltype(dpl::fmaxf(0,0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::fmax((bool)0, (float)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((unsigned short)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((int)0, (long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((int)0, (unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((double)0, (long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((double)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((float)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((int)0, (int)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::fmax((int)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((long double)0, (unsigned long)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((long double)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((float)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::fmax((double)0, (long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::fmax(Ambiguous(), Ambiguous())), Ambiguous>));
                      assert(dpl::fmax(1,0) == 1))
}

ONEDPL_TEST_DECLARE
void test_fmin()
{
    static_assert((std::is_same_v<decltype(dpl::fmin((float)0, (float)0)), float>));
    static_assert((std::is_same_v<decltype(dpl::fminf(0, 0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::fmin((bool)0, (float)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((unsigned short)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((float)0, (unsigned int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((double)0, (long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((int)0, (unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((double)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((float)0, (double)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((int)0, (int)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::fmin((int)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((long double)0, (unsigned long)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((int)0, (long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((long double)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((float)0, (long double)0)), long double>));
        static_assert((std::is_same_v<decltype(dpl::fmin((double)0, (long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::fmin(Ambiguous(), Ambiguous())), Ambiguous>));
                      assert(dpl::fmin(1,0) == 0))
}

ONEDPL_TEST_DECLARE
void test_nan()
{
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::nan("")), double>)))
    static_assert((std::is_same_v<decltype(dpl::nanf("")), float>));
}

ONEDPL_TEST_DECLARE
void test_round()
{
    static_assert((std::is_same_v<decltype(dpl::round((float)0)), float>));
    static_assert((std::is_same_v<decltype(dpl::roundf(0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::round((bool)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::round((unsigned short)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::round((int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::round((unsigned int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::round((long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::round((unsigned long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::round((unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::round((double)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::round((long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::round((long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::round(Ambiguous())), Ambiguous>));
                      assert(dpl::round(1) == 1))
}

ONEDPL_TEST_DECLARE
void test_trunc()
{
    static_assert((std::is_same_v<decltype(dpl::trunc((float)0)), float>));
    static_assert((std::is_same_v<decltype(dpl::truncf(0)), float>));
    IF_DOUBLE_SUPPORT(
        static_assert((std::is_same_v<decltype(dpl::trunc((bool)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::trunc((unsigned short)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::trunc((int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::trunc((unsigned int)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::trunc((long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::trunc((unsigned long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::trunc((long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::trunc((unsigned long long)0)), double>));
        static_assert((std::is_same_v<decltype(dpl::trunc((double)0)), double>)))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::trunc((long double)0)), long double>)))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same_v<decltype(dpl::trunc(Ambiguous())), Ambiguous>));
                      assert(dpl::trunc(1) == 1))
}

ONEDPL_TEST_NUM_MAIN
{
    ONEDPL_TEST_CALL(test_abs)
    ONEDPL_TEST_CALL(test_ceil)
    ONEDPL_TEST_CALL(test_exp)
    ONEDPL_TEST_CALL(test_fabs)
    ONEDPL_TEST_CALL(test_floor)

    ONEDPL_TEST_CALL(test_isgreater)
    ONEDPL_TEST_CALL(test_isgreaterequal)
    ONEDPL_TEST_CALL(test_isinf)
    ONEDPL_TEST_CALL(test_isless)
    ONEDPL_TEST_CALL(test_islessequal)

    ONEDPL_TEST_CALL(test_isnan)
    ONEDPL_TEST_CALL(test_isunordered)

    ONEDPL_TEST_CALL(test_copysign)

    ONEDPL_TEST_CALL(test_fmax)
    ONEDPL_TEST_CALL(test_fmin)

    ONEDPL_TEST_CALL(test_nan)

    ONEDPL_TEST_CALL(test_round)
    ONEDPL_TEST_CALL(test_trunc)

  return 0;
}
